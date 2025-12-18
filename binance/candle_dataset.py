from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from binance.binance_candle_utils import *
from gm.utils.masking import *


class FinancialDataset(Dataset):
    """
    Хранит уже готовую матрицу признаков [T, D] (после build_feature_matrix)
    и (опционально) сырые OHLCV [T, 5], и нарезает их на перекрывающиеся окна
    длиной <= max_length+1.

    sequence: [L_i, D], где L_i <= max_length+1
    raw_sequence: [L_i, 5] (если задано)

    Потом в FinancialDataProcessor делаем:
      inputs = seq[:-1]
      labels = seq[1:]
      raw_inputs = raw_seq[:-1]
      raw_labels = raw_seq[1:]
    """

    def __init__(
            self, features: torch.Tensor, max_length: int,
            raw_ohlcv: torch.Tensor | None = None
    ):
        """
        features: [T, D] — уже обработанные (дельты/отн. значения) фичи.
        raw_ohlcv: [T, 5] — сырые OHLCV (open, high, low, close, volume),
                   выровненные по времени с features (одинаковая длина T),
                   опционально (может быть None).
        """
        super().__init__()
        self.features = features
        self.max_length = max_length
        self.T = features.size(0)

        if raw_ohlcv is not None:
            if raw_ohlcv.size(0) != self.T:
                raise ValueError(
                    f"raw_ohlcv len={raw_ohlcv.size(0)} != features len={self.T}"
                )
        self.raw_ohlcv = raw_ohlcv

    def __len__(self):
        # Количество начальных индексов, с которых можно взять хотя бы 2 точки
        # (чтобы seq[:-1], seq[1:] не были пустыми).
        return max(0, self.T - 1)

    def __getitem__(self, idx: int):
        """
        Берём окно [idx : idx + max_length + 1], не выходя за пределы T.
        """
        start = idx
        end = min(self.T, idx + self.max_length + 1)
        seq = self.features[start:end]  # [L, D], L <= max_length+1

        out = {"sequence": seq}

        if self.raw_ohlcv is not None:
            raw_seq = self.raw_ohlcv[start:end]  # [L, 5]
            out["raw_sequence"] = raw_seq

        return out


class FinancialDataProcessor:
    """
    collate_fn для числовых данных:
      - превращает список sequence [Li, D] в батч
      - строит inputs, labels как сдвинутые последовательности
      - добавляет padding + attention_mask
      - при наличии raw_sequence строит raw_inputs/raw_labels (сырые OHLCV)
    """

    def __init__(self, max_length: int, device=None, pad_value: float = 0.0):
        self.max_length = max_length
        self.device = device
        self.pad_value = pad_value

    def collate_fn(self, batch):
        """
        batch: list of dicts
          минимум: {"sequence": [L_i, D]}
          опционально: {"raw_sequence": [L_i, 5]}

        Возвращает:
          inputs: [B, L, D]
          labels: [B, L, D]
          attention_mask: [B, 1, L, L]
          lengths: [B]

          + при наличии сырых данных:
            raw_inputs: [B, L, 5]
            raw_labels: [B, L, 5]
        """
        sequences = [b["sequence"] for b in batch]  # list of [L_i, D]

        # режем: inputs=t[0..L-2], labels=t[1..L-1]
        inputs_list = [seq[:-1] for seq in sequences]  # [L_i-1, D]
        labels_list = [seq[1:] for seq in sequences]  # [L_i-1, D]

        lengths = torch.tensor([x.size(0) for x in inputs_list], dtype=torch.long)

        # паддим по длине L
        inputs = pad_sequence(inputs_list, batch_first=True,
                              padding_value=self.pad_value)  # [B, L, D]
        labels = pad_sequence(labels_list, batch_first=True,
                              padding_value=self.pad_value)  # [B, L, D]

        B, L, D = inputs.shape

        device = self.device if self.device is not None else inputs.device
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        # паддинг-маска (0 = pad, 1 = реальное значение)
        token_ids = torch.ones(B, L, dtype=torch.long, device=device)
        for i, length in enumerate(lengths):
            if length < L:
                token_ids[i, length:] = 0

        padding_mask = create_padding_mask(token_ids, pad_token_id=0)  # [B,1,1,L]
        causal_mask = create_causal_mask(L, device=device)  # [1,1,L,L]
        combined_mask = combine_masks(padding_mask, causal_mask)  # [B,1,L,L]

        out = {
            "inputs": inputs,  # [B, L, D]
            "labels": labels,  # [B, L, D]
            "attention_mask": combined_mask,  # [B, 1, L, L]
            "lengths": lengths,  # [B]
        }

        # ----- опционально: сырые OHLCV -----
        if "raw_sequence" in batch[0]:
            raw_sequences = [b["raw_sequence"] for b in batch]  # list of [L_i, 5]

            raw_inputs_list = [rs[:-1] for rs in raw_sequences]  # [L_i-1, 5]
            raw_labels_list = [rs[1:] for rs in raw_sequences]  # [L_i-1, 5]

            raw_inputs = pad_sequence(
                raw_inputs_list, batch_first=True, padding_value=0.0
            )  # [B, L, 5]
            raw_labels = pad_sequence(
                raw_labels_list, batch_first=True, padding_value=0.0
            )  # [B, L, 5]

            out["raw_inputs"] = raw_inputs.to(device)  # [B, L, 5]
            out["raw_labels"] = raw_labels.to(device)  # [B, L, 5]

        return out


class HMoEFinancialDataModule:
    """
    DataModule для финансовых рядов.

    Возможности:
    - build_from_binance(): качает свечи за [start_time, end_time], считает индикаторы,
      формирует features [T, D] и (опционально) сырые OHLCV [T, 5].
    - save(path): сохраняет датасет и метаданные.
    - load(path): загружает датасет и метаданные (например, в Colab без доступа к Binance).
    - get_dataloader(split="train"/"test"): даёт DataLoader для train/test.

    Внутри:
      self.features: [T, D]
      self.raw_ohlcv: [T, 5] или None
      self.feature_cols: list[str]
      self.symbol, self.interval
      self.start_time, self.end_time
      self.test_size: количество последних таймстепов под тест.
    """

    def __init__(
            self,
            symbol: str | None = None,
            interval: str | None = None,
            feature_cols: list[str] | None = None,
            max_length: int = 256,
            batch_size: int = 16,
            device=None,
            start_time: int | None = None,
            end_time: int | None = None,
            test_size: int = 0,
    ):
        self.symbol = symbol
        self.interval = interval
        self.feature_cols = feature_cols  # имена колонок признаков
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

        self.start_time = start_time
        self.end_time = end_time
        self.test_size = test_size  # T_test (кол-во последних точек)

        self.processor = FinancialDataProcessor(max_length=max_length, device=device)

        self.features: torch.Tensor | None = None  # [T, D] — полный ряд
        self.raw_ohlcv: torch.Tensor | None = None  # [T, 5] — сырые OHLCV (open, high, low, close, volume)

        self.train_features: torch.Tensor | None = None  # [T_train, D]
        self.test_features: torch.Tensor | None = None  # [T_test, D]

        self.train_raw_ohlcv: torch.Tensor | None = None  # [T_train, 5]
        self.test_raw_ohlcv: torch.Tensor | None = None  # [T_test, 5]

        self.train_dataset: FinancialDataset | None = None
        self.test_dataset: FinancialDataset | None = None

        self.train_dataloader: DataLoader | None = None
        self.test_dataloader: DataLoader | None = None

        self.df_raw: pd.DataFrame | None = None  # для информации/отладки

    # ---------- Построение из Binance (оффлайн) ----------

    def _load_binance_df(self):
        """
        Твоя логика:
          - BinanceCandleClient.get_klines_range(...)
          - klines_to_df
          - индикаторы
        """
        if self.symbol is None or self.interval is None or self.feature_cols is None:
            raise ValueError("symbol, interval и feature_cols должны быть заданы для build_from_binance")

        client = BinanceCandleClient()

        if self.start_time is None:
            raise ValueError("start_time должен быть задан для build_from_binance с большим числом свечей")

        klines = client.get_klines_range(
            symbol=self.symbol,
            interval=self.interval,
            start_time=self.start_time,
            end_time=self.end_time,
            limit_per_request=1000,
        )

        df = klines_to_df(klines)

        # индикаторы (пример)
        df = add_sma(df, period=20)
        df = add_ema(df, period=50)
        df = add_rsi(df, period=14)
        df = add_macd(df)

        df = df.dropna().reset_index(drop=True)

        return df

    def build_from_binance(self):
        """
        Загрузка OHLCV, индикаторов → преобразование в дельтовые/нормированные фичи
        + подготовка сырых OHLCV, выровненных по времени с фичами.
        """
        df = self._load_binance_df()
        self.df_raw = df

        # Генерируем инженерные фичи (первая строка будет выкинута внутри build_feature_matrix)
        features, feature_names = build_feature_matrix(df)  # [T_feat, D]
        self.features = features
        self.feature_cols = feature_names

        # Выровненные сырые OHLCV: drop первой строки, чтобы длина совпала с features
        df_aligned = df.iloc[1:].reset_index(drop=True)  # длина T_feat
        raw_ohlcv_np = df_aligned[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
        self.raw_ohlcv = torch.from_numpy(raw_ohlcv_np)  # [T_feat, 5]

        # Делим на train/test
        self._split_train_test()

    # ---------- Разбиение на train / test ----------

    def _split_train_test(self):
        """
        Делит self.features (и, если есть, self.raw_ohlcv) на train/test по времени:
          - последние self.test_size точек → test
          - остальное → train

        Если test_size == 0 или >= T, считаем, что теста нет.
        """
        if self.features is None:
            raise ValueError("self.features == None. Сначала собираем или грузим данные.")

        T = self.features.size(0)

        if self.raw_ohlcv is not None and self.raw_ohlcv.size(0) != T:
            raise ValueError("raw_ohlcv длины не совпадают с features")

        if self.test_size <= 0 or self.test_size >= T:
            # всё — train, теста нет
            self.train_features = self.features
            self.test_features = None

            self.train_raw_ohlcv = self.raw_ohlcv
            self.test_raw_ohlcv = None
            return

        split_idx = T - self.test_size

        self.train_features = self.features[:split_idx]  # [T_train, D]
        self.test_features = self.features[split_idx:]  # [T_test, D]

        if self.raw_ohlcv is not None:
            self.train_raw_ohlcv = self.raw_ohlcv[:split_idx]  # [T_train, 5]
            self.test_raw_ohlcv = self.raw_ohlcv[split_idx:]  # [T_test, 5]
        else:
            self.train_raw_ohlcv = None
            self.test_raw_ohlcv = None

    # ---------- save / load ----------

    def save(self, path: str):
        """
        Сохраняет текущий датасет в файл.
        """
        if self.features is None:
            raise ValueError("Нечего сохранять: self.features == None. Сначала вызови build_from_binance() или load().")
        if self.feature_cols is None:
            raise ValueError("feature_cols не заданы, нечего сохранять.")

        obj = {
            "symbol": self.symbol,
            "interval": self.interval,
            "feature_cols": self.feature_cols,
            "features": self.features,
            "max_length": self.max_length,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "test_size": self.test_size,
        }

        if self.raw_ohlcv is not None:
            obj["raw_ohlcv"] = self.raw_ohlcv

        torch.save(obj, path)

    def load(self, path: str, map_location: str | torch.device = "cpu"):
        """
        Загружает features, (опционально) raw_ohlcv и метаданные из файла.
        """
        obj = torch.load(path, map_location=map_location)

        self.symbol = obj.get("symbol", None)
        self.interval = obj.get("interval", None)
        self.feature_cols = obj["feature_cols"]
        self.features = obj["features"]
        self.start_time = obj.get("start_time", None)
        self.end_time = obj.get("end_time", None)
        self.test_size = int(obj.get("test_size", 0))

        raw_ohlcv = obj.get("raw_ohlcv", None)
        self.raw_ohlcv = raw_ohlcv

        file_max_length = obj.get("max_length", None)
        if file_max_length is not None:
            self.max_length = int(file_max_length)
            self.processor.max_length = int(file_max_length)

        # после загрузки сразу делим на train/test
        self._split_train_test()

    # ---------- Dataset / DataLoader ----------

    def setup(self):
        """
        Готовит train/test Dataset'ы из train_features/test_features
        (и, если есть, сырые OHLCV).
        """
        if self.features is None:
            raise ValueError("self.features == None. Вызови build_from_binance() или load(path).")

        if self.train_features is None and self.test_features is None:
            self._split_train_test()

        if self.train_features is not None:
            self.train_dataset = FinancialDataset(
                features=self.train_features,
                max_length=self.max_length,
                raw_ohlcv=self.train_raw_ohlcv,
            )
        else:
            self.train_dataset = None

        if self.test_features is not None:
            self.test_dataset = FinancialDataset(
                features=self.test_features,
                max_length=self.max_length,
                raw_ohlcv=self.test_raw_ohlcv,
            )
        else:
            self.test_dataset = None

    def get_dataloader(self, split: str = "train", shuffle: bool = True) -> DataLoader | None:
        """
        split: "train" или "test"
        """
        if self.train_dataset is None and self.test_dataset is None:
            self.setup()

        if split == "train":
            if self.train_dataset is None:
                return None
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                collate_fn=self.processor.collate_fn,
            )
            return self.train_dataloader

        elif split == "test":
            if self.test_dataset is None:
                return None
            # для теста обычно shuffle=False
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                collate_fn=self.processor.collate_fn,
            )
            return self.test_dataloader

        else:
            raise ValueError(f"split должен быть 'train' или 'test', не '{split}'")
