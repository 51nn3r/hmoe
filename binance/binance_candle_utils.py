import time

import requests
from typing import List, Any, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class BinanceCandleClient:
    BASE_URL = "https://api.binance.com"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def get_klines(
            self,
            symbol: str,
            interval: str,
            limit: int = 500,
            start_time: Optional[int] = None,
            end_time: Optional[int] = None,
    ) -> List[List[Any]]:
        if not (1 <= limit <= 1000):
            raise ValueError(f"limit должен быть в диапазоне 1..1000, не {limit}")

        url = f"{self.BASE_URL}/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_klines_range(
            self,
            symbol: str,
            interval: str,
            start_time: int,
            end_time: Optional[int] = None,
            limit_per_request: int = 1000,
            sleep_sec: float = 0.2,
    ) -> List[List[Any]]:
        """
        Качаем много свечей кусками по limit_per_request, пока не дойдём до end_time
        или пока Binance не вернет пустой список.

        start_time / end_time — в миллисекундах (Unix time).
        """
        all_klines: List[List[Any]] = []
        current_start = start_time

        while True:
            klines = self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit_per_request,
                start_time=current_start,
                end_time=end_time,
            )

            if not klines:
                break

            all_klines.extend(klines)

            # последний open_time в текущей партии
            last_open_time = klines[-1][0]

            # если указали end_time и мы уже его пересекли — выходим
            if end_time is not None and last_open_time >= end_time:
                break

            # следующий запрос начинаем после последней свечи
            current_start = last_open_time + 1

            # чуть спим, чтобы не долбить API
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        return all_klines


def klines_to_df(klines: List[List[Any]]) -> pd.DataFrame:
    """
    Преобразование ответов /api/v3/klines в DataFrame.
    """
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)

    # типы
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    float_cols = ["open", "high", "low", "close", "volume",
                  "quote_asset_volume", "taker_buy_base_asset_volume",
                  "taker_buy_quote_asset_volume"]
    df[float_cols] = df[float_cols].astype(float)

    df["number_of_trades"] = df["number_of_trades"].astype(int)

    return df


# ===== Примеры индикаторов на pandas =====

def add_sma(df: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.DataFrame:
    df[f"sma_{period}"] = df[price_col].rolling(window=period, min_periods=period).mean()
    return df


def add_ema(df: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.DataFrame:
    df[f"ema_{period}"] = df[price_col].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, price_col: str = "close") -> pd.DataFrame:
    delta = df[price_col].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_ema = pd.Series(gain, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    loss_ema = pd.Series(loss, index=df.index).ewm(alpha=1 / period, adjust=False).mean()

    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df[f"rsi_{period}"] = rsi
    return df


def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        price_col: str = "close",
) -> pd.DataFrame:
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line

    df["macd"] = macd
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    return df


# ===== Преобразование в torch.Tensor и Dataset =====

class CandleIndicatorDataset(Dataset):
    """
    Dataset, который:
    - берет DataFrame с ценами и индикаторами,
    - создаёт окна длины window_size по признакам feature_cols,
    - таргет — следующий close (или что угодно из df[target_col]).
    """

    def __init__(
            self,
            df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str = "close",
            window_size: int = 50,
            dropna: bool = True,
    ):
        if dropna:
            df = df.dropna().reset_index(drop=True)

        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.window_size = window_size

        # numpy массивы
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)

    def __len__(self) -> int:
        # последнее окно будет иметь таргет "следующая точка"
        return len(self.df) - self.window_size

    def __getitem__(self, idx: int):
        """
        X: окно [window_size, num_features]
        y: скаляр (следующая цена/таргет)
        """
        start = idx
        end = idx + self.window_size

        x_window = self.features[start:end]  # shape: (window_size, num_features)
        # таргет — значение после окна
        y_value = self.targets[end]

        x_tensor = torch.tensor(x_window, dtype=torch.float32)
        y_tensor = torch.tensor(y_value, dtype=torch.float32)

        return x_tensor, y_tensor


def build_feature_matrix(df: pd.DataFrame):
    """
    Превращает OHLCV + индикаторы в дельтовые/нормированные фичи.
    Все большие значения → относительные величины (log-return и ratios).
    """

    eps = 1e-8

    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_volume = np.concatenate([[np.nan], volume[:-1]])

    # ----------- Basic price deltas -----------
    log_ret_open_close = np.log((close + eps) / (open_ + eps))

    high_rel_close = (high - close) / (close + eps)
    low_rel_close = (low - close) / (close + eps)

    # ----------- Volume -----------
    log_vol_rel = np.log((volume + 1.0) / (prev_volume + 1.0))

    # ----------- Indicators → relative/scaled -----------
    sma_20 = df.get("sma_20", pd.Series([np.nan] * len(df))).to_numpy(dtype=np.float64)
    ema_50 = df.get("ema_50", pd.Series([np.nan] * len(df))).to_numpy(dtype=np.float64)
    rsi = df.get("rsi_14", pd.Series([50] * len(df))).to_numpy(dtype=np.float64)
    macd = df.get("macd", pd.Series([0] * len(df))).to_numpy(dtype=np.float64)
    macd_signal = df.get("macd_signal", pd.Series([0] * len(df))).to_numpy(dtype=np.float64)
    macd_hist = df.get("macd_hist", pd.Series([0] * len(df))).to_numpy(dtype=np.float64)

    sma20_rel = sma_20 / (close + eps) - 1.0
    ema50_rel = ema_50 / (close + eps) - 1.0

    rsi_centered = (rsi - 50.0) / 50.0  # [-1..1]

    macd_rel = macd / (close + eps)
    macd_signal_rel = macd_signal / (close + eps)
    macd_hist_rel = macd_hist / (close + eps)

    # ----------- pack -----------
    feats = np.stack([
        log_ret_open_close,
        high_rel_close,
        low_rel_close,
        log_vol_rel,
        sma20_rel,
        ema50_rel,
        rsi_centered,
        macd_rel,
        macd_signal_rel,
        macd_hist_rel,
    ], axis=-1)

    feature_names = [
        "log_ret_open_close",
        "high_rel_close",
        "low_rel_close",
        "log_vol_rel",
        "sma20_rel",
        "ema50_rel",
        "rsi_centered",
        "macd_rel",
        "macd_signal_rel",
        "macd_hist_rel"
    ]

    # выкидываем первую NaN-строку
    mask = ~np.isnan(log_ret_open_close)
    feats = feats[mask]

    return torch.from_numpy(feats).float(), feature_names


# ===== Пример использования всего пайплайна =====

if __name__ == "__main__":
    client = BinanceCandleClient()

    # Интервал: последние 30 дней в миллисекундах
    end = int(time.time() * 1000)
    start = int((time.time() - 365 * 24 * 60 * 60) * 1000)

    # 1. Берём свечи за интервал
    klines = client.get_klines_range(
        symbol="BTCUSDT",
        interval="1h",
        start_time=start,
        end_time=end,
        limit_per_request=1000,
        sleep_sec=0.2,
    )

    # 2. В DataFrame
    df = klines_to_df(klines)

    # 3. Добавляем индикаторы
    df = add_sma(df, period=20)
    df = add_ema(df, period=50)
    df = add_rsi(df, period=14)
    df = add_macd(df)

    # 4. Выбираем признаки
    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "sma_20",
        "ema_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
    ]

    dataset = CandleIndicatorDataset(
        df,
        feature_cols=feature_cols,
        target_col="close",  # можно поменять на лог-доходность и т.п.
        window_size=50,
    )

    print("Размер датасета:", len(dataset))
    print(df.head())
