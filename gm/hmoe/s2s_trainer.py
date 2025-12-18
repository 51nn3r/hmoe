import time
import torch

from gm.utils.model_utils import save_checkpoint


# предполагается, что у тебя где-то уже есть эта функция
# from utils import save_checkpoint


class Seq2SeqTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion_mse,              # основная loss
        criterion_ce=None,          # для NLP (опционально)
        device=None,
        log_interval=100,
        print_usage=False,
        simulate_trading=False,
        close_index: int = 4,
        checkpoint_dir: str = "checkpoints",
        checkpoint_every_n_batches: int = 200,
    ):
        if device is None:
            device = next(model.parameters()).device

        if criterion_mse is None:
            raise ValueError("criterion_mse должен быть задан (например, torch.nn.MSELoss()).")

        self.model = model
        self.optimizer = optimizer
        self.criterion_mse = criterion_mse
        self.criterion_ce = criterion_ce
        self.device = device

        self.log_interval = log_interval
        self.print_usage = print_usage
        self.simulate_trading_flag = simulate_trading
        self.close_index = close_index
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every_n_batches = checkpoint_every_n_batches

    # ===================== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ =====================

    @staticmethod
    def adaptive_chain_penalty(base_penalty, current_loss, loss_threshold=0.1, max_multiplier=10.0):
        if current_loss < loss_threshold:
            return base_penalty
        error_ratio = current_loss / loss_threshold
        multiplier = min(1.0 + (error_ratio - 1.0) * 2.0, max_multiplier)
        return base_penalty * multiplier

    def masked_mse(self, pred, target, lengths: torch.Tensor | None):
        """
        pred, target: [B, L, D]
        lengths: [B] или None
        """
        if lengths is None:
            return self.criterion_mse(pred, target)

        B, L, D = pred.shape
        device_ = pred.device

        time_ids = torch.arange(L, device=device_).unsqueeze(0)  # [1, L]
        mask = time_ids < lengths.view(-1, 1)                    # [B, L]
        mask_expanded = mask.unsqueeze(-1)                       # [B, L, 1]

        diff2 = (pred - target) ** 2
        diff2 = diff2 * mask_expanded

        denom = mask_expanded.sum().clamp_min(1.0)
        return diff2.sum() / denom

    def _prepare_batch(self, batch: dict):
        """
        Приводим batch к единому формату:
        inputs, attn_mask, targets, lengths (все уже на нужном устройстве).
        """
        if "inputs" in batch:  # числовые ряды
            inputs = batch["inputs"].to(self.device)
        elif "input_ids" in batch:  # NLP
            inputs = batch["input_ids"].to(self.device)
        else:
            raise KeyError("В batch нет ни 'inputs', ни 'input_ids'")

        attn_mask = batch.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device)

        if "labels" not in batch:
            raise KeyError("В batch нет ключа 'labels'")
        targets = batch["labels"].to(self.device)

        lengths = batch.get("lengths", None)
        if lengths is not None:
            lengths = lengths.to(self.device)

        return inputs, attn_mask, targets, lengths

    def _forward_model(self, inputs, attn_mask):
        if attn_mask is not None:
            res = self.model(inputs, attn_mask)
        else:
            res = self.model(inputs)

        outputs = res["out"]
        gates_output = res.get("gates_out", None)
        exp_usage = res.get("exp_usage", None)
        return outputs, gates_output, exp_usage

    def _compute_main_loss(self, outputs, targets, lengths):
        """
        Содержит ветвление между:
        - числовым seq2seq (MSE / masked MSE),
        - NLP (CE),
        - запасной вариант (просто MSE).
        """
        # Числовая регрессия (seq2seq по фичам)
        if outputs.dim() == 3 and targets.dim() == 3:
            if hasattr(self.model, "normalize_features"):
                targets_norm = self.model.normalize_features(targets)
            else:
                targets_norm = targets
            loss = self.masked_mse(outputs, targets_norm, lengths)

        # NLP путь (CE по токенам)
        elif outputs.dim() == 3 and targets.dim() == 2 and self.criterion_ce is not None:
            # [B, L, V] vs [B, L]
            logits = outputs.view(-1, outputs.size(-1))  # [B*L, V]
            labels_flat = targets.view(-1)               # [B*L]
            loss = self.criterion_ce(logits, labels_flat)

        else:
            # запасной вариант: без маски, просто MSE
            loss = self.criterion_mse(outputs, targets)

        return loss

    def _apply_expert_regularization(self, loss, gates_output, exp_usage):
        """
        Возвращает:
          loss_with_reg, gate_penalty_loss_value, exp_usage_loss_value
        Чтобы их можно было логировать отдельно.
        """
        exp_usage_loss = 0.0
        if exp_usage is not None and hasattr(exp_usage, "loss_sum"):
            exp_usage_loss = exp_usage.loss_sum * 1e-3
            loss = loss + exp_usage_loss

        gate_penalty_loss = 0.0
        if gates_output is not None:
            gate_penalty_loss = self.criterion_mse(gates_output, torch.zeros_like(gates_output))
            adaptive_penalty = self.adaptive_chain_penalty(gate_penalty_loss.item(), loss.item())
            gate_penalty_loss = gate_penalty_loss * adaptive_penalty
            # если захочешь включить её в общий loss:
            # loss = loss + gate_penalty_loss * 1e-3

        return loss, gate_penalty_loss, exp_usage_loss

    def _log_usage_if_needed(self):
        if not self.print_usage:
            return

        if hasattr(self.model, "experts_usage_stat"):
            print(f"experts usage: {self.model.experts_usage_stat}")
            self.model.experts_usage_stat = torch.zeros_like(self.model.experts_usage_stat)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "experts_usage_stat"):
            print(f"experts usage: {self.model.model.experts_usage_stat}")
            self.model.model.experts_usage_stat = torch.zeros_like(self.model.model.experts_usage_stat)

    def _save_batch_checkpoint(self, epoch, batch_idx, inputs, attn_mask):
        if self.checkpoint_every_n_batches <= 0:
            return
        if batch_idx % self.checkpoint_every_n_batches != 0 or batch_idx == 0:
            return

        with torch.no_grad():
            out = self.model(inputs, attn_mask) if attn_mask is not None else self.model(inputs)
        save_checkpoint(
            f'{self.checkpoint_dir}/tm_e{epoch}_{batch_idx}.pt',
            self.model,
            self.optimizer,
            {'i': inputs, 'm': attn_mask, 'o': out},
        )

    def _save_epoch_checkpoint(self, epoch, last_inputs, last_attn_mask):
        with torch.no_grad():
            out = (
                self.model(last_inputs, last_attn_mask)
                if last_attn_mask is not None
                else self.model(last_inputs)
            )
        save_checkpoint(
            f'{self.checkpoint_dir}/tm_e{epoch}.pt',
            self.model,
            self.optimizer,
            {'i': last_inputs, 'm': last_attn_mask, 'o': out},
        )

    # ===================== ЦИКЛ ОБУЧЕНИЯ / EVAL =====================

    def _run_epoch(self, dataloader, epoch, train: bool):
        """
        Один проход по даталоадеру.
        Возвращает:
          avg_loss, avg_gate_penalty, avg_batch_time, total_time, last_inputs, last_attn_mask
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        epoch_start = time.time()
        running_loss = 0.0
        running_gate_penalty = 0.0
        batch_times = []

        last_inputs = None
        last_attn_mask = None

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for i, batch in enumerate(dataloader):
                batch_start = time.time()

                inputs, attn_mask, targets, lengths = self._prepare_batch(batch)
                last_inputs, last_attn_mask = inputs, attn_mask

                if train:
                    self.optimizer.zero_grad()

                outputs, gates_output, exp_usage = self._forward_model(inputs, attn_mask)

                loss = self._compute_main_loss(outputs, targets, lengths)
                out_loss = loss.item()

                loss, gate_penalty_loss, exp_usage_loss = self._apply_expert_regularization(
                    loss, gates_output, exp_usage
                )

                if train:
                    loss.backward()
                    self.optimizer.step()

                batch_time = time.time() - batch_start
                batch_times.append(batch_time)

                running_loss += out_loss
                running_gate_penalty += gate_penalty_loss.item() if gates_output is not None else 0.0

                # Логирование
                if train and (i % self.log_interval == 0):
                    print(
                        f'[{ "train" if train else "eval" }] epoch {epoch} | '
                        f'batch {i}/{len(dataloader)} | '
                        f'loss={out_loss:.6f} | '
                        f'gate_usage_loss={float(exp_usage_loss):.6f} | '
                        f'batch_time={batch_time:.4f}s'
                    )
                    self._log_usage_if_needed()

                # чекпоинт по батчам только в train
                if train:
                    self._save_batch_checkpoint(epoch, i, inputs, attn_mask)

        epoch_time = time.time() - epoch_start
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_loss = running_loss / len(dataloader)
        avg_gate_penalty = running_gate_penalty / len(dataloader)

        return avg_loss, avg_gate_penalty, avg_batch_time, epoch_time, last_inputs, last_attn_mask

    # ===================== SIM TRADING =====================

    def _simulate_trading(self, test_dataloader, epoch, epochs):
        """
        Вынесена твоя логика симуляции в отдельный метод.
        Предполагается, что test_dataloader.shuffle=False.
        """
        if not self.simulate_trading_flag:
            return

        self.model.eval()
        equity_curve = []
        capital = 1.0

        with torch.no_grad():
            for batch in test_dataloader:
                inputs, attn_mask, targets, lengths = self._prepare_batch(batch)

                outputs, _, _ = self._forward_model(inputs, attn_mask)

                # денормализация предсказаний, если модель её умеет
                if hasattr(self.model, "denormalize_features"):
                    outputs_den = self.model.denormalize_features(outputs)
                else:
                    outputs_den = outputs

                inputs_den = inputs
                targets_den = targets

                # предполагается, что close_index указывает на "цену"
                real_close_curr = inputs_den[..., 1]       # [B, L]
                real_close_true_next = targets_den[..., 1] # [B, L]
                close_curr = inputs_den[..., self.close_index]      # [B, L]
                close_true_next = targets_den[..., self.close_index] # [B, L]
                close_pred_next = outputs_den[..., self.close_index] # [B, L]

                B, L = close_curr.shape

                for b in range(B):
                    # последний валидный timestep
                    if lengths is not None:
                        T = int(lengths[b].item())
                        if T <= 0:
                            continue
                        t_last = T - 1
                    else:
                        valid_mask = torch.any(targets[b] != 0, dim=-1)
                        if not valid_mask.any():
                            continue
                        t_last = valid_mask.nonzero(as_tuple=False)[-1].item()

                    curr = close_curr[b, t_last].item()
                    true_next = close_true_next[b, t_last].item()
                    pred_next = close_pred_next[b, t_last].item()
                    real_curr = real_close_curr[b, t_last].item()
                    real_true_next = real_close_true_next[b, t_last].item()

                    true_delta = true_next - curr
                    pred_delta = pred_next - curr

                    if pred_delta > 0:
                        # LONG
                        capital *= (real_true_next / real_curr)
                        print(
                            f'[LONG] cap={capital:.4f}; curr={curr:.4f}; '
                            f'pred_next={pred_next:.4f}; pred_delta={pred_delta:.4f}; '
                            f'true_next={true_next:.4f}; true_delta={true_delta:.4f}; '
                            f'real_curr={real_curr:.4f}; '
                            f'real_true_next={real_true_next:.4f}'
                        )
                        equity_curve.append(capital)

                    elif pred_delta < 0:
                        # SHORT
                        capital *= (real_curr / real_true_next)
                        print(
                            f'[SHORT] cap={capital:.4f}; curr={curr:.4f}; '
                            f'pred_next={pred_next:.4f}; pred_delta={pred_delta:.4f}; '
                            f'true_next={true_next:.4f}; true_delta={true_delta:.4f}; '
                            f'real_curr={real_curr:.4f}; '
                            f'real_true_next={real_true_next:.4f}'
                        )
                        equity_curve.append(capital)

        if len(equity_curve) > 0:
            n_steps = len(equity_curve)
            print(f'\n[sim] epoch {epoch}/{epochs} | steps={n_steps}')
            for k in range(1, 11):
                idx = int(n_steps * k / 10) - 1
                if idx < 0 or idx >= n_steps:
                    continue
                eq = equity_curve[idx]
                pnl = eq - 1.0
                print(f'[sim] {k * 10:3d}% | equity={eq:.6f} | pnl={pnl:+.6f}')
            final_eq = equity_curve[-1]
            final_pnl = final_eq - 1.0
            print(f'[sim] FINAL | equity={final_eq:.6f} | pnl={final_pnl:+.6f}\n')

    # ===================== ПУБЛИЧНЫЙ ИНТЕРФЕЙС =====================

    def fit(self, train_dataloader, epochs, test_dataloader=None):
        for epoch in range(1, epochs + 1):
            # ---------- TRAIN ----------
            train_loss, train_gate_penalty, train_avg_bt, train_time, last_inputs, last_attn_mask = \
                self._run_epoch(train_dataloader, epoch, train=True)

            print(
                f'\n[train] epoch {epoch}/{epochs} | '
                f'avg_loss={train_loss:.6f} | '
                f'avg_gate_penalty={train_gate_penalty:.6f} | '
                f'epoch_time={train_time:.2f}s | '
                f'avg_batch_time={train_avg_bt:.4f}s'
            )

            # ---------- EVAL ----------
            if test_dataloader is not None:
                test_loss, test_gate_penalty, test_avg_bt, test_time, _, _ = \
                    self._run_epoch(test_dataloader, epoch, train=False)

                print(
                    f'[eval]  epoch {epoch}/{epochs} | '
                    f'avg_loss={test_loss:.6f} | '
                    f'avg_gate_penalty={test_gate_penalty:.6f} | '
                    f'total_time={test_time:.2f}s | '
                    f'avg_batch_time={test_avg_bt:.4f}s'
                )

            # ---------- SIM TRADING ----------
            if test_dataloader is not None and self.simulate_trading_flag:
                self._simulate_trading(test_dataloader, epoch, epochs)

            # ---------- CHECKPOINT ----------
            if last_inputs is not None:
                self._save_epoch_checkpoint(epoch, last_inputs, last_attn_mask)

            # ---------- GPU STATS ----------
            if torch.cuda.is_available():
                print(f'GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB\n')
