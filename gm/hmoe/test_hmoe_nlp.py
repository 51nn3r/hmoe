import os
import time

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import math
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Any, Tuple

from gm.hmoe.embeddings_wrapper import EmbeddingsWrapper
from gm.hmoe.hierarchical_moe import HierarchicalMoE
from gm.hmoe.scenario import Scenario
from gm.utils.masking import *


def fit(
        model, dataloader, epochs, optimizer, criterion_ce, criterion_mse=None, device=None, log_interval=100,
        print_usage=False, save_path=None,
):
    """
    Universal training function for a model with a new data format
    """

    def adaptive_chain_penalty(base_penalty, current_loss, loss_threshold=0.1, max_multiplier=10.0):
        """
        Adaptive penalty: the base penalty is multiplied by a factor depending on the current loss
        """
        if current_loss < loss_threshold:
            return base_penalty  # minimal penalty for small loss

        # Increase penalty proportionally to threshold overflow
        error_ratio = current_loss / loss_threshold
        multiplier = min(1.0 + (error_ratio - 1.0) * 2.0, max_multiplier)

        return base_penalty * multiplier

    if device is None:
        device = next(model.parameters()).device

    model.train()

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        running_perplexity = 0.0
        running_gate_penalty = 0.0
        batch_times = []

        for i, batch in enumerate(dataloader):
            batch_start = time.time()

            # Extract data from batch
            inputs = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
            targets = batch['labels'].to(device)

            optimizer.zero_grad()

            # Forward pass
            start_forward = time.time()
            if attn_mask is not None:
                res = model(inputs, attn_mask)
            else:
                res = model(inputs)

            outputs = res['out']
            gates_output = res['gates_out']
            exp_usage = res['exp_usage']
            # Loss computation
            if outputs.dim() == 3 and targets.dim() == 2:  # for language modeling
                loss = criterion_ce(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            else:
                loss = criterion_ce(outputs, targets)

            out_loss = loss.item()

            # Perplexity computation
            with torch.no_grad():
                perplexity = torch.exp(loss)

            exp_usage_loss = exp_usage.loss_sum / 5  # @TODO: get rid of the coefficient
            loss += exp_usage_loss

            # Regularization (if gates_output is present)
            gate_penalty_loss = 0.0
            if gates_output is not None and criterion_mse is not None:
                gate_penalty_loss = criterion_mse(gates_output, torch.zeros_like(gates_output))
                adaptive_penalty = adaptive_chain_penalty(gate_penalty_loss.item(), loss.item())
                gate_penalty_loss = gate_penalty_loss * adaptive_penalty
                # loss = loss + gate_penalty_loss

            # Backward pass
            loss.backward()

            optimizer.step()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            running_loss += out_loss
            running_perplexity += perplexity.item()
            running_gate_penalty += gate_penalty_loss.item() if gates_output is not None else 0.0

            # Logging
            if i % log_interval == 0:
                gate_info = f" | Gate penalty: {exp_usage_loss.item():.6f}" if gates_output is not None else ""
                print(f'Batch {i} / {len(dataloader)} | '
                      f'Loss: {out_loss:.6f} | '
                      f'Perplexity: {perplexity.item():.2f}{gate_info} | ')

                # Experts usage statistics (if present)
                if print_usage:
                    if hasattr(model, 'experts_usage_stat'):
                        print(f"Experts usage: {model.experts_usage_stat}")
                        model.experts_usage_stat = torch.zeros_like(model.experts_usage_stat)
                    elif hasattr(model, 'model') and hasattr(model.model, 'experts_usage_stat'):
                        # For cases when the model is wrapped in EmbeddingsWrapper
                        print(f"Experts usage: {model.model.experts_usage_stat}")
                        model.model.experts_usage_stat = torch.zeros_like(model.model.experts_usage_stat)

        # Epoch statistics
        epoch_time = time.time() - epoch_start
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_loss = running_loss / len(dataloader)
        avg_perplexity = running_perplexity / len(dataloader)
        avg_gate_penalty = running_gate_penalty / len(dataloader)

        print(f'\nEpoch {epoch + 1}/{epochs} | '
              f'Avg Loss: {avg_loss:.6f} | '
              f'Avg Perplexity: {avg_perplexity:.2f} | '
              f'Avg Gate Penalty: {avg_gate_penalty:.6f} | '
              f'Epoch time: {epoch_time:.2f}s | '
              f'Avg batch time: {avg_batch_time:.4f}s\n')

        if save_path:
            torch.save(model.state_dict(), f'{save_path}_{epoch}')

        # GPU memory measurement
        if torch.cuda.is_available():
            print(f'GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB')


class DataProcessor:
    def __init__(self, tokenizer, max_length=256, device=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def tokenize_function(self, examples):
        """Tokenization function"""
        tokens = self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.max_length + 1,
            padding=False,
        )
        return {'input_ids': tokens['input_ids']}

    def collate_fn(self, batch):
        """Function for creating batches with device awareness"""
        sequences = [torch.tensor(x['input_ids'][:-1]) for x in batch]
        labels = [torch.tensor(x['input_ids'][1:]) for x in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        # Create masks on the same device as input_ids
        padding_mask = create_padding_mask(input_ids, self.tokenizer.pad_token_id)
        causal_mask = create_causal_mask(input_ids.size(1), input_ids.device)
        combined_mask = combine_masks(padding_mask, causal_mask)

        # Move data to the specified device if provided
        if self.device is not None:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            combined_mask = combined_mask.to(self.device)

        return {
            'input_ids': input_ids,
            'attention_mask': combined_mask,
            'labels': labels
        }


class HMoEDataModule:
    def __init__(self, dataset_name, max_length=256, batch_size=8, device=None):
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize data processor with device
        self.processor = DataProcessor(self.tokenizer, max_length, device)

    def setup(self):
        """Load and prepare data"""
        # Load dataset
        self.dataset = load_dataset(self.dataset_name, split="train[:2000]")

        # Tokenize
        self.tokenized_dataset = self.dataset.map(
            self.processor.tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names,
        )

        splits = self.tokenized_dataset.train_test_split(test_size=0.1)

        self.train_ds = splits["train"]
        self.test_ds = splits["test"]

    def get_dataloader(self):
        """Create DataLoader"""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.processor.collate_fn,
            shuffle=True,
        )

    def get_test_dataloader(self):
        """Create DataLoader"""
        return DataLoader(
            self.test_ds,
            batch_size=1,
            collate_fn=self.processor.collate_fn,
            shuffle=False,
        )

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size


def get_test_scenario():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # d_model = 768
    mem_vectors = 64
    d_model = 128
    dim_feedforward = d_model * 4
    num_heads = 4
    train_size = 1000
    batch_size = 8
    # time_steps = 256
    time_steps = 64
    top_k = 4
    lr = 5e-4

    data_module = HMoEDataModule(
        dataset_name="roneneldan/TinyStories",
        max_length=time_steps,
        batch_size=batch_size,
        device=device,
    )
    data_module.setup()
    dataloader = data_module.get_dataloader()
    test_dataloader = data_module.get_test_dataloader()

    print('[*] init model')
    inner_model, experts_storage = HierarchicalMoE.create_hierarchical_moe(
        experts_count=100,
        chain_sizes=[4, 8, 16],
        top_k=top_k,
        tau=0.25,
        num_heads=num_heads,
        mem_vectors=mem_vectors,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
    )
    model = EmbeddingsWrapper(inner_model, data_module.vocab_size, d_model)
    model.to(device)

    MODEL_SAVE_PATH = 'models/hmoe_memory_expert_3.pt'
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[+] Loading model state from {MODEL_SAVE_PATH}")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        batch = next(iter(test_dataloader))
        inputs = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
        targets = batch['labels'].to(device)
        model(inputs, attn_mask)
        model.load_state_dict(checkpoint)
    else:
        print("[*] No saved state found, training model...")

    print(f'[+] HMoE params count: {sum(p.numel() for p in inner_model.parameters())}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # experts_storage.disable_grad()

    '''
    fit(model, dataloader, optimizer=optimizer, criterion_ce=nn.CrossEntropyLoss(),
        criterion_mse=nn.MSELoss(), epochs=10, log_interval=5, device=device, print_usage=True)
    '''

    batch = next(iter(test_dataloader))
    inputs = batch['input_ids'].to(device)
    attn_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
    targets = batch['labels'].to(device)
    res = model(inputs, attn_mask, create_scenario=True)
    scenario: Scenario = res['scenario']

    model.eval()

    return model, scenario


def safe_load_checkpoint(model: nn.Module, path: str, device):
    """
    Попытка корректно загрузить checkpoint в модель.
    Возвращает True если успешно применено (частично/полностью).
    """
    if not os.path.exists(path):
        print(f"[!] Snapshot not found: {path}")
        return False
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        print(f"[!] Error loading {path}: {e}")
        return False

    # Иногда нужно пробросить dummy forward чтобы инициализировать веса/модули
    model.zero_grad()
    try:
        # if there is a test batch available, do a dummy forward
        # We avoid assuming global variables: user will pass a batch if needed outside
        # Here just try a tiny action to avoid failure.
        # This call may be skipped by caller if they already did forward earlier.
        pass
    except Exception:
        pass

    try:
        model.load_state_dict(checkpoint, strict=False)
        return True
    except Exception as e:
        print(f"[!] load_state_dict failed for {path} with strict=False: {e}")
        try:
            # fallback: try assigning keys manually where shapes match
            sd = checkpoint
            own_state = model.state_dict()
            for name, param in sd.items():
                if name in own_state and own_state[name].size() == param.size():
                    own_state[name].copy_(param)
            model.load_state_dict(own_state)
            return True
        except Exception as e2:
            print(f"[!] fallback load failed: {e2}")
            return False


def evaluate_model_on_batch(model: nn.Module, batch: Dict[str, torch.Tensor], device, top_k=5):
    """
    Прямой прогон одного батча через модель (eval mode) и вычисление метрик per-position.
    Ожидает, что batch содержит 'input_ids', 'attention_mask' и 'labels'
    Возвращает словарь со следующими ключами:
        - pos_max_prob: np.array (seq_len,) averaged over batch
        - pos_entropy: np.array (seq_len,) averaged over batch
        - pos_topk_acc: np.array (seq_len,) averaged over batch (fraction)
        - topk_tokens_per_pos: list of lists (len=seq_len) — top-k token ids for first sample
    """
    model.eval()
    with torch.no_grad():
        inputs = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
        labels = batch['labels'].to(device)

        # forward
        if attn_mask is not None:
            res = model(inputs, attn_mask)
        else:
            res = model(inputs)

        logits = res['out']  # shape [B, L, V] assumed
        if logits is None:
            raise RuntimeError("Model forward did not return 'out' logits")

        probs = torch.softmax(logits, dim=-1)  # [B, L, V]

        # compute per-position max prob averaged across batch
        pos_max_prob = probs.max(dim=-1)[0]  # [B, L]
        pos_max_prob_mean = pos_max_prob.mean(dim=0).cpu().numpy()  # [L]

        # entropy per position
        p_safe = probs.clamp(min=1e-12)
        entropy = - (p_safe * torch.log(p_safe)).sum(dim=-1)  # [B, L]
        pos_entropy_mean = entropy.mean(dim=0).cpu().numpy()

        # top-k accuracy per position (ignoring label padding -100)
        B, L = labels.shape
        topk = torch.topk(probs, k=top_k, dim=-1).indices  # [B, L, top_k]
        # labels: shape [B, L], where padded positions == -100
        pos_topk_hits = torch.zeros(B, L, device=device)
        valid_mask = (labels != -100)
        for b in range(B):
            for pos in range(L):
                if not valid_mask[b, pos]:
                    pos_topk_hits[b, pos] = float('nan')
                    continue
                true_tok = labels[b, pos].item()
                topk_list = topk[b, pos].tolist()
                pos_topk_hits[b, pos] = 1.0 if (true_tok in topk_list) else 0.0
        # compute mean ignoring nan
        pos_topk_acc = []
        for pos in range(L):
            col = pos_topk_hits[:, pos]
            valid = ~torch.isnan(col)
            if valid.sum() == 0:
                pos_topk_acc.append(float('nan'))
            else:
                pos_topk_acc.append(col[valid].mean().item())
        pos_topk_acc = np.array(pos_topk_acc)

        # top-k tokens per position for the first sample (for qualitative printouts)
        topk_tokens_first = topk[0].cpu().numpy().tolist()  # [L, top_k]

    return {
        'pos_max_prob': pos_max_prob_mean,
        'pos_entropy': pos_entropy_mean,
        'pos_topk_acc': pos_topk_acc,
        'topk_tokens_first': topk_tokens_first,
        'logits': logits.cpu()  # optionally saved if needed
    }


def build_metrics_for_snapshots(
        model_factory,  # callable -> (model, tokenizer, test_dataloader, device)
        snapshot_pattern: str,  # e.g. 'models/hmoe_memory_expert.pt_{}'
        epochs_range: List[int],  # list of epoch indices to check
        out_dir: str = 'analysis',
        n_eval_batches: int = 8,
        top_k: int = 5,
        conv_thresh: float = 0.10
):
    """
    Основная функция: для каждого снапшота загружает веса, прогоняет n_eval_batches из теста,
    накапливает per-position метрики (усреднённые по батчам), сохраняет CSV и графики.
    model_factory должен возвращать (model, tokenizer, test_dataloader, device)
    """

    os.makedirs(out_dir, exist_ok=True)
    # собираем dataframe rows
    rows = []
    snapshots_found = []

    # Создадим модель один раз через фабрику (но будем пере-загружать веса)
    model, tokenizer, test_dataloader, device = model_factory()
    model.to(device)

    # Список батчей для оценки (копируем из тест_dataloader первые n_eval_batches)
    eval_batches = []
    for i, batch in enumerate(test_dataloader):
        eval_batches.append(batch)
        if i + 1 >= n_eval_batches:
            break
    if not eval_batches:
        raise RuntimeError("Test dataloader is empty; cannot evaluate snapshots")

    for epoch_idx in epochs_range:
        path = snapshot_pattern.format(epoch_idx)
        print(f"\n[*] Processing snapshot: {path}")
        ok = safe_load_checkpoint(model, path, device)
        if not ok:
            print(f"[!] Skipping snapshot {path}")
            continue
        snapshots_found.append(epoch_idx)

        # Ensure model in eval
        model.eval()
        # Evaluate across N batches and average metrics per position
        accum_max = []
        accum_ent = []
        accum_topk = []
        topk_tokens_examples = None

        for batch in eval_batches:
            try:
                metrics = evaluate_model_on_batch(model, batch, device, top_k=top_k)
            except Exception as e:
                print(f"[!] Error evaluating batch on {path}: {e}")
                continue
            accum_max.append(metrics['pos_max_prob'])
            accum_ent.append(metrics['pos_entropy'])
            accum_topk.append(metrics['pos_topk_acc'])
            if topk_tokens_examples is None:
                topk_tokens_examples = metrics['topk_tokens_first']

        # Convert to numpy arrays and average over batches (axis=0: positions)
        if len(accum_max) == 0:
            print(f"[!] No successful eval batches for {path}, skipping.")
            continue
        avg_max = np.nanmean(np.stack(accum_max, axis=0), axis=0)
        avg_ent = np.nanmean(np.stack(accum_ent, axis=0), axis=0)
        avg_topk = np.nanmean(np.stack(accum_topk, axis=0), axis=0)

        L = avg_max.shape[0]
        for pos in range(L):
            topk_tokens = topk_tokens_examples[pos] if topk_tokens_examples is not None and pos < len(
                topk_tokens_examples) else []
            rows.append({
                'snapshot_epoch': epoch_idx,
                'position': pos,
                'max_prob': float(avg_max[pos]) if pos < len(avg_max) else float('nan'),
                'entropy': float(avg_ent[pos]) if pos < len(avg_ent) else float('nan'),
                f'top{top_k}_acc': float(avg_topk[pos]) if pos < len(avg_topk) else float('nan'),
                'topk_tokens': json.dumps(topk_tokens),
            })

        # Save per-snapshot quick summary
        summary = {
            'snapshot': path,
            'positions': int(L),
            'mean_max_prob': float(np.nanmean(avg_max)),
            'mean_entropy': float(np.nanmean(avg_ent)),
            'mean_topk_acc': float(np.nanmean(avg_topk[np.isfinite(avg_topk)])) if np.any(
                np.isfinite(avg_topk)) else float('nan'),
        }
        summary_path = os.path.join(out_dir, f"summary_snapshot_{epoch_idx}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"[+] Snapshot {epoch_idx} summary saved -> {summary_path}")

    # Build dataframe and save CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'snapshots_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"[+] All snapshots metrics saved to {csv_path}")

    # ---------- Plots ----------
    # 1) max_prob_by_pos: for each snapshot a line
    pivot_max = df.pivot_table(index='position', columns='snapshot_epoch', values='max_prob')
    plt.figure(figsize=(10, 5))
    for col in sorted(pivot_max.columns):
        plt.plot(pivot_max.index, pivot_max[col], label=f"ep{col}")
    plt.xlabel('position')
    plt.ylabel('max_prob')
    plt.title('Max probability by position (snapshots)')
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()
    fig1 = os.path.join(out_dir, 'max_prob_by_pos.png')
    plt.savefig(fig1)
    plt.close()
    print(f"[+] Plot saved: {fig1}")

    # 2) entropy heatmap: snapshots x positions
    pivot_ent = df.pivot_table(index='snapshot_epoch', columns='position', values='entropy')
    ent_arr = pivot_ent.sort_index().values  # shape (snapshots, positions)
    plt.figure(figsize=(12, 6))
    plt.imshow(ent_arr, aspect='auto', interpolation='nearest')
    plt.colorbar(label='entropy')
    plt.xlabel('position')
    plt.ylabel('snapshot_epoch')
    plt.title('Entropy heatmap (snapshots x position)')
    plt.tight_layout()
    fig2 = os.path.join(out_dir, 'entropy_heatmap.png')
    plt.savefig(fig2)
    plt.close()
    print(f"[+] Plot saved: {fig2}")

    # 3) convergence epoch per position: first snapshot where max_prob >= conv_thresh
    conv_epochs = {}
    grouped = df.groupby(['position'])
    for pos, g in grouped:
        # sort by epoch
        g_sorted = g.sort_values('snapshot_epoch')
        conv_epoch = None
        for _, row in g_sorted.iterrows():
            # безопасно получить max_prob как число
            val = row.get('max_prob', None)
            try:
                # если это скаляр (np/py number) — норм, иначе попытаемся привести
                if isinstance(val, (int, float, np.floating, np.integer)):
                    v = float(val)
                else:
                    v = float(np.asarray(val)) if np.isscalar(val) else float(val)
            except Exception:
                # если привести нельзя — пропускаем
                v = float('nan')

            if not math.isnan(v) and v >= conv_thresh:
                try:
                    conv_epoch = int(row['snapshot_epoch'])
                except Exception:
                    conv_epoch = row['snapshot_epoch']
                break
        # нормализуем ключ позиции в int, если возможно
        try:
            pos_key = int(pos)
        except Exception:
            pos_key = pos
        conv_epochs[pos_key] = conv_epoch if conv_epoch is not None else np.nan

    # конвертируем в Series и явно привести к числам (non-convertible -> NaN)
    conv_series = pd.Series(conv_epochs)
    conv_series = pd.to_numeric(conv_series, errors='coerce').sort_index()

    # Подготовка оси X: пытаемся использовать численные позиции, иначе порядковые индексы
    try:
        x_positions = np.array([int(p) for p in conv_series.index])
        x_labels = [str(p) for p in conv_series.index]
    except Exception:
        x_positions = np.arange(len(conv_series))
        x_labels = [str(p) for p in conv_series.index]

    plt.figure(figsize=(10, 4))
    plt.plot(x_positions, conv_series.values, marker='o')
    plt.xlabel('position')
    plt.ylabel(f'first_epoch where max_prob >= {conv_thresh}')
    plt.title('Convergence epoch per position')
    # подписи только если не слишком много точек
    if len(x_positions) <= 60:
        plt.xticks(x_positions, x_labels, rotation=90)
    plt.tight_layout()
    fig3 = os.path.join(out_dir, 'convergence_per_position.png')
    plt.savefig(fig3)
    plt.close()
    print(f"[+] Plot saved: {fig3}")

    return {
        'df': df,
        'plots': {'max_prob': fig1, 'entropy_heatmap': fig2, 'convergence': fig3},
        'snapshots': snapshots_found
    }


# d_model = 768
mem_vectors = 64
d_model = 128
dim_feedforward = d_model * 4
num_heads = 4
train_size = 1000
batch_size = 8
# time_steps = 256
time_steps = 64
top_k = 4
lr = 5e-4


# Example model_factory that reuses your datamodule and model construction.
# Adjust to your own construction if placed elsewhere.
def default_model_factory():
    """
    Возвращает (model, tokenizer, test_dataloader, device).
    Использует ту же конфигурацию, что и ваш основной блок выше.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Recreate datamodule and model skeleton (must match how snapshots were saved)
    data_module = HMoEDataModule(
        dataset_name="roneneldan/TinyStories",
        max_length=time_steps if 'time_steps' in globals() else 64,
        batch_size=1,
        device=device,
    )
    data_module.setup()
    test_dataloader = data_module.get_test_dataloader()
    tokenizer = data_module.tokenizer

    inner_model, experts_storage = HierarchicalMoE.create_hierarchical_moe(
        experts_count=100,
        chain_sizes=[4, 8, 16],
        top_k=top_k if 'top_k' in globals() else 4,
        tau=0.25,
        num_heads=num_heads if 'num_heads' in globals() else 4,
        mem_vectors=mem_vectors if 'mem_vectors' in globals() else 64,
        d_model=d_model if 'd_model' in globals() else 128,
        dim_feedforward=dim_feedforward if 'dim_feedforward' in globals() else (128 * 4),
    )
    model = EmbeddingsWrapper(inner_model, data_module.vocab_size,
                              inner_model.d_model if hasattr(inner_model, 'd_model') else d_model)
    model.to(device)
    return model, tokenizer, test_dataloader, device


# If you want to run this right away after training / loading, call:
# (Place this call in your main area after model is created / trained.)
def run_snapshot_analysis():
    SNAPSHOT_PATTERN = 'models/hmoe_memory_expert.pt_{}'  # pattern
    EPOCHS_TO_CHECK = list(range(0, 10))  # 0..9
    out = build_metrics_for_snapshots(
        model_factory=default_model_factory,
        snapshot_pattern=SNAPSHOT_PATTERN,
        epochs_range=EPOCHS_TO_CHECK,
        out_dir='analysis',
        n_eval_batches=8,
        top_k=5,
        conv_thresh=0.2
    )
    print("[+] Snapshot analysis finished. Generated files:")
    for k, v in out['plots'].items():
        print(f"    {k}: {v}")
    print("CSV:", os.path.join('analysis', 'snapshots_metrics.csv'))


if __name__ == '__main__':
    run_snapshot_analysis()
