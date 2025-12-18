import time

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from gm.hmoe.embeddings_wrapper import EmbeddingsWrapper
from gm.hmoe.hierarchical_moe import HierarchicalMoE
from gm.utils.masking import *


def fit(
        model, dataloader, epochs, optimizer, criterion_ce, criterion_mse=None, device=None, log_interval=100,
        print_usage=False
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
        self.dataset = load_dataset(self.dataset_name, split="train[:1000]")

        # Tokenize
        self.tokenized_dataset = self.dataset.map(
            self.processor.tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names,
        )

    def get_dataloader(self):
        """Create DataLoader"""
        return DataLoader(
            self.tokenized_dataset,
            batch_size=self.batch_size,
            collate_fn=self.processor.collate_fn,
            shuffle=True,
        )

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size


# Testing
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # d_model = 768
    d_model = 32
    dim_feedforward = d_model * 4
    num_heads = 12
    train_size = 1000
    batch_size = 8
    # time_steps = 256
    time_steps = 32
    lr = 5e-4

    data_module = HMoEDataModule(
        dataset_name="roneneldan/TinyStories",
        max_length=time_steps,
        batch_size=batch_size,
        device=device,
    )
    data_module.setup()
    dataloader = data_module.get_dataloader()

    print('[*] init model')
    inner_model, experts_storage = HierarchicalMoE.create_hierarchical_moe(
        experts_count=21,
        chain_sizes=[2, 4, 8],
        top_k=4,
        tau=0.25,
        num_heads=12,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
    )
    model = EmbeddingsWrapper(inner_model, data_module.vocab_size, d_model)
    model.to(device)

    print(f'[+] HMoE params count: {sum(p.numel() for p in inner_model.parameters())}')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # experts_storage.disable_grad()
    fit(model, dataloader, optimizer=optimizer, criterion_ce=nn.CrossEntropyLoss(),
        criterion_mse=nn.MSELoss(), epochs=10, log_interval=5, device=device, print_usage=True)

    # fit(primitive_model, dataloader, epochs=10)
