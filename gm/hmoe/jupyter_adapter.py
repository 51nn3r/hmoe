import torch

from gm.hmoe.embeddings_wrapper import EmbeddingsWrapper
from gm.hmoe.hierarchical_moe import HierarchicalMoE
from gm.hmoe.hmoe_nlp import HMoEDataModule
from gm.hmoe.scenario import Scenario


class JupiterAdapter:
    def __init__(self):
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
        self.test_dataloader = data_module.get_test_dataloader()

        print('[*] init model')
        self.inner_model, self.experts_storage = HierarchicalMoE.create_hierarchical_moe(
            experts_count=21,
            chain_sizes=[2, 4, 8],
            top_k=top_k,
            tau=0.25,
            num_heads=12,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
        )
        self.model = EmbeddingsWrapper(self.inner_model, data_module.vocab_size, d_model)
        self.model.to(device)

        batch = next(iter(self.test_dataloader))
        inputs = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
        targets = batch['labels'].to(device)
        res = self.model(inputs, attn_mask, create_scenario=True)
        scenario: Scenario = res['scenario']
