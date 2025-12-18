from torch import nn


class ExpertsStorage(nn.Module):
    _experts_count: int
    _experts: nn.ModuleList

    def __init__(
            self,
            experts_count: int,
    ):
        super().__init__()

        self._experts_count = 0
        self._experts = nn.ModuleList([])

    def set_experts(self, experts: nn.ModuleList):
        self._experts = experts
        self._experts_count = len(experts)

    def enable_grad(self):
        for param in self.parameters():
            param.requires_grad = True

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    @property
    def experts(self):
        return self._experts
