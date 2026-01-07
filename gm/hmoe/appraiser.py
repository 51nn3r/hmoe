from gm.hmoe.hierarchical_moe import HierarchicalMoE


class Appraiser:
    hmoe: HierarchicalMoE

    def __init__(self, hmoe: HierarchicalMoE):
        self.hmoe = hmoe
