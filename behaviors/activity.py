# =====================================================================================================
# =====================================================================================================
# =====================================================================================================
# ====================================  AMIR ALI AMINI ================================================
# ====================================    610399102    ================================================
# =====================================================================================================
# =====================================================================================================

from pymonntorch import Behavior


class Activity(Behavior):
    def initialize(self, ng):
        ng.T = ng.spikes.byte().sum() / ng.size

    def forward(self, ng):
        ng.T = ng.spikes.byte().sum() / ng.size