import torch
from pymonntorch import (
    NeuronGroup,
    NeuronDimension,
    EventRecorder,
    Recorder,
    SynapseGroup,
)
from conex import (
    Neocortex,
    NeuronAxon,
    SpikeTrace,
    SimpleDendriteComputation,
    SimpleDendriteStructure,
    LIF,
)
from conex import (
    Synapsis,
    SynapseInit,
    WeightInitializer,
    Conv2dDendriticInput,
    Conv2dSTDP,
    prioritize_behaviors,
    Fire,
)
from conex import ActivityBaseHomeostasis, KWTA, LateralDendriticInput
from conex.helpers import Poisson

from matplotlib import pyplot as plt

import behaviors.InputData as InputData


def encode(
    data,
    height,
    width,
    method="ITL",
    RECORDER_INDEX=460,
    EV_RECORDER_INDEX=461,
    OUT_R=10,
    OUT_THRESHOLD=15,
    OUT_TAU=3,
    OUT_V_RESET=0,
    OUT_V_REST=5,
    T=50,
    ratio=0.003,
):

    net = Neocortex(dt=1, device="cpu", dtype=torch.float32)
    ng1 = NeuronGroup(
        size=NeuronDimension(height=height, width=width),
        behavior={
            **prioritize_behaviors(
                [
                    SimpleDendriteStructure(),
                    SimpleDendriteComputation(),
                    LIF(
                        R=OUT_R,
                        threshold=OUT_THRESHOLD,
                        tau=OUT_TAU,
                        v_reset=OUT_V_RESET,
                        v_rest=OUT_V_REST,
                    ),  # 260
                    Fire(),  # 340
                    SpikeTrace(tau_s=3),
                    NeuronAxon(),
                ]
            ),
            **{
                10: InputData.ResetMemory(),
                345: InputData.Encode(
                    data=data.unsqueeze(0),
                    time=T,
                    ratio=height * ratio,
                    method=method,
                ),
                EV_RECORDER_INDEX: EventRecorder("spikes", tag="input_ev_recorder"),
            },
        },
        net=net,
    )
    net.initialize(info=False)
    net.simulate_iterations(T)
    return ng1["input_ev_recorder", 0].variables["spikes"]
