#!/usr/bin/env python3
"""
Basic spike generator
"""
# Standard libraries
import numpy as np
# Lava libraries
# Import Process level primitives
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import OutPort
# Import parent classes for ProcessModels
from lava.magma.core.model.py.model import PyLoihiProcessModel
# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU
# Import decorators
from lava.magma.core.decorator import implements, requires
# Runtime libraries
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps


class TemporalSpikeGenerator(AbstractProcess):
    """
    Transform linearly float values into spike times
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = (kwargs["shape"])
        t_max = kwargs["t_max"]
        x_max = kwargs["x_max"]
        t_min = 1 # Determined by Lava dynamics
        v_th = kwargs["vth"]
        ignore_zero = kwargs.pop("ignore_zero", False) # No spikes for 0s
        # Interface variables
        self.spikes_out = OutPort(shape=shape)  # Input spikes to the classifier
        self.input_val = Var(shape=shape, init=kwargs["input_val"])
        # Neuron model variables
        self.v = Var(shape=shape, init=0)
        self.refractory = Var(shape=shape, init=0)
        self.vth = Var(shape=(1,), init=v_th)
        self.spike_times = Var(shape=shape, init=np.zeros(shape).astype(np.int32))
        # Encoding parameters
        self.t_max = Var(shape=(1,), init=t_max)
        self.x_max = Var(shape=(1,), init=x_max)
        if ignore_zero:
            t_max += 1
        k1 = (t_max-t_min) / x_max
        self.k1 = Var(shape=(1,), init=k1)
        # Auxiliary variables
        self.debug = Var(shape=(1,), init=kwargs.pop("debug", False))


@implements(proc=TemporalSpikeGenerator, protocol=LoihiProtocol)
@requires(CPU)
class TemporalSpikeGeneratorModel(PyLoihiProcessModel):
    # Interface variables
    spikes_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    input_val: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    # Neuron model variables
    v: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    refractory: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)
    vth: int = LavaPyType(float, float, precision=32)
    spike_times: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    # Encoding parameters
    t_max: int = LavaPyType(int, int, precision=32)
    x_max: float = LavaPyType(float, float, precision=32)
    k1: float = LavaPyType(float, float, precision=32)
    # Auxiliary variables
    debug: bool = LavaPyType(bool, bool, precision=1)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

    def run_spk(self):
        """
        Spiking phase: executed unconditionally at every time-step
        """
        dv = 1 / (1 + self.k1*(self.x_max-self.input_val))
        self.v[:] = (self.v + dv) * (1-self.refractory)
        if self.debug:
            pass
        s_out = np.logical_or(self.v > self.vth, np.isclose(self.v, self.vth))
        # reset voltage to 0 after a spike
        self.v[s_out] = 0
        self.spikes_out.send(s_out)

        self.refractory += s_out
        # Inhibit spikes after max time
        if (self.time_step % (self.t_max)) == 0:
            self.refractory = np.ones_like(self.refractory, dtype=bool)
        # Register the spike times
        self.spike_times[s_out] = self.time_step


def main():
    np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
    input_val = np.array((0.0, 0.2, 0.4, 1.0))
    n_inputs = input_val.shape
    x_max = 1.0
    sim_time = 10
    # Instantiate and run the spike generator
    spike_gen = TemporalSpikeGenerator(
        shape=n_inputs,
        du=0,
        dv=0,
        input_val=input_val,
        vth=1,
        t_max=sim_time,
        x_max=x_max
    )
    spike_gen.run(
        condition=RunSteps(num_steps=sim_time), run_cfg=Loihi1SimCfg()
    )
    print("Finished simulation")
    print(spike_gen.spike_times)

    spike_gen.stop()


if __name__ == "__main__":
    main()
