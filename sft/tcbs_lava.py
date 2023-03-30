#!/usr/bin/env python3
"""
Time-based charge before spike (TCBS) neuron model in Lava
"""
# Standard libraries
import numpy as np
# Lava libraries
# Import Process level primitives
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
# Import parent classes for ProcessModels
from lava.magma.core.model.py.model import PyLoihiProcessModel
# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU
# Import decorators
from lava.magma.core.decorator import implements, requires
# Runtime libraries
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps


class TCBS(AbstractProcess):
    """
    Two-stages neuron model for temporal coding
    """
    states = {
        "idle": 0,
        "silent": 1,
        "spiking": 2,
    }
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        shape = (kwargs["shape"])
        debug = kwargs.pop("debug", False)
        start_time = kwargs.pop("start_time", 0)
        phase_time = kwargs["phase_time"]
        end_time = start_time + 2*phase_time
        total_time = kwargs.pop("total_time", end_time)
        v_th = kwargs["vth"]
        init_state = "silent" if start_time==0 else "idle"
        init_state = self.states[init_state]
        init_refractory = True if init_state==self.states["idle"] else False
        if debug:
            v_shape = (total_time, *shape,)
        else:
            v_shape = (1,)
        # Interface variables
        self.spikes_out = OutPort(shape=shape)
        self.spikes_in = InPort(shape=shape)
        # Neuron model variables
        self.u = Var(shape=shape, init=0)
        self.v = Var(shape=shape, init=0)
        self.charging_current = Var(shape=shape, init=kwargs["Iext"])
        self.bias = Var(shape=shape, init=kwargs.pop("bias", 0))
        self.refractory = Var(shape=shape, init=init_refractory)
        self.vth = Var(shape=(1,), init=v_th)
        self.spike_times = Var(shape=shape, init=np.zeros(shape).astype(np.int32))
        self.voltages = Var(shape=v_shape, init=np.zeros(v_shape).astype(np.float32))
        # Auxiliary variables
        self.state = Var(shape=(1,), init=init_state)
        self.relative_step = Var(shape=(1,), init=0)
        self.spike_count = Var(shape=(1,), init=0)
        self.max_spikes = Var(shape=(1,), init=kwargs.pop("max_spikes", 0))
        # Phase change times
        self.start_time = Var(shape=(1,), init=start_time)
        self.phase_change_time = Var(shape=(1,), init=start_time+phase_time)
        self.end_time = Var(shape=(1,), init=end_time)
        self.total_time = Var(shape=(1,), init=total_time)
        self.debug = Var(shape=(1,), init=debug)


@implements(proc=TCBS, protocol=LoihiProtocol)
@requires(CPU)
class TCBSModel(PyLoihiProcessModel):
    # Interface variables
    spikes_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, bool, precision=1)
    spikes_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float32)
    # Neuron model variables
    u: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    v: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    charging_current: float = LavaPyType(float, float, precision=32)
    bias: float = LavaPyType(float, float, precision=32)
    refractory: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)
    vth: int = LavaPyType(float, float, precision=32)
    spike_times: np.ndarray = LavaPyType(np.ndarray, int, precision=32)
    voltages: np.ndarray = LavaPyType(np.ndarray, float, precision=32)
    # Auxiliary variables
    state: int = LavaPyType(int, int, precision=16)
    relative_step: int = LavaPyType(int, int, precision=16)
    spike_count: int = LavaPyType(int, int, precision=16)
    max_spikes: int = LavaPyType(int, int, precision=16)
    # Phase change times
    start_time: int = LavaPyType(int, int, precision=16)
    phase_change_time: int = LavaPyType(int, int, precision=16)
    end_time: int = LavaPyType(int, int, precision=16)
    total_time: int = LavaPyType(int, int, precision=16)
    debug: bool = LavaPyType(bool, bool, precision=1)

    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

    def post_guard(self):
        """
        Guard function for PostManagement phase.
        """
        events = [0, self.start_time, self.phase_change_time, self.end_time]
        if self.relative_step in events:
            return True
        return False

    def run_post_mgmt(self):
        """
        Post-Management phase: executed when guard function returns True
        """
        # State machine with 3 possible states in the neuron
        # Idle stage
        if self.state == 0:
            if self.relative_step==self.start_time:
                self.state = 1
                self.u = 0 + self.bias
                self.refractory *= False
        # Charging stage
        elif self.state == 1:
            if self.relative_step==self.phase_change_time:
                self.state = 2
                self.u = np.ones_like(self.u) * self.charging_current
        # Spiking stage
        elif self.state == 2:
            if self.relative_step==self.start_time:
                self.state = 1
                self.u = 0 + self.bias
                self.refractory = False
            elif self.relative_step in [0, self.end_time]:
                self.state = 0
                self.v *= 0
                self.u *= 0
        else:
            raise ValueError("Unrecognized state: {}".format(self.state))
        if self.debug:
            pass


    def run_spk(self):
        """
        Spiking phase: executed unconditionally at every time-step
        """
        self.relative_step =  (self.time_step) % self.total_time
        spikes_in_data = self.spikes_in.recv()
        if self.state == 1:
            du = spikes_in_data
        else:
            du = np.zeros_like(spikes_in_data)
        self.u += du
        self.v[:] = (self.v + self.u) * (1-self.refractory)
        if self.debug:
            self.voltages[self.relative_step] = self.v
        if self.state == 2:
            s_out = np.logical_or(self.v > self.vth,
                                  np.isclose(self.v, self.vth)
                                 )
        else:
            s_out = np.zeros_like(self.v, dtype=bool)
        # reset voltage to 0 after a spike
        self.v[s_out] = 0
        if self.max_spikes>0 and self.spike_count>=self.max_spikes:
            s_out = np.zeros_like(self.v, dtype=bool)
        self.spikes_out.send(s_out)
        self.spike_count += s_out.sum()

        self.refractory += s_out
        # Register the spike times
        self.spike_times[s_out] = self.time_step


def main():
    delay = 1
    sim_time = 100
    n_stages = 3
    total_time = sim_time * n_stages + delay
    debug = False

    if_neuron = TCBS(
        shape=(10,5),
        start_time=delay,
        phase_time=sim_time,
        total_time=total_time,
        vth=10,
        Iext=1,
        bias=0,
        debug=debug
    )

    if_neuron.run(condition=RunSteps(num_steps=total_time), run_cfg=Loihi1SimCfg())
    if_neuron.stop()
    return


if __name__ == "__main__":
    main()
