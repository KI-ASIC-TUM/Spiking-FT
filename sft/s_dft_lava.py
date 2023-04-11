"""
Class for the spiking DFT algorithm in Lava
"""
# Standard libraries
import numpy as np
# 3rd party libraries
from lava.magma.compiler.compiler import Compiler
from lava.magma.core.process.message_interface_enum import ActorType
from lava.magma.runtime.runtime import Runtime
from lava.proc.dense.process import Dense
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.monitor.process import Monitor
# Local libraries
import pyrads.algorithm
import sft.spike_gen
import sft.tcbs_lava


class SDFTLava(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    NAME = "S-DFT-Lava"
    neuron_params = {
        "x_max": 1.0,
        "x_min": -1.0,
        "alpha": 0.5
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load DFT parameters
        self.n_dims = kwargs.get("n_dims")
        # Load simulations parameters
        self.timesteps = kwargs.get("timesteps", 100)
        self.debug = kwargs.get("debug", False)
        # SNN population variables
        self.input_pop = None
        self.tcbs_pop = None
        self.weights_re = None
        self.weights_im = None
        # Calculate SNN paramters
        self.calculate_weights()
        alpha = self.neuron_params["alpha"]
        self.vth = alpha*0.25*self.weights[0].sum()*self.timesteps
        if self.debug:
            self.v_monitor = Monitor()


    def calculate_out_shape(self):
        """
        Split in imag and real components. Only half spectrum is useful
        """
        self.out_data_shape = self.in_data_shape[:-1]
        self.out_data_shape += (int(self.in_data_shape[-1]/2), )
        self.out_data_shape += (2, )


    def calculate_weights(self):
        """
        Calculate the weights based on the DFT equation
        """
        c = 2 * np.pi/self.in_data_shape[-1]
        n = np.arange(self.in_data_shape[-1]).reshape(self.in_data_shape[-1], 1)
        k = np.arange(self.in_data_shape[-1]).reshape(1, self.in_data_shape[-1])
        trig_factors = np.dot(n, k) * c
        real_weights = np.cos(trig_factors)[:self.out_data_shape[-2]]
        imag_weights = -np.sin(trig_factors)[:self.out_data_shape[-2]]
        self.weights = np.vstack((real_weights, imag_weights))

    def init_snn(self, input_val):
        # Input spike generation population
        self.input_pop = sft.spike_gen.TemporalSpikeGenerator(
            shape=(self.in_data_shape[-1],),
            input_val=input_val,
            vth=1.0,
            t_max=self.timesteps,
            x_max=self.neuron_params["x_max"],
            x_min=self.neuron_params["x_min"],
            ignore_zero=True,
            debug=self.debug
        )
        # S-FT neuron population
        iext = 2 * self.vth / self.timesteps
        self.tcbs_pop = sft.tcbs_lava.TCBS(
            shape=(self.in_data_shape[-1],),
            start_time=1,
            phase_time=self.timesteps,
            total_time=self.timesteps*2,
            vth=self.vth,
            Iext=iext,
            bias=0,
            max_spikes=0,
            debug=self.debug,
            name="TCBS"
        )
        # Dense layer for connecting input spikes with S-FT layer
        self.dense = Dense(shape=self.weights.shape, weights=self.weights)
        self.input_pop.spikes_out.connect(self.dense.s_in)
        self.dense.a_out.connect(self.tcbs_pop.spikes_in)
        if self.debug:
            # Set up voltage probe
            self.v_monitor.probe(self.tcbs_pop.v, self.timesteps*2)


    def _run(self, in_data):
        self.init_snn(in_data[-1])
        # Compile network
        compiler = Compiler()
        executable = compiler.compile(self.input_pop, run_cfg=Loihi1SimCfg())
        # Create a runtime and add spike times
        mp = ActorType.MultiProcessing
        runtime = Runtime(exe=executable,
                  message_infrastructure_type=mp)
        runtime.initialize()
        # self.input_pop.input_val.set(in_data[-1])
        # Execute network
        runtime.start(run_condition=RunSteps(num_steps=self.timesteps*2))
        # Get output spikes and stop execution
        spike_times = self.tcbs_pop.spike_times.get()
        if self.debug:
            self.voltages = self.v_monitor.get_data()["TCBS"]["v"]
        runtime.stop()
        output = np.stack((spike_times[:self.out_data_shape[-2]],
                           spike_times[self.out_data_shape[-2]:])).transpose()
        output = output.reshape(self.out_data_shape)
        return output
