"""
Class for the spiking DFT algorithm in Lava
"""
# Standard libraries
import numpy as np
# 3rd party libraries
from lava.proc.dense.process import Dense
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

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
        "threshold": 100.0,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load DFT parameters
        self.n_dims = kwargs.get("n_dims")
        # Load simulations parameters
        self.timesteps = kwargs.get("timesteps", 100)
        # SNN population variables
        self.input_pop = None
        self.tcbs_pop = None
        self.weights_re = None
        self.weights_im = None
        # Initialize SNN and its connections
        self.calculate_weights()
        self.init_snn()


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
        # Normalize for the allowed range in SpiNNaker
        self.weights_re = real_weights*127
        self.weights_im = imag_weights*127
        import pdb; pdb.set_trace()
        self.weights = np.stack((self.weights_re, self.weights_im))

    def init_snn(self):
        # Input spike generation population
        self.input_pop = sft.spike_gen.TemporalSpikeGenerator(
            shape=self.in_data_shape,
            input_val=None,
            vth=1.0,
            t_max=self.timesteps,
            x_max=1.0,
            ignore_zero=True,
            debug=True
        )
        # S-FT neuron population
        vth = self.neuron_params["threshold"]
        iext = vth / self.timesteps
        self.tcbs_pop = sft.tcbs_lava.TCBS(
            shape=self.out_data_shape,
            start_time=1,
            phase_time=self.timesteps2,
            total_time=self.timesteps*2,
            vth=self.neuron_params["threshold"],
            Iext=iext,
            bias=0,
            max_spikes=0,
            debug=True
        )
        # Dense layer for connecting input spikes with S-FT layer
        self.dense = Dense(shape=self.weights.shape, weights=self.weights)
        self.input_pop.spike_out.connect(self.dense.s_in)
        self.dense.a_out.connect(self.tcbs_pop.spikes_in)


    def _run(self, in_data):
        # Run SNN on neuromorphic chip
        self.input_pop.input_val.set(in_data)
        self.input_pop.run(
            condition=RunSteps(num_steps=self.timesteps*2),
            run_cfg=Loihi1SimCfg()
        )
        spike_times = self.tcbs_pop.spike_times.get()
        self.input_pop.stop()
        output = spike_times
        return output
