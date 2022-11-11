"""
Class for the spiking OS-CFAR algorithm
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algorithm
from spinnaker2 import snn, hardware


class SDFT(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    NAME = "S-DFT"
    neuron_params = {
        "threshold": 1.0,
        "alpha_decay": 1.0,
        "exc_decay": 1.0,
        "inh_decay": 1.0,
        "i_offset": 0.0,
        "v_reset": 0.0,
        "reset": 'reset_to_v_reset',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load DFT parameters
        self.n_dims = kwargs.get("n_dims")
        # Load simulations parameters
        self.timesteps = kwargs.get("timesteps", 100)
        # SNN population variables
        self.in_pop = None
        self.out_pop_re = None
        self.out_pop_im = None
        self.weights_re = None
        self.weights_im = None
        self.proj_re = None
        self.proj_im = None
        # Initialize SNN and its connections
        self.calculate_weights()
        self.init_snn()
        self.create_projections()
        # Create network and add populations to it
        self.net = snn.Network("my network")
        self.net.add(self.in_pop, self.out_pop_re, self.proj_re)
        self.net.add(self.in_pop, self.out_pop_im, self.proj_im)
        # Instantiate hardware
        self.hw = hardware.FPGA_Rev2()


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
        self.weights_re = np.ceil(real_weights*127)
        self.weights_im = np.ceil(imag_weights*127)


    def init_snn(self):
        """
        Initialize the SNN populations

        The output of the OS-DFT is represented by the out_pop, and the
        input is represented by in_pop.
        """
        self.in_pop = snn.Population(
            self.in_data_shape[-1],
            neuron_model="spike_list",
            params={},
            name="in",
        )
        self.out_pop_re = snn.Population(
            self.out_data_shape[-2],
            neuron_model="lif_curr_exp_no_delay",
            params=self.neuron_params,
            name="out_re",
        )
        self.out_pop_im = snn.Population(
            self.out_data_shape[-2],
            neuron_model="lif_curr_exp_no_delay",
            params=self.neuron_params,
            name="out_im",
        )
        return


    def create_projections(self):
        """
        Create the synaptic connections between populations
        """
        connections = self.get_connections()
        self.proj_re = snn.Projection(
            pre=self.in_pop,
            post=self.out_pop_re,
            connections=connections
        )

        self.proj_im = snn.Projection(
            pre=self.in_pop,
            post=self.out_pop_im,
            connections=connections
        )
        return

    
    def get_connections(self):
        """
        Define the synaptic connections between in and out neurons
        """
        connections_re = []
        connections_im = []
        for idx_in in range(self.in_data_shape[-1]):
            for idx_out in range(self.out_data_shape[-2]):
                connections_re.append([idx_in, idx_out, self.weights_re[idx_out], 0])
                connections_im.append([idx_in, idx_out, self.weights_im[idx_out], 0])
        return (connections_re, connections_im)

    
    def assign_input_spikes(self, spike_times):
        """
        Assign the input spike times to the spike populations
        """
        for idx in range(self.in_data_shape[-1]):
            self.in_pop.params[idx] = [int(spike_times[..., idx])]
        return


    def format_output(self, spike_times_re, spike_times_im):
        """
        Tranform spike dictionary into output data array
        """
        spike_list_re = list(spike_times_re.values())
        spike_list_im = list(spike_times_im.values())
        filled_list_re = np.array([
                self.timesteps if not len(v) else v[0] for v in spike_list_re
        ])
        filled_list_im = np.array([
                self.timesteps if not len(v) else v[0] for v in spike_list_im
        ])
        spike_list = np.stack((filled_list_re, filled_list_im))
        output = np.array(spike_list).reshape(self.out_data_shape)
        return output


    def _run(self, in_data):
        self.assign_input_spikes(in_data)
        self.hw.run(self.net, self.timesteps)
        spike_times_re = self.out_pop_re.get_spikes()
        spike_times_im = self.out_pop_im.get_spikes()
        output = self.format_output(spike_times_re, spike_times_im)
        return output
