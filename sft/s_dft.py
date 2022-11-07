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
        "threshold":1.,
        "alpha_decay":1.0,
        "i_offset":0.0,
        "v_reset": 0.0,
        "reset": 'reset_to_v_reset',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load os-cfar parameters
        self.n_dims = kwargs.get("n_dims")
        # Load simulations parameters
        self.timesteps = kwargs.get("timesteps", 100)
        # SNN population variables
        self.in_pop = None
        self.out_pop_re = None
        self.out_pop_im = None
        self.proj_re = None
        self.proj_im = None
        # Initialize SNN and its connections
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
            neuron_model="lif_no_delay",
            params=self.neuron_params,
            name="out_re",
        )
        self.out_pop_im = snn.Population(
            self.out_data_shape[-2],
            neuron_model="lif_no_delay",
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
        connections = []
        for idx_in in range(self.in_data_shape[-1]):
            for idx_out in range(self.out_data_shape[-2]):
                connections.append([idx_in, idx_out, 1, 0])
        return connections

    
    def assign_input_spikes(self, spike_times):
        """
        Assign the input spike times to the spike populations
        """
        for idx in range(self.in_data_shape[-1]):
            self.in_pop.params[idx] = [int(spike_times[..., idx, 0])]
        return


    def format_output(self, spike_times):
        """
        Tranform spike dictionary into output data array
        """
        spike_list = list(spike_times.values())
        filled_list = [0 if not len(v) else 1 for v in spike_list]
        output = np.array(filled_list).reshape(self.out_data_shape)
        return output


    def _run(self, in_data):
        self.assign_input_spikes(in_data)
        self.hw.run(self.net, self.timesteps)
        spike_times = self.out_pop.get_spikes()
        output = self.format_output(spike_times)
        return output
