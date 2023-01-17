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
        "threshold": 10000.0,
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
        self.out_type = kwargs.get("out_type", "spike")
        if self.out_type not in ["spike", "voltage"]:
            raise ValueError(
                    "{} not a valid value for 'out_type'".format(self.out_type)
            )
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
        self.net = snn.Network(self.NAME)
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
        self.weights_re = real_weights*127
        self.weights_im = imag_weights*127


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
            record=['spikes','v']
        )
        self.out_pop_re.set_max_atoms_per_core(50)

        self.out_pop_im = snn.Population(
            self.out_data_shape[-2],
            neuron_model="lif_curr_exp_no_delay",
            params=self.neuron_params,
            name="out_im",
            record=['spikes','v']
        )
        self.out_pop_im.set_max_atoms_per_core(50)

        return


    def create_projections(self):
        """
        Create the synaptic connections between populations
        """
        connections_re, connections_im = self.get_connections()
        self.proj_re = snn.Projection(
            pre=self.in_pop,
            post=self.out_pop_re,
            connections=connections_re
        )

        self.proj_im = snn.Projection(
            pre=self.in_pop,
            post=self.out_pop_im,
            connections=connections_im
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
                connections_re.append(
                        [idx_in, idx_out, self.weights_re[idx_out, idx_in], 0]
                )
                connections_im.append(
                        [idx_in, idx_out, self.weights_im[idx_out, idx_in], 0]
                )
        return (connections_re, connections_im)

    
    def assign_input_spikes(self, spike_times):
        """
        Assign the input spike times to the spike populations
        """
        for idx in range(self.in_data_shape[-1]):
            self.in_pop.params[idx] = [int(spike_times[..., idx])]
        return


    def format_output(self, output_re, output_im):
        """
        Tranform output dictionary into output data array
        """
        # Parse the output dictionaries into lists
        out_list_re = list(output_re.values())
        out_list_im = list(output_im.values())
        # Turn multiple-output lists into single-value array
        if self.out_type == "spike":
            filled_list_re = np.array([
                    self.timesteps if not len(v) else v[0] for v in out_list_re
            ])
            filled_list_im = np.array([
                    self.timesteps if not len(v) else v[0] for v in out_list_im
            ])
        elif self.out_type == "voltage":
            filled_list_re = np.array([v[-1] for v in out_list_re])
            filled_list_im = np.array([v[-1] for v in out_list_im])
        # Merge real and imaginary outputs into single array
        spike_list = np.stack((filled_list_re, filled_list_im))
        output = np.array(spike_list).reshape(self.out_data_shape)
        return output


    def _run(self, in_data):
        # Write spike times into spike input population
        self.assign_input_spikes(in_data)
        # Run SNN on neuromorphic chip
        self.hw.run(self.net, self.timesteps)
        # Fetch  and format SNN output
        if self.out_type == "spike":
            output_re = self.out_pop_re.get_spikes()
            output_im = self.out_pop_im.get_spikes()
        elif self.out_type == "voltage":
            output_re = self.out_pop_re.get_voltages()
            output_im = self.out_pop_im.get_voltages()
        output = self.format_output(output_re, output_im)
        return output
