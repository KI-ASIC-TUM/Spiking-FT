"""
Class for the spiking OS-CFAR algorithm
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algorithm
try:
    from spinnaker2 import snn, hardware
except ModuleNotFoundError:
    pass


class SDFT(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    NAME = "S-DFT"
    neuron_params = {
        "threshold":20000., # high value so neurons don't spike
        "i_offset":500.0,
        "t_silent":50,

    }

    def __init__(self, *args, **kwargs):
        # Load simulations parameters
        self.alpha = kwargs.get("alpha", 0.4)
        self.timesteps = kwargs.get("timesteps", 100)
        self.out_type = kwargs.get("out_type", "spike")
        self.out_abs = kwargs.get("out_abs", False)
        self.normalize = kwargs.get("normalize", False)
        self.off_bins = kwargs.get("off_bins", 0)
        self.debug = kwargs.get("debug", False)
        if self.out_type not in ["spike", "voltage"]:
            raise ValueError(
                    "{} not a valid value for 'out_type'".format(self.out_type)
            )
        super().__init__(*args, **kwargs)
        # Load DFT parameters
        self.n_dims = kwargs.get("n_dims")
        # SNN population variables
        self.in_pop = None
        self.out_pop = None
        self.out_pop_re = None
        self.out_pop_im = None
        self.weights_re = None
        self.weights_im = None
        self.proj_re = None
        self.proj_im = None
        # Initialize SNN and its connections
        self.calculate_weights()
        v_th = self.alpha*0.25*self.weights_re[0].sum()*self.timesteps
        self.voltage = np.zeros((2*self.timesteps,) + self.layer_dim)
        # Charge-and-spike neuron parameters
        self.neuron_params = {}
        self.neuron_params["threshold"] = v_th
        self.neuron_params["t_silent"] = self.timesteps
        self.neuron_params["i_offset"] =2*v_th // self.timesteps
        self.init_snn()
        self.create_projections()
        # Create network and add populations to it
        self.net = snn.Network(self.NAME)
        self.net.add(self.in_pop, self.out_pop, self.proj)
        # Instantiate hardware
        self.hw = hardware.SpiNNaker2Chip(eth_ip="192.168.1.59")


    def calculate_out_shape(self):
        """
        Split in imag and real components. Only half spectrum is useful
        """
        self.out_data_shape = self.in_data_shape[:-1]
        self.out_data_shape += (int(self.in_data_shape[-1]/2)-self.off_bins, )
        if not self.out_abs:
            self.out_data_shape += (2, )
        self.layer_dim = self.in_data_shape[:-1] + (int(self.in_data_shape[-1]/2), 2)


    def calculate_weights(self):
        """
        Calculate the weights based on the DFT equation
        """
        c = 2 * np.pi/self.in_data_shape[-1]
        n = np.arange(self.in_data_shape[-1]).reshape(self.in_data_shape[-1], 1)
        k = np.arange(self.in_data_shape[-1]).reshape(1, self.in_data_shape[-1])
        trig_factors = np.dot(n, k) * c
        real_weights = np.cos(trig_factors)[:self.layer_dim[-2]]
        imag_weights = -np.sin(trig_factors)[:self.layer_dim[-2]]
        # Normalize for the allowed range in SpiNNaker
        self.weights_re = real_weights*127
        self.weights_im = imag_weights*127
        self.weights = np.vstack((self.weights_re, self.weights_im))


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
        self.out_pop = snn.Population(
            self.in_data_shape[-1],
            neuron_model="charge_and_spike",
            params=self.neuron_params,
            name="out",
            record=['spikes']
        )
        self.out_pop.set_max_atoms_per_core(50)
        return


    def create_projections(self):
        """
        Create the synaptic connections between populations
        """
        connections = self.get_connections()
        self.proj = snn.Projection(
            pre=self.in_pop,
            post=self.out_pop,
            connections=connections
        )
        return


    def get_connections(self):
        """
        Define the synaptic connections between in and out neurons
        """
        connections = []
        for idx_in in range(self.in_data_shape[-1]):
            for idx_out in range(self.in_data_shape[-1]):
                connections.append(
                        [idx_in, idx_out, self.weights[idx_out, idx_in], 0]
                )
        return connections

    
    def assign_input_spikes(self, spike_times):
        """
        Assign the input spike times to the spike populations
        """
        for idx in range(self.in_data_shape[-1]):
            self.in_pop.params[idx] = [int(spike_times[..., idx])]
        return
    

    def format_output(self, spikes):
        """
        Tranform output dictionary into output data array
        """
        # Parse the output dictionaries into lists
        spike_list = list(spikes.values())
        # Turn multiple-output lists into single-value array
        if self.out_type == "spike":
            filled_array = np.array([
                    2*self.timesteps if not len(v) else v[0] for v in spike_list
            ])
        elif self.out_type == "voltage":
            filled_array = np.array([v[self.neuron_params["t_silent"]] for v in spike_list])
        # Merge real and imaginary outputs into single array
        reshaped_spikes = np.vstack((
            filled_array[:self.layer_dim[-2]],
            filled_array[self.layer_dim[-2]:]
        )).transpose()
        output = 1.5*self.timesteps - reshaped_spikes.reshape(self.layer_dim)
        output = output[..., self.off_bins:,:]
        if self.out_abs:
            output = np.sqrt(output[..., 0]**2 + output[..., 1]**2)
        return output


    def _run(self, in_data):
        # Write spike times into spike input population
        self.assign_input_spikes(in_data)
        # Run SNN on neuromorphic chip
        self.hw.run(self.net, 2*self.timesteps)
        # Fetch  and format SNN output
        if self.out_type == "spike":
            spikes = self.out_pop.get_spikes()
        elif self.out_type == "voltage":
            output = self.out_pop.get_voltages()
        output = self.format_output(spikes)
        return output
