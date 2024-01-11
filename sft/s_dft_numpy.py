"""
Class for the spiking OS-CFAR algorithm
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algorithm


class SDFT(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    NAME = "S-DFT"

    def __init__(self, *args, **kwargs):
        # Load DFT parameters
        self.n_dims = kwargs.get("n_dims")
        # Load simulations parameters
        self.timesteps = kwargs.get("timesteps", 100)
        self.alpha = kwargs.get("alpha", 0.0625)
        self.time_step = 1
        self.out_type = kwargs.get("out_type", "spike")
        self.out_abs = kwargs.get("out_abs", False)
        self.normalize = kwargs.get("normalize", False)
        self.off_bins = kwargs.get("off_bins", 0)
        # Load parent class
        super().__init__(*args, **kwargs)
        if self.out_type not in ["spike", "voltage"]:
            raise ValueError(
                    "{} not a valid value for 'out_type'".format(self.out_type)
            )
        # SNN population variables
        self.spikes = np.zeros(self.layer_dim)
        self.voltage = np.zeros((2*self.timesteps,) + self.layer_dim)
        # Initialize SNN and its connections
        self.calculate_weights()
        v_th = self.alpha*0.25*self.weights_re[0].sum()*self.timesteps
        # Charge-and-spike neuron parameters
        self.neuron_params = {}
        self.neuron_params["threshold"] = v_th
        self.neuron_params["t_silent"] = self.timesteps
        self.neuron_params["i_offset"] =2*v_th // self.timesteps
        self.neuron_params["strict_silent"] = kwargs.get("strict_silent", True)
        self.l1 = self.init_snn()

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

    def init_snn(self):
        """
        Initializes compartments depending on the number of samples
        """
        l1 = SpikingNeuralLayer(
                self.layer_dim,
                (self.weights_re, self.weights_im),
                v_threshold=self.neuron_params["threshold"],
                time_step=1
        )
        # If the simulation of the silent stage is strict,
        # spikes are not allowed until reaching spiking stage
        if self.neuron_params["strict_silent"]:
            l1.spiking = False
        else:
            l1.spiking = True
        return l1

    def simulate(self, spike_times):
        """
        Run the S-DFT over the total simulation time
        """
        # Charging stage of layer 1
        counter = 0
        current_time = 0
        while counter < self.timesteps:
            current_time = counter * self.time_step
            causal_neurons = (spike_times < current_time)
            out_l1 = self.l1.update_state(causal_neurons) * current_time
            self.voltage[counter] = self.l1.v_membrane
            self.spikes += out_l1
            counter += 1
        # Spiking stage
        self.l1.spiking = True
        self.l1.bias = self.neuron_params["i_offset"]
        causal_neurons = np.zeros_like(causal_neurons)
        while counter < 2*self.timesteps:
            current_time = counter * self.time_step
            out_l1 = self.l1.update_state(causal_neurons)
            self.spikes += out_l1 * (current_time)
            self.voltage[counter] = self.l1.v_membrane
            counter += 1
        return self.spikes

    def format_output(self, spikes):
        """
        Tranform output dictionary into output data array
        """
        # All neurons that didn't spike are forced to spike in the last step,
        # since the spike-time of 1 corresponds to the lowest possible value.
        spikes_all = np.where(spikes == 0, 2*self.timesteps, spikes)
        # Decode from TTFS spikes to input data format
        output = 1.5*self.timesteps - spikes_all.reshape(self.layer_dim)
        output = output[..., self.off_bins:,:]
        if self.out_abs:
            output = np.sqrt(output[..., 0]**2 + output[..., 1]**2)
        if self.normalize:
            output = output - output.min()
            output /= output.max()
        return output


    def _run(self, in_data):
        # Run SNN
        self.simulate(in_data)
        # Fetch  and format SNN output
        output = self.format_output(self.spikes)
        return output


class SpikingNeuralLayer():
    """
    Class for implementing a single spiking-DFT neural layer

    Args:
        shape (int|list): number of neurons in the layer. Int for a 1D
            layer or an iterable of ints for N-D layers
        weights (np.array): Matrix containing the input weights to the
            layer. They have to be real numbers
        **bias (double): external current fed to the neurons
        **threshold (double): membrane voltage for generating a spike
        **time_step (double): time gap between iterations
    """
    def __init__(self, shape, weights, **kwargs):
        """
        Initialize the class
        """
        # Neuron properties
        self.bias = kwargs.get("bias", 0)
        self.v_threshold = kwargs.get("v_threshold", 0.05)
        self.spiking = False
        # Neuron variables
        self.v_membrane = np.zeros(shape)
        self.spikes = np.zeros(shape)
        self.refactory = np.zeros(shape)
        self.weights = weights

        # Simulation parametersupdate_input_currents
        self.time_step = kwargs.get("time_step", 0.001)

    def update_input_currents(self, input_spikes):
        """
        Calculate the total current that circulates inside each neuron
        """
        # Calculate separately the currents to the real and imaginary neurons
        z_real = np.dot(self.weights[0], input_spikes)
        z_imag = np.dot(self.weights[1], input_spikes)
        z = np.vstack((z_real, z_imag)).transpose()
        # Add bias to the result and multiply by time_step
        z += self.bias
        z *= self.time_step
        z = z.reshape(self.v_membrane.shape)
        return z

    def update_membrane_potential(self, z):
        """
        Update membrane potential of each neuron

        The membrane potential increases based on the input current, and
        it returns to the rest voltage after a spike
        """
        self.v_membrane += z
        self.v_membrane *= (1-self.refactory)
        # Force a saturation point at vth and -vth
        if not self.spiking:
            self.v_membrane = np.where(
                (self.v_membrane>self.v_threshold),
                self.v_threshold,
                self.v_membrane
            )
            self.v_membrane = np.where(
                (self.v_membrane<-self.v_threshold),
                -self.v_threshold,
                self.v_membrane
            )
        return self.v_membrane

    def generate_spikes(self):
        """
        Determine which neurons spike, based on membrane potential
        """
        # Generate a spike when the voltage is higher than the threshold
        if self.spiking:
            self.spikes = np.where((self.v_membrane>self.v_threshold), True, False)
        # Activate the refactory period for the neurons that spike
        self.refactory += self.spikes
        return self.spikes

    def update_state(self, in_spikes):
        """
        Update internal state of neurons, based on input spikes
        """
        z = self.update_input_currents(in_spikes)
        self.update_membrane_potential(z)
        out_spikes = self.generate_spikes()
        return out_spikes
