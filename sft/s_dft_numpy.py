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
    neuron_params = {
        "threshold":20000., # high value so neurons don't spike
        "i_offset":500.0,
        "t_silent":50,

    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load DFT parameters
        self.n_dims = kwargs.get("n_dims")
        # Load simulations parameters
        self.timesteps = kwargs.get("timesteps", 100)
        self.time_step = 1
        self.out_type = kwargs.get("out_type", "spike")
        if self.out_type not in ["spike", "voltage"]:
            raise ValueError(
                    "{} not a valid value for 'out_type'".format(self.out_type)
            )
        # SNN population variables
        self.spikes = np.zeros(self.out_data_shape)
        self.voltage = np.zeros((2*self.timesteps,) + self.out_data_shape)
        # Initialize SNN and its connections
        self.calculate_weights()
        alpha = 0.25
        v_th = alpha*0.25*self.weights_re[0].sum()*self.timesteps
        # Charge-and-spike neuron parameters
        self.neuron_params = {}
        self.neuron_params["threshold"] = v_th
        self.neuron_params["t_silent"] = self.timesteps // 2
        self.neuron_params["i_offset"] =2*v_th // (self.timesteps/2)
        self.l1 = self.init_snn()

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
        Initializes compartments depending on the number of samples
        """
        l1 = SpikingNeuralLayer(
                self.out_data_shape,
                (self.weights_re, self.weights_im),
                v_threshold=self.neuron_params["threshold"],
                time_step=1
        )
        return l1

    def simulate(self, spike_times):
        """
        Run the S-DFT over the total simulation time
        """
        # Charging stage of layer 1
        counter = 0
        current_time = 0
        while counter < self.timesteps:
            causal_neurons = (spike_times < current_time)
            out_l1 = self.l1.update_state(causal_neurons) * current_time
            self.voltage[counter] = self.l1.v_membrane
            self.spikes += out_l1
            current_time = counter * self.time_step
            counter += 1

        # Spiking stage
        self.l1.bias = 2*self.neuron_params["threshold"] / self.timesteps
        causal_neurons = np.zeros_like(causal_neurons)
        while counter < 2*self.timesteps:
            out_l1 = self.l1.update_state(causal_neurons)
            self.spikes += out_l1 * (current_time)
            self.voltage[counter] = self.l1.v_membrane
            current_time = counter * self.time_step
            counter += 1
        return self.spikes

    def format_output(self, spikes):
        """
        Tranform output dictionary into output data array
        """
        output = 1.5*self.timesteps - spikes.reshape(self.out_data_shape)
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
        z_real = np.dot(self.weights[0], input_spikes[0,0,0,0,:])
        z_imag = np.dot(self.weights[1], input_spikes[0,0,0,0,:])
        z = np.hstack((z_real, z_imag))
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
        return self.v_membrane

    def generate_spikes(self):
        """
        Determine which neurons spike, based on membrane potential
        """
        # Generate a spike when the voltage is higher than the threshold
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
