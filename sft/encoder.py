"""
Class for encoding floating values into temporal spikes

The dual encoder generates two sets of encoded values: The first set
converts the original values into spike times, and the second set
applies the alpha scaling value to the input beforehand. That way, the
os-cfar scaling value is directly included in the encoded spikes
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algorithm


class Encoder(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    NAME = "TimeEncoder"

    def __init__(self, *args, **kwargs):
        self.t_min = kwargs.get("t_min", 0)
        self.t_max = kwargs.get("t_max")
        self.x_min = kwargs.get("x_min", 0)
        self.x_max = kwargs.get("x_max")
        self.alpha = kwargs.get("alpha")
        super().__init__(*args, **kwargs)
        self.setup_encoder_params()


    def setup_encoder_params(self):
        time_range = self.t_max - self.t_min
        value_range = self.x_max - self.x_min
        self.scale_factor = time_range / value_range
        return


    def calculate_out_shape(self):
        """
        The encoder does not alter the data dimensionality
        """
        self.out_data_shape = self.in_data_shape + (2,)


    def encode(self, values):
        """
        Returns the time encoding of the value(s)

        Encoding formula in LaTeX:
        t_i = (t_{max}-t_{min}) cdot 
              (1- frac{x_i - x_{min}}{x_{max}-x_{min}} ) + t_{min}

        @param values: np.array / float / double to encode
        """
        encoded = self.t_min + (self.x_max - values) * self.scale_factor
        encoded = encoded.astype(np.int)
        encoded = np.where(encoded<self.t_min, self.t_min, encoded)
        encoded = np.where(encoded>self.t_max, self.t_max, encoded)
        return encoded


    def _run(self, in_data):
        result = self.encode(in_data)
        return result
