#!/usr/bin/env python3
"""
Run OS-CFAR over test dataset
"""
# Standard libraries
from multiprocessing import Value
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# Local libraries
import pyrads.algms.fft
import pyrads.pipeline
import sft.encoder
import sft.preproc_pipeline
import sft.utils.metrics
import data_loader
import sft.s_dft_numpy
try:
    import sft.s_dft_loihi2
except ImportError:
    print("Warning: Unable to load Loihi libraries")
    pass


# Pipeline configuration
timesteps = 100
off_bins = 12
fft_params = {
    "n_dims": 1,
    "type": "range",
    "out_format": "modulus",
    "normalize": True,
    # "unitary": True,
    "off_bins": off_bins,
}
encoder_params = {
    "t_min": 0,
    "t_max": timesteps,
    "x_min": -1.0,
    "x_max": 1.0,
}
sft_params = {
    "n_dims": 1,
    "alpha": 0.4,
    "timesteps": timesteps,
    "out_type": "spike",
    "off_bins": off_bins,
    "normalize": True,
    "out_abs": True,
    "strict_silent": False
}


def run_std(raw_data):
    """
    Run pipeline with standard algorithms on the input data
    """
    fft_shape = raw_data.shape
    pre_pipeline = sft.preproc_pipeline.PreprocPipeline(fft_shape)
    # Define pipeline stages
    fft_alg = pyrads.algms.fft.FFT(
        fft_shape,
        **fft_params
    )
    # Create instance of pipeline and run it with input data
    std_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, fft_alg])
    std_pipeline(raw_data)
    fft = std_pipeline.output[-1]
    return fft


def get_abs(voltages):
    volts_re = voltages[:voltages.size//2]
    volts_im = voltages[voltages.size//2:]
    modulus = np.sqrt(volts_re**2 + volts_im**2)[off_bins:]
    norm_mod = modulus - modulus.min()
    norm_mod /= norm_mod.max()
    return norm_mod

def get_error(data, chirp_n, title, platform, plot, ax):
    std_ft = run_std(data)
    fpath = Path("results/{}/sft_{}_{}.npy".format(platform, title, chirp_n))
    snn_ft = np.load(fpath)
    rmse = sft.utils.metrics.get_rmse(std_ft, snn_ft)
    if plot:
        ax.plot(snn_ft, label=platform)
    return rmse

def get_voltage_error(data, chirp_n, platform, plot, ax):
    std_ft = run_std(data)
    fpath = Path("results/{}/voltage_{}.npy".format(platform, chirp_n))
    voltages = np.load(fpath)
    ft_voltages_abs = get_abs(voltages)
    rmse = sft.utils.metrics.get_rmse(std_ft, ft_voltages_abs)
    if plot:
        ax.plot(ft_voltages_abs, label=platform)
    return rmse

def main(target="car", noise="highnoise", chirp_n=30, plot=True):
    # Load data
    data, targets = data_loader.load_bbm_chirp(target, noise)
    chirp = data[:, 0, 0, chirp_n]
    subdata = "{}_{}".format(target, noise)

    # Compute and plot performance
    if plot:
        fig, axs = plt.subplots(2)
    error_loihi = get_error(chirp, chirp_n, subdata, "loihi", plot, axs[0])
    error_loihi_volts = get_voltage_error(chirp, chirp_n, "loihi", plot, axs[1])
    error_numpy = get_error(chirp, chirp_n, subdata, "numpy", plot, axs[0])
    error_numpy_volts = get_voltage_error(chirp, chirp_n, "numpy", plot, axs[1])
    print("Error Loihi: {}".format(error_loihi))
    print("Error Loihi volts: {}".format(error_loihi_volts))
    print("Error numpy: {}".format(error_numpy))
    print("Error numpy volts: {}".format(error_numpy_volts))
    if plot:
        axs[0].set_title("S-FT based on spike times")
        axs[1].set_title("S-FT based on voltages")
        axs[0].legend()
        axs[1].legend()
        plt.tight_layout()
        plt.show()
    # spectrum_plotter(data, init_chirp, end_chirp, subdata)
    # plotter([errors_loihi, errors_numpy], init_chirp, end_chirp, subdata)
    return


if __name__ == "__main__":
    main()
