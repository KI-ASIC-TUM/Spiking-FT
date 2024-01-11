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
    "timesteps": timesteps,
    "out_type": "spike",
    "off_bins": off_bins,
    "normalize": True,
    "out_abs": True,
    "strict_silent": False
}


def run_snn(raw_data, platform):
    """
    Encode raw data to TTFS spikes and run S-FT on specific platform

    The argument platform can take the values 'loihi' or 'numpy'
    """
    fft_shape = raw_data.shape
    # create preprocessing pipeline instance
    pre_pipeline = sft.preproc_pipeline.PreprocPipeline(fft_shape)
    # Create algorithms and main pipeline instances
    encoder = sft.encoder.Encoder(fft_shape, **encoder_params)
    if platform is "loihi":
        s_dft = sft.s_dft_loihi2.SDFT(encoder.out_data_shape, **sft_params)
    elif platform is "numpy":
        s_dft = sft.s_dft_numpy.SDFT(encoder.out_data_shape, **sft_params)
    else:
        raise ValueError("'{}' is not a valid platform".format(platform))
    spinn_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, encoder, s_dft])
    spinn_pipeline(raw_data)
    s_dft_out = spinn_pipeline.output[-1]
    # np.save("results/snn_{}.npy".format(title), s_dft_out
    voltages = spinn_pipeline._algorithms["S-DFT"].voltage[:, :,:]
    vth = spinn_pipeline._algorithms["S-DFT"].neuron_params["threshold"]
    print("vth: {}".format(vth))

    # # Create and run standard pipeline instance
    # fft_alg = pyrads.algms.fft.FFT(
    #     fft_shape,
    #     **fft_params
    # )
    # std_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, fft_alg])
    # std_pipeline(raw_data)
    # fft_out = std_pipeline.output[-1][:]
    # np.save("results/std_{}.npy".format(title), fft_out)
    return (s_dft_out, voltages)


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


def run_batch(data, init_chirp, end_chirp, title, platform="loihi"):
    """
    Iterate over a batch of chirps for running the S-FT on them
    """
    for chirp_n in range(init_chirp, end_chirp):
        fpath = Path("results/{}/sft_{}_{}.npy".format(platform, title, chirp_n))
        # If chirp was already computed, jump to next iteration
        if fpath.is_file():
            print("Skipping chirp {}, results already exist".format(chirp_n))
            continue
        sft_out, _ = run_snn(data[:, 0, 0, chirp_n], platform)
        print("Saving results for chirp {}".format(chirp_n))
        np.save(fpath, sft_out)
    return


def get_errors(data, init_chirp, end_chirp, title, platform):
    rmse = []
    for chirp_n in range(init_chirp, end_chirp):
        std_ft = run_std(data[:, 0, 0, chirp_n])
        fpath = Path("results/{}/sft_{}_{}.npy".format(platform, title, chirp_n))
        snn_ft = np.load(fpath)
        rmse.append(sft.utils.metrics.get_rmse(std_ft, snn_ft))
    return rmse

def plotter(errors, init_chirp, end_chirp, subdata):
    latex_w = 5
    latex_h = latex_w
    fig = plt.figure(figsize=(latex_w, latex_h))
    plt.boxplot(
        errors,
        widths=0.2,
        showfliers=False
    )
    plt.xticks([1, 2], ["loihi", "numpy"])
    plt.show()
    pass


def main(target="car", noise="highnoise", init_chirp=30, end_chirp=50):
    # Load data
    data, targets = data_loader.load_bbm_chirp(target, noise)
    subdata = "{}_{}".format(target, noise)

    # Run experiment
    for platform in ["loihi", "numpy"]:
        run_batch(data, init_chirp, end_chirp, subdata, platform)

    # Compute and plot performance
    errors_loihi = get_errors(data, init_chirp, end_chirp, subdata, "loihi")
    errors_numpy = get_errors(data, init_chirp, end_chirp, subdata, "numpy")
    plotter([errors_loihi, errors_numpy], init_chirp, end_chirp, subdata)
    return


if __name__ == "__main__":
    main()
