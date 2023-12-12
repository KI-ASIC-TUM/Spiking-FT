#!/usr/bin/env python3
"""
Run OS-CFAR over test dataset
"""
# Standard libraries
import time
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import pyrads.algms.fft
import pyrads.pipeline
import sft.encoder
import sft.preproc_pipeline
import sft.s_dft_loihi2
import sft.utils.metrics


def run(raw_data, timesteps, out_type, title, off_bins=4):
    fft_shape = raw_data.shape
    pre_pipeline = sft.preproc_pipeline.PreprocPipeline(fft_shape)

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
        "out_type": out_type,
    }
    encoder = sft.encoder.Encoder(fft_shape, **encoder_params)
    s_dft = sft.s_dft_loihi2.SDFT(encoder.out_data_shape, **sft_params)

    spinn_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, encoder, s_dft])
    spinn_pipeline(raw_data)
    s_dft_out = spinn_pipeline.output[-1]
    s_dft_out = s_dft_out[off_bins:,:]
    np.save("results/snn_{}.npy".format(title), s_dft_out)

    voltages = spinn_pipeline._algorithms["S-DFT"].voltage[:, :,:]
    vth = spinn_pipeline._algorithms["S-DFT"].neuron_params["threshold"]
    print("vth: {}".format(vth))

    fft_alg = pyrads.algms.fft.FFT(
        fft_shape,
        **fft_params
    )
    std_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, fft_alg])
    std_pipeline(raw_data)
    fft_out = std_pipeline.output[-1][:]
    np.save("results/std_{}.npy".format(title), fft_out)
    return (s_dft_out, fft_out, voltages)


def plot_voltages(voltages):
    for i in range (voltages.shape[-2]-4):
        plt.plot(voltages[:, i+4, 0])
        plt.plot(voltages[:, i+4, 0])
    plt.show()
    plt.plot(voltages[300, :, 0])
    plt.plot(voltages[300, :, 1])
    plt.show()

def plot_ft_components(ft_re, ft_im, ft_abs, timesteps):
    fig, axs = plt.subplots(3)
    axs[0].plot(ft_re)
    axs[0].set_title("Real component")
    axs[0].set_ylim([-0.5*timesteps, 0.5*timesteps])
    axs[1].plot(ft_im)
    axs[1].set_title("Imaginary component")
    axs[2].plot(ft_abs)
    axs[2].set_title("Modulus")
    plt.tight_layout()
    plt.show()
    return

def main(
        timesteps=300,
        plot=True,
        out_type="spike",
        source="special_cases"
    ):
    if source=="sample":
        raw_data = np.load("data/sample_chirp.npy")
    elif source=="special_cases":
        raw_data = np.load("data/TI_radar/special_cases/data_tum.npy")
        # reshape to standard data format
        raw_data = raw_data.reshape((4, 1024))

    raw_data = raw_data[0,::4]

    s_dft_out, fft_data, voltages = run(raw_data, timesteps, out_type, "64_bins")
    s_dft_abs = np.sqrt(s_dft_out[..., 0]**2 + s_dft_out[..., 1]**2)

    fig, axs = plt.subplots(2)
    axs[0].set_title("FFT")
    axs[0].plot(fft_data)
    axs[1].set_title("S-FT Loihi")
    axs[1].plot(s_dft_abs)
    fig.savefig("results/loihi2_sft.eps")
    plt.show()
    plot_voltages(voltages)
    plot_ft_components(s_dft_out[:,0], s_dft_out[:, 1], s_dft_abs, timesteps)
    return


if __name__ == "__main__":
    main()
