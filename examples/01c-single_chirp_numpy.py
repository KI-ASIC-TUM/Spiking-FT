#!/usr/bin/env python3
"""
Run OS-CFAR over test dataset
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import pyrads.algms.fft
import pyrads.pipeline
import sft.encoder
import sft.preproc_pipeline
import sft.s_dft_numpy


def main(
        timesteps=100,
        plot=True,
        out_type="spike"
    ):
    raw_data = np.load("data/sample_chirp.npy")
    # Downsample data so it fits in the neuromorphic chip
    raw_data = raw_data[...,::4]

    fft_shape = raw_data.shape
    pre_pipeline = sft.preproc_pipeline.PreprocPipeline(fft_shape)

    fft_params = {
        "n_dims": 1,
        "type": "range",
        "out_format": "modulus",
        "normalize": True,
        "off_bins": 1,
    }

    n_plots = 2
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
    s_dft = sft.s_dft_numpy.SDFT(encoder.out_data_shape, **sft_params)

    spinn_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, encoder, s_dft])
    spinn_pipeline(raw_data)
    s_dft_out = spinn_pipeline.output[-1]
    s_dft_out = np.sqrt(s_dft_out[..., 0]**2 + s_dft_out[..., 1]**2)
    np.save("spinn_out.npy", s_dft_out)


    fft_alg = pyrads.algms.fft.FFT(
        fft_shape,
        **fft_params
    )
    std_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, fft_alg])
    std_pipeline(raw_data)
    fft_out = std_pipeline.output[-1]
    np.save("std_out.npy", fft_out)

    fft_data = fft_out[0, 0, 0, 0, :]
    fig, axs = plt.subplots(n_plots, figsize=(10,6))

    axs[0].set_title("FFT")
    axs[0].plot(fft_data)
    axs[1].set_title("S-FT Numpy")
    axs[1].plot(s_dft_out[0, 0, 0, 0, :])
    fig.savefig("out_fig.eps")
    if plot:
        plt.show()
    return


if __name__ == "__main__":
    main()
