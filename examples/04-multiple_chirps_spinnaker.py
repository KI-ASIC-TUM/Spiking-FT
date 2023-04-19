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
try:
    import sft.s_dft_spinnaker
except ModuleNotFoundError:
    pass


def run_single_chirp(raw_data, timesteps, out_type):
    fft_shape = raw_data.shape
    pre_pipeline = sft.preproc_pipeline.PreprocPipeline(fft_shape)

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
    s_dft = sft.s_dft_spinnaker.SDFT(encoder.out_data_shape, **sft_params)

    spinn_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, encoder, s_dft])
    spinn_pipeline(raw_data)
    s_dft_out = spinn_pipeline.output[-1]
    if out_type=="spike":
        s_dft_out = 0.75*timesteps - s_dft_out
    s_dft_out = np.sqrt(s_dft_out[..., 0]**2 + s_dft_out[..., 1]**2)
    return s_dft_out


def main(timesteps=100, out_type="spike"):
    raw_data = np.load("data/scene_1.npy")
    # Downsample data so it fits in the neuromorphic chip
    raw_data = raw_data[...,::4]
    for frame_n in range(raw_data.shape[0]):
        s_dft_out = run_single_chirp(raw_data[frame_n], timesteps, out_type)
        np.save("spinn_out_{}.npy".format(frame_n), s_dft_out)
    return


if __name__ == "__main__":
    main()
