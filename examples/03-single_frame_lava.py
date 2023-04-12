#!/usr/bin/env python3
"""
Run OS-CFAR over test dataset
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import dhandler.h5_handler
import pyrads.algms.fft
import pyrads.pipeline
import sft.encoder
import sft.preproc_pipeline
import sft.s_dft_lava


def main(
        frame_n=30,
        scene_n=7,
        timesteps=100,
        plot=True,
        downsample=False,
        out_type="voltage"
    ):
    h5_handler = dhandler.h5_handler.H5Handler("OTH/scene{}_0".format(scene_n))
    data, radar_config, calib_vec = h5_handler.load(
        dataset_dir=None
    )
    images = h5_handler.load_images()
    images = np.array(images)
    image = images[frame_n]
    import pdb; pdb.set_trace()
    raw_data = data[frame_n, 0, 0, :, :]
    if downsample:
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
    sft_params = {
        "n_dims": 1,
        "timesteps": timesteps,
        "out_type": out_type,
        "debug": True,
    }
    # encoder = sft.encoder.Encoder(fft_shape, **encoder_params)
    s_dft = sft.s_dft_lava.SDFTLava(fft_shape, **sft_params)

    lava_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, s_dft])
    lava_pipeline(raw_data)
    s_dft_out = lava_pipeline.output[-1]
    # Calculate S-FT modulus
    s_dft_decoded = 1.5*timesteps - s_dft_out
    s_dft_out = np.sqrt(s_dft_decoded[..., 0]**2 + s_dft_decoded[..., 1]**2)
    np.save("lava_out.npy", s_dft_out)

    fft_alg = pyrads.algms.fft.FFT(
        fft_shape,
        **fft_params
    )
    std_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, fft_alg])
    std_pipeline(raw_data)
    fft_out = std_pipeline.output[-1]
    np.save("std_out.npy", fft_out)

    fft_data = fft_out[0, 0, 0, 0, 2:]
    fig, axs = plt.subplots(n_plots, figsize=(10,6))

    axs[0].plot(fft_data)
    axs[1].plot(s_dft_out[0, 0, 0, 0, 2:])
    fig.savefig("out_fig.eps")
    if plot:
        plt.show()
    return


if __name__ == "__main__":
    main()
