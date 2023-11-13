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
import sft.s_dft_numpy
import sft.utils.metrics


sim_times = [
    16, 24, 32, 40, 48, 56, 64,
    72, 80, 88, 96, 104, 112, 120,
    128, 136, 144, 152, 160, 168,
    176, 184, 192, 200, 208, 216,
    224, 232, 240, 248, 256, 257
]

bin_sizes = [64, 256, 1024]


def run(raw_data, timesteps, out_type, title):
    fft_shape = raw_data.shape
    pre_pipeline = sft.preproc_pipeline.PreprocPipeline(fft_shape)

    fft_params = {
        "n_dims": 1,
        "type": "range",
        "out_format": "modulus",
        "normalize": True,
        "off_bins": 4,
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
    s_dft = sft.s_dft_numpy.SDFT(encoder.out_data_shape, **sft_params)

    spinn_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, encoder, s_dft])
    spinn_pipeline(raw_data)
    s_dft_out = spinn_pipeline.output[-1]
    s_dft_out = np.sqrt(s_dft_out[..., 0]**2 + s_dft_out[..., 1]**2)
    np.save("results/snn_{}.npy".format(title), s_dft_out)


    fft_alg = pyrads.algms.fft.FFT(
        fft_shape,
        **fft_params
    )
    std_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, fft_alg])
    std_pipeline(raw_data)
    fft_out = std_pipeline.output[-1]
    np.save("results/std_{}.npy".format(title), fft_out)
    return (s_dft_out, fft_out)


def run_batch(raw_data, timesteps):
    rmse_avg = 0
    for i in range(4):
        title = "TI_{}".format(i)
        s_dft_out, fft_out = run(raw_data[i], timesteps, "spike", title)
        s_dft_out = s_dft_out[0,0,0,0,4:]
        # Measure RMSE
        rmse = sft.utils.metrics.get_rmse(s_dft_out, fft_out)
        rmse_avg += rmse
        # print(rmse)
        # Plot results
        # fft_data = fft_out[0, 0, 0, 0, :]
        # fig, axs = plt.subplots(2, figsize=(10,6))
    rmse_avg /= 4
    return rmse_avg


def run_experiment(raw_data, n_bins):
    # Reduce data size based on number of bins
    if n_bins == 64:
        step = 16
    elif n_bins == 256:
        step = 4
    elif n_bins == 1024:
        step = 1
    raw_data = raw_data[:,:,:,:,:,::step]
    # Iterate for different simulation times
    errors = []
    for i in range(len(sim_times)):
        rmse_avg = run_batch(raw_data, timesteps=sim_times[i])
        errors.append(rmse_avg)
    plt.plot(sim_times, errors, label=r'$n_{{bins}}={}$'.format(n_bins))
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
        raw_data = raw_data.reshape((4, 1,1,1,1, 1024))

    for n_bins in bin_sizes:
        run_experiment(raw_data, n_bins)
    plt.legend()
    plt.savefig("results/TI_rmse.eps")
    plt.show()


        # axs[0].set_title("FFT")
        # axs[0].plot(fft_data)
        # axs[1].set_title("S-FT Numpy")
        # axs[1].plot(s_dft_out)
        # fig.savefig("results/TI_{}.eps".format(i))
        # if plot:
        #     plt.show()
    return


if __name__ == "__main__":
    main()
