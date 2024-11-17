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
import pyrads.algms.os_cfar
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
    224, 232, 240, 248, 256, 268,
    274, 282, 290, 300, 325, 350,
    375, 400, 425, 450, 475, 500
]

bin_sizes = [1024]

# One width per bin size
cfar_widths = [16, 16, 16]


def plot_cfar(ft, std_targets, sft, spinn_targets, n_bins, n_steps, scene_n):
    plt.subplot(2, 1 ,1)
    plt.plot(ft)
    for target in std_targets:
        plt.axvline(x=target, color="orange", linestyle="--")
    plt.title("FFT {} bins".format(n_bins))
    plt.subplot(2, 1 ,2)
    plt.plot(sft)
    for target in spinn_targets:
        plt.axvline(x=target, color="orange", linestyle="--")
    plt.title("S-FT {} steps".format(n_steps))
    plt.tight_layout()
    plt.savefig("results/cfar_{}bins_scenario{}".format(n_bins, scene_n))
    plt.show()


def run(raw_data, timesteps, out_type, title, cfar_width, scene_n):
    fft_shape = raw_data.shape
    pre_pipeline = sft.preproc_pipeline.PreprocPipeline(fft_shape)

    fft_params = {
        "n_dims": 1,
        "type": "range",
        "out_format": "modulus",
        "normalize": True,
        "off_bins": 4,
        "unitary": True
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
        "off_bins": 4,
        "normalize": True,
        "out_abs": True
    }
    oscfar_params = {
        "n_dims": 1,
        "window_width": 32,
        "ordered_k": 6,
        "alpha": 0.4,
        "n_guard_cells": 4,
    }
    spinn_oscfar_params = {
        "n_dims": 1,
        "window_width": 32,
        "ordered_k": 6,
        "alpha": 0.4,
        "n_guard_cells": 4,
    }
    encoder = sft.encoder.Encoder(fft_shape, **encoder_params)
    s_dft = sft.s_dft_numpy.SDFT(encoder.out_data_shape, **sft_params)
    spinn_oscfar = pyrads.algms.os_cfar.OSCFAR(
        s_dft.out_data_shape,
        **spinn_oscfar_params
    )

    spinn_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, encoder, s_dft, spinn_oscfar])
    spinn_pipeline(raw_data)
    s_dft_out = spinn_pipeline.output[-2]
    spinn_cfar = spinn_pipeline.output[-1]
    spinn_targets = np.where(spinn_cfar)[-1]
    np.save("results/snn_{}.npy".format(title), s_dft_out)

    fft_alg = pyrads.algms.fft.FFT(
        fft_shape,
        **fft_params
    )
    oscfar = pyrads.algms.os_cfar.OSCFAR(
        s_dft.out_data_shape,
        **oscfar_params
    )
    std_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, fft_alg, oscfar])
    std_pipeline(raw_data)
    fft_out = std_pipeline.output[-2]
    std_cfar = std_pipeline.output[-1]
    targets = np.where(std_cfar)[-1]
    # plot_cfar(
    #     fft_out[0,0,0,0,:],
    #     targets,
    #     s_dft_out[0,0,0,0,:],
    #     spinn_targets,
    #     raw_data.size,
    #     timesteps,
    #     scene_n
    # )
    np.save("results/std_{}.npy".format(title), fft_out)
    return (spinn_cfar, std_cfar)


def run_batch(raw_data, timesteps, cfar_width):
    acc_avg, prec_avg, rec_avg = (0, 0, 0)
    for i in range(4):
        title = "TI_{}".format(i)
        s_dft_out, fft_out = run(raw_data[i], timesteps, "spike", title, cfar_width, i)
        # Measure accuracy
        acc = sft.utils.metrics.get_accuracy(s_dft_out, fft_out)
        prec = sft.utils.metrics.get_precision(s_dft_out, fft_out)
        rec = sft.utils.metrics.get_recall(s_dft_out, fft_out)
        acc_avg += acc
        prec_avg += prec
        rec_avg += rec
        # print(rmse)
        # Plot results
        # fft_data = fft_out[0, 0, 0, 0, :]
        # fig, axs = plt.subplots(2, figsize=(10,6))
    acc_avg /= 4
    prec_avg /= 4
    rec_avg /= 4
    return (acc_avg, prec_avg, rec_avg)


def run_experiment(raw_data, n_bins, cfar_width):
    # Reduce data size based on number of bins
    if n_bins == 64:
        step = 16
    elif n_bins == 256:
        step = 4
    elif n_bins == 1024:
        step = 1
    raw_data = raw_data[:,::step]
    # Iterate for different simulation times
    acc = []
    prec = []
    rec = []
    for i in range(len(sim_times)):
        acc_avg, prec_avg, rec_avg = run_batch(raw_data, sim_times[i], cfar_width)
        acc.append(acc_avg)
        prec.append(prec_avg)
        rec.append(rec_avg)
    plt.plot(sim_times, acc, label="Accuracy")
    plt.plot(sim_times, prec, label="Precision")
    plt.plot(sim_times, rec, label="Recall")
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

    for i, n_bins in enumerate(bin_sizes):
        run_experiment(raw_data, n_bins, cfar_widths[i])
    plt.legend()
    plt.savefig("results/TI_rmse.eps")
    plt.show()
    return


if __name__ == "__main__":
    main()
