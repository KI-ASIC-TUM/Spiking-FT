#!/usr/bin/env python3
"""
Utility functions for the examples
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
# Local libraries
import pyrads.algms.dbscan
import pyrads.algms.fft
import pyrads.algms.os_cfar
import pyrads.pipeline
import sft.utils.metrics


# Radar configuration in the BBM simulator
config = {
    "Nrange": 512,
    "bandwidth": 607.7e6,
    "Tramp": 2*20.48e-6+0e-6,
    "c0": 299792458
}

# Pipeline configuration
off_bins = 12
fft_params = {
    "n_dims": 1,
    "type": "range",
    "out_format": "modulus",
    "normalize": True,
    "off_bins": off_bins,
    "unitary": True
}
oscfar_params = {
    "n_dims": 1,
    "window_width": 32,
    "ordered_k": 6,
    "alpha": 0.4,
    "n_guard_cells": 4,
}
dbscan_params = {
    "n_dims": 1,
    "min_pts": 1,
    "epsilon": 3,
}


def load_bbm_chirp(target, noise):
    """
    Load chirp from BBM dataset
    """
    root_path = "data/BBM_{}_{}".format(target, noise)
    data_path = "{}/data.mat".format(root_path)
    targets_path = "{}/targets.mat".format(root_path)
    data_dict = scipy.io.loadmat(data_path)
    targets_dict = scipy.io.loadmat(targets_path)
    data = data_dict["data"]
    targets = targets_dict["targets"]
    targets_list = []
    for i in range(targets.size):
        targets_list.append(targets[0,0,i][1])
    return data, targets_list


def get_range_labels(n_distances=None):
    """
    Extract range labels from radar parameters
    """
    fs = 50e6
    eff_b = float(config['Nrange'])*config['bandwidth']/(fs*config['Tramp']/2+1)
    d_res = config['c0']/(2*eff_b)
    d_max = float(config['Nrange'])//2*d_res
    if n_distances is None:
        distance_range = np.linspace(0, d_max, config['Nrange'] // 2, endpoint=False)
    else:
        distance_range = np.linspace(0, d_max, n_distances, endpoint=False)

    return distance_range


def cluster_plotter(fft, clusters, off_bins, ax=None, title=""):
    """
    Generate a plot of the FT and CFAR detections on top
    """
    if not ax:
        _, ax = plt.subplots(1)
    # First color is for noise or border points
    cluster_palette = ["grey", "orange", "green", "salmon"]
    # Get range labels for FT spectrum
    range_labels = get_range_labels()
    # Calculate ratio between max range and max bin
    ratio = range_labels[-1] / 256
    ax.plot(range_labels[off_bins:], np.abs(fft),zorder=10)
    detections = np.where(clusters>=0)[-1]
    for detection in detections:
        ax.axvline(
            x=(detection+off_bins)*ratio,
            color=cluster_palette[clusters[detection]],
            linestyle="--",
            linewidth=1,
            zorder=1
        )
    ax.set_xlabel("Range (m)")
    ax.set_yticks([])
    ax.set_title(title)


def voltage_plotter(voltages, off_bins, ax=None):
    """
    Generate a plot of the SNN voltages over time
    """
    plt.set_cmap('Blues')
    if not ax:
        _, ax = plt.subplots(1)
    timesteps, bins, _ = voltages.shape
    colors = plt.cm.Blues(np.linspace(0.4,1,10))
    for i in range(off_bins, bins//2):
        ax.plot(voltages[:, i, 0], color=colors[2*i%10])
        ax.plot(voltages[:, i, 1], color=colors[2*i%10])
    # Set vertical line at stage change
    ax.axvline(
        x=timesteps//2,
        color="gray",
        linestyle="--",
        linewidth=1,
        zorder=1
    )
    ax.set_yticks([])
    ax.set_xlabel("Timesteps")
    ax.set_title("S-FT voltages")


def run_std(raw_data, dataset="unknown", subdata="", id=0):
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
    oscfar = pyrads.algms.os_cfar.OSCFAR(
        fft_alg.out_data_shape,
        **oscfar_params
    )
    dbscan = pyrads.algms.dbscan.DBSCAN(
        oscfar.out_data_shape,
        **dbscan_params
    )
    # Create instance of pipeline and run it with input data
    std_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, fft_alg, oscfar, dbscan])
    std_pipeline(raw_data)
    fft = std_pipeline.output[-3]
    cfar = std_pipeline.output[-2]
    clusters = std_pipeline.output[-1]
    np.save("results/{}/{}_{}_std.npy".format(dataset, subdata, id), fft)
    return fft, cfar, clusters


def run_snn(
        raw_data,
        off_bins,
        source,
        timesteps=100,
        dataset="unknown",
        subdata="",
        id=0
    ):
    """
    Run pipeline with spiking FT on the input data

    The algorithms other than the S-FT are classic algorithms
    """
    # raw_data = raw_data.reshape((1,1,1,1,-1))
    fft_shape = raw_data.shape
    pre_pipeline = sft.preproc_pipeline.PreprocPipeline(fft_shape)
    # Define pipeline stages
    encoder_params = {
        "t_min": 0,
        "t_max": timesteps,
        "x_min": -1.0,
        "x_max": 1.0,
    }
    sft_params = {
        "n_dims": 1,
        "timesteps": timesteps,
        "alpha": 0.0625,
        "out_type": "spike",
        "off_bins": off_bins,
        "normalize": True,
        "out_abs": True
    }
    encoder = sft.encoder.Encoder(fft_shape, **encoder_params)
    if source=="numpy":
        s_dft = sft.s_dft_numpy.SDFT(encoder.out_data_shape, **sft_params)
    elif source=="loihi2":
        s_dft = sft.s_dft_loihi2.SDFT(encoder.out_data_shape, **sft_params)
    spinn_oscfar = pyrads.algms.os_cfar.OSCFAR(
        s_dft.out_data_shape,
        **oscfar_params
    )
    dbscan = pyrads.algms.dbscan.DBSCAN(
        spinn_oscfar.out_data_shape,
        **dbscan_params
    )
    spinn_pipeline = pyrads.pipeline.Pipeline(
        [pre_pipeline, encoder, s_dft, spinn_oscfar, dbscan]
    )
    spinn_pipeline(raw_data)
    # Fetch membrane voltages over time for debugging purposes
    voltages = spinn_pipeline._algorithms["S-DFT"].voltage
    sft_out = spinn_pipeline.output[-3]
    cfar_out = spinn_pipeline.output[-2]
    dbscan_out = spinn_pipeline.output[-1]
    np.save(
        "results/{}/{}_{}_{}.npy".format(dataset, subdata, id, source),
        sft_out
    )
    return sft_out, voltages, cfar_out, dbscan_out


def experiment_single_chirp(
        raw_data,
        targets,
        source="numpy",
        timesteps=100,
        plot=True,
        print_metrics=True,
        dataset="BBM",
        subdata="car_highnoise",
        id=0
    ):
    """
    Run FFT and SFT on a single radar chirp

    str source: numpy | loihi2
    """
    fft, cfar, clusters = run_std(
        raw_data,
        dataset,
        subdata,
        id
    )
    snn, voltages, scfar, sclusters = run_snn(
        raw_data,
        off_bins,
        source,
        timesteps,
        dataset,
        subdata,
        id
    )
    # Print distance to target
    target_range = targets[0][0]
    target = [target_range]
    # Get precision
    range_labels = get_range_labels()
    ratio = range_labels[-1] / 256
    std_prec, std_rec = sft.utils.metrics.get_clustering_performance(
        clusters, target, off_bins, ratio
    )
    snn_prec, snn_rec = sft.utils.metrics.get_clustering_performance(
        sclusters, target, off_bins, ratio
    )
    rmse = sft.utils.metrics.get_rmse(fft, snn)
    if print_metrics:
        print("Target distance: {}".format(target_range))
        print("std pipeline precision: {}".format(std_prec))
        print("std pipeline recall: {}".format(std_rec))
        print("SNN pipeline precision: {}".format(snn_prec))
        print("SNN pipeline recall: {}".format(snn_rec))
    if plot:
        latex_w = 5
        latex_h = latex_w
        fig, axs = plt.subplots(3, layout="constrained", figsize=(latex_w, latex_h))
        cluster_plotter(fft, clusters, off_bins, axs[0], "FFT")
        cluster_plotter(snn, sclusters, off_bins, axs[1], "S-FT")
        voltage_plotter(voltages, off_bins, axs[2])
        plt.savefig("results/sft_single_chirp_simulation.pdf", dpi=300)
        plt.show()
    return (std_prec, std_rec, snn_prec, snn_rec, rmse)


def run_batch(
        data,
        targets_list,
        timesteps=100,
        source="numpy",
        print_metrics=False,
        dataset="unknown",
        subdata=""
    ):
    """
    Run FFT and SFT for all scenes for a specific SFT configuration
    """
    n_scenes = len(targets_list)
    std_prec  = std_rec  = snn_prec  = snn_rec  = 0
    rmse = []
    print("Running {} iterations".format(n_scenes))
    for chirp_n in range(n_scenes):
        print("Starting iteration {}".format(chirp_n+1))
        _std_prec, _std_rec, _snn_prec, _snn_rec, _rmse = experiment_single_chirp(
            data[:, 0, 0, chirp_n],
            targets_list[chirp_n],
            source,
            timesteps=timesteps,
            plot=False,
            print_metrics=False,
            dataset=dataset,
            subdata=subdata,
            id=chirp_n
        )
        std_prec += _std_prec
        std_rec += _std_rec
        snn_prec += _snn_prec
        snn_rec += _snn_rec
        rmse.append(_rmse)
    std_prec /= n_scenes
    std_rec /= n_scenes
    snn_prec /= n_scenes
    snn_rec /= n_scenes
    rmse = np.array(rmse)
    np.save("results/rmse_{}.npy".format(source), rmse)
    rmse_dist = np.array((rmse.mean(), rmse.std()))
    if print_metrics:
        print("std pipeline precision: {}".format(std_prec))
        print("std pipeline recall: {}".format(std_rec))
        print("SNN pipeline precision: {}".format(snn_prec))
        print("SNN pipeline recall: {}".format(snn_rec))
        print("Root mean squared error: {}".format(rmse_dist[0]))
    return (std_prec, std_rec, snn_prec, snn_rec, rmse)
