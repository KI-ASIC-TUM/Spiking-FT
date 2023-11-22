#!/usr/bin/env python3
"""
Basic example with BBM data
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
import sft.encoder
import sft.preproc_pipeline
import sft.s_dft_numpy
import sft.utils.metrics


plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    'axes.unicode_minus': False,
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.right": False,
    "axes.spines.top": False
    })

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


def mean_shift(data, axis=-1):
    data = data - np.expand_dims(np.mean(data,axis=axis),axis=axis)
    return data


def load_chirp(target, noise):
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


def classic(chirp_n=0):
    """
    Implementation of FFT without pyradSP library
    """
    chirp, targets = load_chirp(chirp_n)
    target_range = targets[chirp_n]
    
    zeroed = mean_shift(chirp)
    window = np.hanning(512)
    windowed = window * zeroed
    fft = np.fft.fft(windowed)

    print("Target distance: {}".format(target_range[0][0]))
    ft_plotter(fft)


def ft_plotter(fft):
    """
    Generate plot of the FT
    """
    range_labels = get_range_labels()
    plt.plot(range_labels[1:], np.abs(fft[1:256]))
    plt.show()


def cfar_plotter(fft, cfar):
    """
    Generate a plot of the FT and CFAR detections on top
    """
    # Get range labels for FT spectrum
    range_labels = get_range_labels()
    # Calculate ratio between max range and max bin
    ratio = range_labels[-1] / 256
    plt.plot(range_labels[1:], np.abs(fft[1:256]))
    targets = np.where(cfar)[-1]
    for target in targets:
        plt.axvline(x=target*ratio, color="orange", linestyle="--")
    plt.show()


def cluster_plotter(fft, clusters):
    """
    Generate a plot of the FT and CFAR detections on top
    """
    # First color is for noise or border points
    cluster_palette = ["grey", "orange", "green", "salmon"]
    # Get range labels for FT spectrum
    range_labels = get_range_labels()
    # Calculate ratio between max range and max bin
    ratio = range_labels[-1] / 256
    plt.plot(range_labels[off_bins:], np.abs(fft),zorder=10)
    detections = np.where(clusters>=0)[-1]
    for detection in detections:
        plt.axvline(
            x=(detection+off_bins)*ratio,
            color=cluster_palette[clusters[detection]],
            linestyle="--",
            linewidth=1,
            zorder=1
        )
    plt.show()


def performance_plotter(snn_prec, snn_rec, std_prec, std_rec, errors, timesteps, title):
    latex_w = 5
    latex_h = latex_w * 0.7
    fig = plt.figure(figsize=(latex_w, latex_h))
    plt.plot(timesteps, snn_prec, color="cornflowerblue", label="Precision")
    plt.plot(timesteps, snn_rec, color="red", label="Recall")
    plt.ylim((0.5, 1))
    plt.legend()
    plt.axhline(y=std_prec, color="lightblue", linestyle="--")
    plt.axhline(y=std_rec, color="orange", linestyle="--")
    plt.xlabel("S-FT timesteps")
    plt.savefig("results/performance_{}.eps".format(title))
    plt.show()
    # Plot errors
    fig = plt.figure(figsize=(latex_w, latex_h))
    plt.plot(timesteps, errors, color="cornflowerblue", label="RMSE")
    plt.ylim((0.04, 0.2))
    plt.legend()
    plt.xlabel("S-FT timesteps")
    plt.savefig("results/rmse_{}.eps".format(title))
    plt.show()


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
    return fft, cfar, clusters


def run_snn(raw_data, timesteps=100):
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
    s_dft = sft.s_dft_numpy.SDFT(encoder.out_data_shape, **sft_params)
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
    sft_out = spinn_pipeline.output[-3]
    cfar_out = spinn_pipeline.output[-2]
    dbscan_out = spinn_pipeline.output[-1]
    return sft_out, cfar_out, dbscan_out


def experiment_single_chirp(
        raw_data,
        targets,
        timesteps=100,
        plot=True,
        print_metrics=True
    ):
    """
    Run FFT and SFT on a single radar chirp
    """
    fft, cfar, clusters = run_std(raw_data)
    snn, scfar, sclusters = run_snn(raw_data, timesteps=timesteps)
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
        cluster_plotter(fft, clusters)
        cluster_plotter(snn, sclusters)
    return (std_prec, std_rec, snn_prec, snn_rec, rmse)


def run_batch(data, targets_list, timesteps=100, print_metrics=False):
    """
    Run FFT and SFT for all scenes for a specific SFT configuration
    """
    n_scenes = len(targets_list)
    std_prec  = std_rec  = snn_prec  = snn_rec  = rmse = 0
    for chirp_n in range(n_scenes):
        _std_prec, _std_rec, _snn_prec, _snn_rec, _rmse = experiment_single_chirp(
            data[:, 0, 0, chirp_n],
            targets_list[chirp_n],
            timesteps=timesteps,
            plot=False,
            print_metrics=False
        )
        std_prec += _std_prec
        std_rec += _std_rec
        snn_prec += _snn_prec
        snn_rec += _snn_rec
        rmse += _rmse
    std_prec /= n_scenes
    std_rec /= n_scenes
    snn_prec /= n_scenes
    snn_rec /= n_scenes
    rmse /= n_scenes
    if print_metrics:
        print("std pipeline precision: {}".format(std_prec))
        print("std pipeline recall: {}".format(std_rec))
        print("SNN pipeline precision: {}".format(snn_prec))
        print("SNN pipeline recall: {}".format(snn_rec))
        print("Root mean squared error: {}".format(rmse))
    return (std_prec, std_rec, snn_prec, snn_rec, rmse)


def single_run(data, targets, timesteps):
    """
    Run the scenes for a single SFT configuration
    """
    run_batch(data, targets, timesteps, print_metrics=True)


def multiple_runs(data, targets, title="car_no_noise"):
    """
    Run the scenes for different SFT configurations
    """
    snn_rec = []
    snn_prec = []
    timesteps = []
    errors = []
    for n in range(30, 290, 20):
        std_prec, std_rec, snn_prec_, snn_rec_, err = run_batch(data, targets, n)
        snn_prec.append(snn_prec_)
        snn_rec.append(snn_rec_)
        errors.append(err)
        timesteps.append(n)
    performance_plotter(snn_prec, snn_rec, std_prec, std_rec, errors, timesteps, title)


def main(target="pedestrian", noise="lownoise"):
    # Plot the result for a single chirp and SFT configuration
    # data, targets = load_chirp(target, noise)
    # experiment_single_chirp(data[:, 0, 0, 40],
    #         targets[40],
    #         timesteps=100,
    #     )
    # single_run(data, targets, timesteps=160)

    # Run the SFT for different configurations and for different datasets
    for target in ["car", "pedestrian"]:
        data, targets = load_chirp(target, noise)
        title = "{}_{}".format(target, noise)
        multiple_runs(data, targets, title)


if __name__ == "__main__":
    main()
