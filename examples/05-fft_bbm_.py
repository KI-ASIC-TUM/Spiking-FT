#!/usr/bin/env python3
"""
Basic example with BBM data
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
# Local libraries
import pyrads.algms.fft
import pyrads.algms.os_cfar
import pyrads.pipeline
import sft.encoder
import sft.preproc_pipeline
import sft.s_dft_numpy
import sft.utils.metrics


config = {
    "Nrange": 512,
    "bandwidth": 607.7e6,
    "Tramp": 2*20.48e-6+0e-6,
    "c0": 299792458
}


def distance_range(n_distances=None):
    """
    Extract ranges from frequencies
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


def load_chirp(chirp_n):
    data_path = "data/BBM/data.mat"
    targets_path = "data/BBM/targets.mat"
    data_dict = scipy.io.loadmat(data_path)
    targets_dict = scipy.io.loadmat(targets_path)
    data = data_dict["data"]
    targets = targets_dict["targets"]
    targets_list = []
    for i in range(targets.size):
        targets_list.append(targets[0,0,i][1])
    chirp = data[:, 0, 0, chirp_n]
    return chirp, targets_list


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
    plotter(fft)


def plotter(fft):
    range_labels = distance_range()
    plt.plot(range_labels[1:], np.abs(fft[1:256]))
    plt.show()


def run_std(raw_data):
    fft_shape = raw_data.shape
    pre_pipeline = sft.preproc_pipeline.PreprocPipeline(fft_shape)
    # Define pipeline stages
    fft_params = {
        "n_dims": 1,
        "type": "range",
        "out_format": "modulus",
        "normalize": True,
        "off_bins": 0,
        "unitary": True
    }
    oscfar_params = {
        "n_dims": 1,
        "window_width": 32,
        "ordered_k": 6,
        "alpha": 0.4,
        "n_guard_cells": 4,
    }
    fft_alg = pyrads.algms.fft.FFT(
        fft_shape,
        **fft_params
    )
    oscfar = pyrads.algms.os_cfar.OSCFAR(
        fft_alg.out_data_shape,
        **oscfar_params
    )
    # Create instance of pipeline and run it with input data
    std_pipeline = pyrads.pipeline.Pipeline([pre_pipeline, fft_alg, oscfar])
    std_pipeline(raw_data)
    fft = std_pipeline.output[-2]
    return fft


def run_snn(raw_data, out_type="spike", timesteps=200):
    raw_data = raw_data.reshape((1,1,1,1,-1))
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
        "out_type": out_type,
        "off_bins": 0,
        "normalize": True,
        "out_abs": True
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
    sft_out = spinn_pipeline.output[-2][0,0,0,0,:]
    return sft_out


def main(chirp_n=0):
    raw_data, targets = load_chirp(chirp_n)
    fft = run_std(raw_data)
    sft = run_snn(raw_data)
    # Print distance to target
    target_range = targets[chirp_n]
    print("Target distance: {}".format(target_range[0][0]))
    plotter(fft)
    plotter(sft)


if __name__ == "__main__":
    main()
