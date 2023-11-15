#!/usr/bin/env python3
"""
Module containing accuracy metrics
"""
# Standard libraries
import numpy as np
# Local libraries


def normalize(data):
    # Normalize resulting FT
    result = data - data.min()
    result /= result.max()
    return result

def simplify_ft(data):
    """
    Remove irrelevant bins and normalize the FT data between 0 and 1

    The irrelevant bins are the offset bin and the bins that belong to
    the negative frequency spectrum.

    @data is a 2D (Nx2) np.array containing the output of a 1D, where N
    is the amount of samples per chirp, and each column represents the
    real and imaginary components, respectively.
    """
    # Remove offset and negative spectrum bins
    half_length = int(data.shape[0]/2)
    cropped = data[1:half_length, :]
    return cropped


def get_accuracy(signal, ref, simp=False):
    """
    Calculate the accuracy for a classification map

    Both @signal and @ref must be bool numpy arrays of the same shape
    """
    total_error = np.abs((signal.astype(np.int) - ref.astype(np.int))).sum()
    rel_error = total_error / signal.size
    accuracy = 1 - rel_error
    return accuracy


def get_precision(signal, ref, simp=False):
    """
    Calculate the accuracy for a classification map

    Both @signal and @ref must be bool numpy arrays of the same shape
    """
    total_error = np.abs((signal.astype(np.int) - ref.astype(np.int))).sum()
    rel_error = total_error / signal.size
    precision = 1 - rel_error
    return precision


def get_mse(signal, ref, simp=False):
    """
    Calculate the mean square error of the signal

    Both @signal and @ref must be numpy arrays of the same shape
    """
    signal = normalize(signal)
    ref = normalize(ref)
    quadratic_diff = ((signal - ref)**2)
    mse = quadratic_diff.sum() / ref.size
    return mse


def get_rmse(signal, ref):
    """
    Calculate the root mean square error of the signal

    Both @signal and @ref must be numpy arrays of the same shape
    """
    mse = get_mse(signal, ref)
    rmse = np.sqrt(mse)
    return rmse

def get_error_hist(signal, ref):
    """
    Get the histogram of the relative error along the output
    """
    # Add small value for avoding divide-by-zero situations
    signal = np.abs(simplify_ft(signal))
    ref = np.abs(simplify_ft(ref))
    # Get the absolute error of each bin.
    diff = np.abs(signal - ref)
    # Get the relative error by dividing by the ideal intensity
    # relative_error = diff / ref
    relative_error = diff
    return relative_error
