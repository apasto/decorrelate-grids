#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute a rolling-window linear regression on 2 grids.
"""
import argparse
import shlex
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.stats import linregress
import multiprocessing as mp


def wrap_linregress(a, b, e_i, hw_y_i, hw_x_i):
    """
    Perform the window extraction and call scipy.stats.linregress() on the subarray (rolling window).
    Note that there is no assertion for 'not enough non-nan samples' here, this is left to linregress.

    Parameters
    ----------
    a : numpy.ndarray
        Array of the dependent variable A (A = f(B))
    b : numpy.ndarray
        Array of the dependent independent B
    e_i : tuple
        A tuple of two integers representing the central coordinates (row, column) of the submatrices
    hw_y_i : int
        Half-width of the window in the column direction
    hw_x_i : int
        Half-width of the window in the row direction

    Returns
    -------
    tuple
        A tuple with regression results (A = c0 * B + c1) and statistics
        c0 : float
            Intercept of the linear regression
        c1 : float
            Slope of the linear regression
        rv : float
            r-value: Pearson correlation coefficient.
        pv : float
            p-value: two-sided p-value for a hypothesis test where the null hypothesis is that the slope is zero.
        c0_stderr : float
            Standard error of the intercept.
        c1_stderr : float
            Standard error of the slope.
        w_no_nan_count : int
            Count of non-NaN elements in the window
    """
    w_a = a[
        e_i[0] - hw_y_i : e_i[0] + hw_y_i + 1,
        e_i[1] - hw_x_i : e_i[1] + hw_x_i + 1]
    w_b = b[
        e_i[0] - hw_y_i : e_i[0] + hw_y_i + 1,
        e_i[1] - hw_x_i : e_i[1] + hw_x_i + 1]
    w_no_nan = np.logical_not(
        np.logical_or(np.isnan(w_a), np.isnan(w_b))
        )
    w_no_nan_count = np.count_nonzero(w_no_nan)

    w_valid_a = w_a.to_numpy()[w_no_nan]
    w_valid_b = w_b.to_numpy()[w_no_nan]

    result = linregress(x=w_valid_a, y=w_valid_b)
    c0 = result.intercept
    c1 = result.slope
    rv = result.rvalue
    pv = result.pvalue
    c0_stderr = result.intercept_stderr
    c1_stderr = result.stderr

    return c0, c1, rv, pv, c0_stderr, c1_stderr, w_no_nan_count


if len(valid_elements_idx) < n_processes:
    n_processes = len(valid_elements_idx)

if n_processes == 1:
    for element in valid_elements_idx:
        (
            out["c0"]["c0"][element[0], element[1]],
            out["c1"]["c1"][element[0], element[1]],
            out["rv"]["rv"][element[0], element[1]],
            out["pv"]["pv"][element[0], element[1]],
            out["ie"]["ie"][element[0], element[1]],
            out["se"]["se"][element[0], element[1]],
            out["np"]["np"][element[0], element[1]]) = wrap_linregress(
                a=A, b=B, e_i=element,
                hw_x_i=window_halfwidth_x_i,
                hw_y_i=window_halfwidth_y_i)
else:
    pool = mp.Pool(n_processes)
    for element in valid_elements_idx:
        (
            out["c0"]["c0"][element[0], element[1]],
            out["c1"]["c1"][element[0], element[1]],
            out["rv"]["rv"][element[0], element[1]],
            out["pv"]["pv"][element[0], element[1]],
            out["ie"]["ie"][element[0], element[1]],
            out["se"]["se"][element[0], element[1]],
            out["np"]["np"][element[0], element[1]]) = pool.apply(
                wrap_linregress,
                kwds={
                    'a': A, 'b': B, 'e_i': element,
                    'hw_x_i': window_halfwidth_x_i,
                    'hw_y_i': window_halfwidth_y_i})

# window_no_nan_ratio
out["nr"]["nr"] = out["np"]["np"] / window_size

# save results (coefficients, quality) to grids
for key in out.keys():
    out[key].to_netcdf(
        path=out_filename[key],
        format="NETCDF4",
        encoding={key: {"zlib": True, "complevel": 9}})
