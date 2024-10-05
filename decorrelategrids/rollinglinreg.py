#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute a rolling-window linear regression on 2 grids.
"""
import multiprocessing as mp

import numpy as np
import xarray as xr
from scipy.stats import linregress


def wrap_linregress(a, b):
    """
    Call scipy.stats.linregress() on two vectors a (linregress x) and b (linregress y), then return the regression results
    in a dict - to be then used to populate the regression output.
    This serves as a placeholder to implement different (or any) regression methods,
    converting the returned output to the dict used here.
    """
    # nan discarding
    no_nan = np.logical_not(
        np.logical_or(np.isnan(a), np.isnan(b))
    )

    result = linregress(x=a[no_nan], y=b[no_nan])
    return {
        'c0': result.intercept, 'c1': result.slope,
        'rv': result.rvalue, 'pv': result.pvalue,
        'c0_stderr': result.intercept_stderr, 'c1_stderr': result.stderr,
        'no_nan_count': np.count_nonzero(no_nan)}


def rolling_linregress(a, b, e_i, hw_y_i, hw_x_i):
    """
    Perform the window extraction and call the linear regression wrapper on the subarray (rolling window).
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
    w_a = a[e_i[0] - hw_y_i: e_i[0] + hw_y_i + 1, e_i[1] - hw_x_i: e_i[1] + hw_x_i + 1]
    w_b = b[e_i[0] - hw_y_i: e_i[0] + hw_y_i + 1, e_i[1] - hw_x_i: e_i[1] + hw_x_i + 1]

    r = wrap_linregress(a=w_a, b=w_b)

    return r["c0"], r["c1"], r["rv"], r["pv"], r["c0_stderr"], r["c1_stderr"], r["no_nan_count"]


def _generate_pool_arguments(A, B, valid_elements_idx, hw_x_i, hw_y_i):
    """
    Generate the arguments (as tuples of arguments) for the call to
    rolling_linregress in pool.imap_unordered.
    This is equivalent (in practice, not in implementation) to
    the 'kwds' argument in pool.apply:
        pool.apply(rolling_linregress,
            kwds={
                'a': A.to_numpy(), 'b': B.to_numpy(), 'e_i': element,
                'hw_x_i': window_halfwidth_x_i,
                'hw_y_i': window_halfwidth_y_i
                }
            )
    """
    for element in valid_elements_idx:
        yield (A.to_numpy(), B.to_numpy(), element, hw_x_i, hw_y_i)


def regression(A, B, window_halfwidth_x=None, window_halfwidth_y=None,
               edge_width_x_i=0, edge_width_y_i=0,
               global_regression=False, out_history=None, small_scale_n=None, n_processes=1):
    """
    Call the rolling regression, provided with the two dataarrays,
    windowing and edge parameters, number of processes.
    """

    if window_halfwidth_x is None:
        raise ValueError()
    if window_halfwidth_y is None:
        window_halfwidth_y = window_halfwidth_x

    # assert for same sampling on A, B grids
    if not np.all(np.isclose(A.x.values, B.x.values)):
        raise ValueError("Sampling along x in A, B grids is not equal")
    if not np.all(np.isclose(A.y.values, B.y.values)):
        raise ValueError("Sampling along y in A, B grids is not equal")

    # grid intervals: assert for (close) equality of min and max interval
    # testing grid A is enough, since we already tested for 'same sampled coords'

    A_x_int = np.diff(A.x.values)
    A_x_int_range = (A_x_int.min(), A_x_int.max())
    if not np.isclose(A_x_int_range[0], A_x_int_range[1]):
        raise ValueError("Grid sampling along x is not constant")

    A_y_int = np.diff(A.y.values)
    A_y_int_range = (A_y_int.min(), A_y_int.max())
    if not np.isclose(A_y_int_range[0], A_y_int_range[1]):
        raise ValueError("Grid sampling along y is not constant")

    # dict of xr.DataArray for each of regression results
    # same shape as data in A
    # NaN-filled -> not assigned nodes will stay NaN
    # TODO: use a single dataset with all the output variables
    # it can be then used in gmt using the filename?variable notation
    out_titles = {
        "c0": "Regression intercept",
        "c1": "Regression slope",
        "rv": "r-value of regression",
        "pv": "p-value of regression",
        "se": "Standard error of slope",
        "ie": "Standard error of intercept",
        "np": "Number of non-nan points in rolling regression window",
        "nr": "Ratio of non-nan points and total number of points in window",
    }
    out = {}
    for key, value in out_titles.items():
        out[key] = xr.Dataset(
            data_vars={key: (["y", "x"], np.full(A.shape, np.nan))},
            coords=dict(
                x=(["x"], A.x.values),
                y=(["y"], A.y.values),
            ),
            attrs={"title": value, "history": out_history},
        )

    if not global_regression:
        # grid intervals
        x_int = A_x_int_range[0]
        y_int = A_y_int_range[0]

        # window half width from units to number-of-nodes
        window_halfwidth_x_i = round(window_halfwidth_x / x_int)
        window_halfwidth_y_i = round(window_halfwidth_y / y_int)

        # number of grid elements to be discarded at the edges
        edge_fullwidth_x_i = edge_width_x_i + window_halfwidth_x_i
        edge_fullwidth_y_i = edge_width_y_i + window_halfwidth_y_i

        # create a "discard this element mask"
        # mask nan elements in A or B
        discard = np.logical_or(np.isnan(A), np.isnan(B))
        # mask edge elements
        # x, left edge
        discard[:, 0:edge_fullwidth_x_i] = True
        # x, right edge
        discard[:, -edge_fullwidth_x_i:] = True
        # y, bottom edge
        discard[0:edge_fullwidth_y_i, :] = True
        # y, top edge
        discard[-edge_fullwidth_y_i:, :] = True

        # get indices in a (m, n) fashion of not-to-be-discarded points
        # to center each rolling linreg on
        valid_elements_idx = np.argwhere(np.logical_not(discard.values))

        if small_scale_n is not None:
            valid_elements_idx = valid_elements_idx[:small_scale_n, :]

        # number of elements in window, used for ratio of non-NaN samples in window
        window_size = (window_halfwidth_x_i * 2 + 1) * (window_halfwidth_y_i * 2 + 1)

        # avoid more processes than window centers
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
                    out["np"]["np"][element[0], element[1]]) = rolling_linregress(
                    a=A.to_numpy(), b=B.to_numpy(), e_i=element,
                    hw_x_i=window_halfwidth_x_i,
                    hw_y_i=window_halfwidth_y_i)
        else:
            with mp.Pool(n_processes) as pool:
                results = pool.imap_unordered(
                    rolling_linregress,
                    _generate_pool_arguments(
                        A, B, valid_elements_idx, window_halfwidth_x_i, window_halfwidth_y_i))

            for element, result in zip(valid_elements_idx, results):
                out["c0"]["c0"][element[0], element[1]], out["c1"]["c1"][element[0], element[1]], \
                    out["rv"]["rv"][element[0], element[1]], out["pv"]["pv"][element[0], element[1]], \
                    out["ie"]["ie"][element[0], element[1]], out["se"]["se"][element[0], element[1]], \
                    out["np"]["np"][element[0], element[1]] = result

        # no_nan : window size ratio
        out["nr"]["nr"] = out["np"]["np"] / window_size
    else:  # global regression
        r = wrap_linregress(a=A.to_numpy(), b=B.to_numpy())
        # no_nan : global size ratio
        no_nan_ratio = r["no_nan_count"] / A.size
        out_shape = A.shape
        out["c0"]["c0"] = (["y", "x"], np.full(shape=out_shape, fill_value=r["c0"]))
        out["c1"]["c1"] = (["y", "x"], np.full(shape=out_shape, fill_value=r["c1"]))
        out["rv"]["rv"] = (["y", "x"], np.full(shape=out_shape, fill_value=r["rv"]))
        out["pv"]["pv"] = (["y", "x"], np.full(shape=out_shape, fill_value=r["pv"]))
        out["ie"]["ie"] = (["y", "x"], np.full(shape=out_shape, fill_value=r["c0_stderr"]))
        out["se"]["se"] = (["y", "x"], np.full(shape=out_shape, fill_value=r["c1_stderr"]))
        out["np"]["np"] = (["y", "x"], np.full(shape=out_shape, fill_value=r["no_nan_count"]))
        out["nr"]["nr"] = (["y", "x"], np.full(shape=out_shape, fill_value=no_nan_ratio))

    return out
