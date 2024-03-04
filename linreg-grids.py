#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute a rolling-window linear regression on 2 grids.
"""
import argparse
import multiprocessing as mp
import shlex
import sys
from pathlib import Path

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
                    rolling_linregress,
                    kwds={
                        'a': A.to_numpy(), 'b': B.to_numpy(), 'e_i': element,
                        'hw_x_i': window_halfwidth_x_i,
                        'hw_y_i': window_halfwidth_y_i})

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute a rolling-window linear regression on 2 grids. "
                    + "The variable in grid A is assumed to be, at least partially, "
                    + "in a linear relationship with grid B. "
                    + "A linear regression is performed in boxcar-shaped rolling windows, "
                    + "of given size, avoiding nodes which are closer to edges less than "
                    + "an half window, plus (optional) an edge of given width. "
                    + "Set either a 1:1 window half size, both x and y window half sizes, or "
                    + "the x/y aspect ratio of window sizes. "
                    + "The c0, c1 coefficients of a relationship in the form "
                    + "A = c0 + c1 * B are returned (as a grid) for each point "
                    + "in the input grids (which must cover the same region, "
                    + " with the same x, y intervals). "
                    + "Other returned grids are: rvalue, pvalue, c1 std error, c0 std error "
                    + "(see documentation of scipy.stats.linregress), "
                    + "the number of non-nan points in each window and the non-nan/total points ratio. "
                    + "All the returned grids have the same size and interval of grids A, B. "
                    + "The default form of the output grid files is: {path and stem of file A}_linreg_c0.grd "
                    + "(e.g. for c0)."
    )

    # arg_defs keys: argument name, values: kwargs to be passed to add_argument
    arg_defs = {
        # required: filename of grids A, B
        "A_filename": {
            "metavar": "A.grd",
            "type": str,
            "help": "filename of dependent variable grid A, (A = f(B))"
        },
        "B_filename": {
            "metavar": "B.grd",
            "type": str,
            "help": "filename of independent variable grid B"
        },
        # required: window half size along x
        "window_halfwidth_x": {
            "metavar": "HALF_SIZE",
            "type": float,
            "help": "half size of window, in grid units, along x"
        },
        # window half size along y
        "--window_halfwidth_y": {
            "metavar": "HALF_SIZE_Y",
            "type": float,
            "default": None,
            "help": "half size of window, in grid units, along y "
                    + "(set either this, aspect ratio, or neither (implies 1:1 aspect ratio)"
        },
        # window size x/y aspect ratio
        "--window_aspect_ratio": {
            "metavar": "ASPECT_RATIO",
            "type": float,
            "default": None,
            "help": "x/y aspect ratio of window size"
                    + "(set either this, half size along y, or neither (implies 1:1 aspect ratio))"
        },
        # edge widths
        "--edge_width_x_i": {
            "metavar": "EDGE_WIDTH",
            "type": int,
            "default": 0,
            "help": "number of grid nodes to be discarded at left and right edges, along x "
                    + "(default: 0)"
        },
        "--edge_width_y_i": {
            "metavar": "EDGE_WIDTH_Y",
            "type": int,
            "default": 0,
            "help": "number of grid nodes to be discarded at top and bottom edges, along y "
                    + " (default: 0)"
        },
        # cutoff min point per window: if less, no regression
        "--minpoints": {
            "metavar": "MIN_POINTS",
            "type": int,
            "default": 2,
            "help": "do not perform regression on node if the rolling window "
                    + "contains less than MIN_POINTS non-NaN points"
        },
        # switch: parameter metadata in output filename(s)
        "--metanames": {
            "action": "store_true",
            "help": "include the x and y half-window size (in grid units) "
                    + "in the output filenames, in the form "
                    + "'(...)linreg_x{windowsize_x}_y{windowsize_y}_(...)'. "
                    + "If the aspect ratio is provided, then the output is in the form: "
                    + "(...)linreg_x{windowsize_x}_(...)'."
        },
        # switch: global regression
        "--global_regression": {
            "action": "store_true",
            "help": "instead of rolling regression, fit one regression "
                    + "to the entire region (except edges)"
        },
        # small scale run on n elements
        "--small_scale_n": {
            "metavar": "SMALL_SCALE_N",
            "type": int,
            "default": None,
            "help": "run the regression only on the first N valid window centers"
                    + "(use case: small scale test run)"
        },
        # n processes (parallel with multiprocessing)
        "--n_processes": {
            "metavar": "N",
            "type": int,
            "default": 1,
            "help": "number of processes for parallel run "
                    + "(1 : serial, <1 : use all logical CPU cores)"
        },
        # output grids in multiple nc files
        "--split_out_files": {
            "action": "store_true",
            "help": "save each output grid in a separate nc file (default: as fields in a single file)"
        },
        # output path (defaults to same as A grid path)
        "--output_path": {
            "metavar": "OUT_PATH",
            "type": str,
            "default": None,
            "help": "path to save the output file(s) in (default: same directory as A grid)"
        },
        # output stuff
        "--out_basename_prefix": {
            "metavar": "OUT_PREFIX",
            "type": str,
            "default": None,
            "help": "prepend this to the name of output file(s) (default: A grid filename, with no extension)"
        }
    }

    for key, value in arg_defs.items():
        parser.add_argument(
            key,
            **value
        )

    # parse arguments, to dict
    args = vars(parser.parse_args())

    return args


def record_arguments():
    """
    concatenate how the script was called (script name and arguments)
    this will be stored in the 'history' attribute of the output files
    (gmt-like behaviour)
    shlex is used to escape single quotes, if there are any
    """
    return " ".join(shlex.quote(arg) if " " in arg else arg for arg in sys.argv)


def window_halfwidths_from_args(
        window_halfwidth_x, window_halfwidth_y=None, window_aspect_ratio=None):
    """
    Manage the rolling window halfwidths provided in arguments, with the following cases:
     - window_aspect_ratio provided, window_halfwidth_y computed accordingly
     - window_halfwidth_y provided
     - only window_halfwidth_x is provided, implies 1:1 aspect ratio
    """
    # x/y windows size or aspect ratio
    if window_halfwidth_y is not None and window_aspect_ratio is not None:
        # both provided: raise error
        raise ValueError(
            "Clashing parameters: "
            + "both window half width along y and window size aspect ratio provided. "
            + "Provide either the window size aspect ratio, size along y, "
            + "or neither (implies 1:1 aspect ratio)."
        )
    elif window_halfwidth_y is None and window_aspect_ratio is None:
        # neither provided: aspect ratio is 1:1
        window_halfwidth_y = window_halfwidth_x
        window_aspect_ratio = 1  # unused at the moment, keep record of it
    elif window_aspect_ratio is not None:
        window_halfwidth_y = window_halfwidth_x / window_aspect_ratio
    return window_halfwidth_x, window_halfwidth_y, window_aspect_ratio


def define_out_filenames(out_basename_prefix, out_dirname, windows_halfwidths_str, out_keys):
    """
    Define the output filenames, in a script call.
    Populates a dict of filenames as: dirname / prefix + '_linreg' + halfwidths_str + key + '.'grd'
    provided with a prefix, dirname, a string documenting the window size (can be empty).
    Iterates on all the provided 'out_keys', suitable both for 'one file' and multiple file output.
    """
    out_filename = {
        key: out_dirname / (out_basename_prefix + "_linreg_" + windows_halfwidths_str + str(key) + ".grd")
        for key in out_keys
    }
    return out_filename


def save_results(out, out_filename, zlib=True, complevel=9):
    """
    TODO: docstring
    save results (coefficients, quality) to grids
    """
    for key in out.keys():
        out[key].to_netcdf(
            path=out_filename[key],
            format="NETCDF4",
            encoding={key: {"zlib": zlib, "complevel": complevel}})


def main():
    parsed_args = parse_arguments()
    # manage window halfwidths arguments
    (
        parsed_args["window_halfwidth_x"],
        parsed_args["window_halfwidth_y"],
        parsed_args["window_aspect_ratio"]) = window_halfwidths_from_args(**parsed_args)

    if not parsed_args["split_out_files"]:
        raise NotImplementedError()  # TODO: implement one-file output

    # window half widths as string in filenames
    if parsed_args["metanames"]:
        if parsed_args["global_flag"]:
            windows_halfwidths_str = "GLOBAL_"
        elif parsed_args["window_aspect_ratio"] is None:
            # x and y sizes
            windows_halfwidths_str = "x{:d}_y{:d}_".format(
                round(parsed_args["window_halfwidth_x"]), round(parsed_args["window_halfwidth_y"]))
        else:
            # only x size
            windows_halfwidths_str = "x{:d}_".format(
                round(parsed_args["window_halfwidth_x"]))
    else:
        windows_halfwidths_str = ""

    # output path for output grids
    if parsed_args["output_path"] is None:
        # default to path to A
        out_dirname = parsed_args["A_filename"].resolve().parents[0]
    else:
        out_dirname = Path(parsed_args["output_path"])
    if not out_dirname.exists():
        raise FileNotFoundError(
            "Provided output path: \"{:s}\" not found.".format(str(out_dirname)))
        # TODO: "mkdir -p" here, error if fails

    # history field for output grids: record arguments in call
    out_history = record_arguments()

    # using 'load' instead of 'open'
    # this is an application-specific choice:
    # usually InSAR unwrap/dem grids are relatively small
    # resulting in no need for lazy-loading
    A = xr.load_dataarray(parsed_args["A_filename"])
    B = xr.load_dataarray(parsed_args["B_filename"])

    # linreg call here
    out = regression(A=A, B=B,
                     out_history=out_history, **parsed_args)

    # default prefix of output filenames: stem of A grid
    if parsed_args["out_basename_prefix"] is None:
        parsed_args["out_basename_prefix"] = Path(parsed_args["A_filename"]).stem
    out_filename = define_out_filenames(parsed_args["out_basename_prefix"], out_dirname,
                                        windows_halfwidths_str, out.keys())
    save_results(out, out_filename)
    return None


if __name__ == "__main__":
    main()
