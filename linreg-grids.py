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
import time

# TO DO: rolling linreg as function, separate the rest in functions and main()

# argument parser, description
parser = argparse.ArgumentParser(
    description="Compute a rolling-window linear regression on 2 grids. "
    + "The variable in grid A is assumed to be, at least partially, "
    + "in a linear relationship with grid B. "
    + "A linear regression is performed in boxcar-shaped rolling windows, "
    + "of given size, avoding nodes which are closer to edges less than "
    + "an half window, plus (optional) an edge of given width. "
    + "Set either a 1:1 window half size, both x and y window half sizes, or "
    + "the x/y aspect ratio of window sizes. "
    + "The c0, c1 coefficients of a relationship in the form "
    + "A = c0 + c1 * B are returned (as a grid) for each point "
    + "in the input grids (which must cover the same region, "
    + " with the same x, y intervals). "
    + "Other returned grids are: rvalue, pvalue, c1 std error, c0 std error "
    + "(see documentation of scipy.stats.linregress) "
    + "and the number of non-nan points in each window. "
    + "All the returned grids have the same size and interval of grids A, B. "
    + "The default form of the output grid files is: {path and stem of file A}_linreg_c0.grd "
    + "(e.g. for c0)."
)

# arguments: parser.add_argument 'dest' strings
# assigned to variables to be re-called later as dict keys
arg_keys_A = "A"
arg_keys_B = "B"
arg_keys_windowsize = "windowsize"
arg_keys_windowsize_y = "windowsize_y"
arg_keys_windowsize_aspect_ratio = "windowsize_aspect_ratio"
arg_keys_edgewidth = "edgewidth"
arg_keys_edgewidth_y = "edgewidth_y"
arg_keys_global = "global"
arg_keys_minpoints = "minpoints"
arg_keys_metanames = "metanames"
arg_keys_small_scale_n = "small_scale_n"
arg_keys_n_processes = "n_proc"

# arguments: filename of grids A, B
parser.add_argument(
    arg_keys_A,
    metavar="A.grd",
    type=str,
    help="filename of dependent variable grid A, (A = f(B))",
)
parser.add_argument(
    arg_keys_B,
    metavar="B.grd",
    type=str,
    help="filename of independent variable grid B",
)

# argument: window half size along x
parser.add_argument(
    arg_keys_windowsize,
    metavar="HALF_SIZE",
    type=float,
    help="half size of window, in grid units, along x",
)

# optional argument: window half size along y
parser.add_argument(
    "--" + arg_keys_windowsize_y,
    metavar="HALF_SIZE_Y",
    type=float,
    default=None,
    help="half size of window, in grid units, along y "
    + "(set either this, aspect ratio, or neither)",
)

# optional argument: window size x/y aspect ratio
parser.add_argument(
    "--" + arg_keys_windowsize_aspect_ratio,
    metavar="ASPECT_RATIO",
    type=float,
    default=None,
    help="x/y aspect ratio of window size"
    + "(set either this, half size along y, or neither)",
)

# optional argument: edge widths
parser.add_argument(
    "--" + arg_keys_edgewidth,
    metavar="EDGE_WIDTH",
    type=int,
    default=None,
    help="number of grid nodes to be discarded at edges "
    + "(along x and y, along x only if EDGE_WIDTH_Y is provided) [default: 0]",
)
parser.add_argument(
    "--" + arg_keys_edgewidth_y,
    metavar="EDGE_WIDTH_Y",
    type=int,
    default=None,
    help="number of grid nodes to be discarded at edges, along y "
    + " [default: 0 if EDGE_WIDTH is not set, otherwise equal to EDGE_WIDTH]",
)

# optional argument: discard if less than n points in window
parser.add_argument(
    "--" + arg_keys_minpoints,
    metavar="MIN_POINTS",
    type=int,
    default=None,
    help="do not perform regression on node if the rolling window "
    + "contains less than MIN_POINTS non-NaN points",
)

# optional argument: perform global regression
parser.add_argument(
    "--" + arg_keys_global,
    metavar="y/[n]",
    type=str,
    default="n",
    choices=["y", "Y", "n", "N"],
    help="instead of rolling regression, fit one regression "
    + "to the entire region (except edges)",
)

# optional argument: put parameter metadata in output filenames
parser.add_argument(
    "--" + arg_keys_metanames,
    metavar="y/[n]",
    type=str,
    default="n",
    choices=["y", "Y", "n", "N"],
    help="include the x and y half-window size (in grid units) "
    + "in the output filenames, in the form "
    + "'(...)linreg_x{windowsize_x}_y{windowsize_y}_(...)'. "
    + "If the aspect ratio is provided, then the output is in the form: "
    + "(...)linreg_x{windowsize_x}_(...)'.",
)

# optional argument: small scale run on n elements
parser.add_argument(
    "--" + arg_keys_small_scale_n,
    metavar="SMALL_SCALE_N",
    type=int,
    default=None,
    help="run the regression only on the first N valid points (window centers)",
)

# optional argument: n processes (parallel, multiprocessing)
parser.add_argument(
    "--" + arg_keys_n_processes,
    metavar="N_PROCESSES",
    type=int,
    default=1,
    help="number of processes for parallel run "
    + "(1 : serial, <1 : use all logical CPU cores)",
)

# parse arguments (convert argparse.Namespace to dict)
args = vars(parser.parse_args())

# concatenate how the script was called (script name and arguments)
# this will be stored in the 'history' attribute of the output files
# (gmt-like behaviour)
# shlex is used to escape single quotes, if there are any
args_string = " ".join(shlex.quote(arg) if " " in arg else arg for arg in sys.argv)

# assign arguments to variables
# input filenames
A_filename = Path(args[arg_keys_A])
B_filename = Path(args[arg_keys_B])
# window half-sizes and aspect ratio
window_halfwidth_x = args[arg_keys_windowsize]
window_halfwidth_y = args[arg_keys_windowsize_y]
window_aspect_ratio = args[arg_keys_windowsize_aspect_ratio]
# edge widths
edge_width_x_i = args[arg_keys_edgewidth]
edge_width_y_i = args[arg_keys_edgewidth_y]
# global regression flag
global_flag_arg = args[arg_keys_global]
if global_flag_arg.lower() == "y":
    global_flag = True
else:
    global_flag = False
# minimum points
min_points = args[arg_keys_minpoints]
if min_points is None or min_points < 2:
    min_points = 2  # minimum requirement for linear regression
# window size in output filenames
metanames_arg = args[arg_keys_metanames]
if metanames_arg.lower() == "y":
    metanames = True
else:
    metanames = False
# small scale run on first n elements
small_scale_n = args[arg_keys_small_scale_n]
# number of processes and serial/parallel switch
n_processes = args[arg_keys_n_processes]
if n_processes < 1:
    n_processes = mp.cpu_count()

# x/y windows size or aspect ratio
if window_halfwidth_y is not None and window_aspect_ratio is not None:
    # both provided: raise error
    raise ValueError(
        "Both window half width along y and window size aspect ratio provided. "
        + "Provide either the window size aspect ratio, size along y, "
        + "or neither (implies 1:1 aspect ratio)."
    )
elif window_halfwidth_y is None and window_aspect_ratio is None:
    # neither provided: aspect ratio is 1:1
    window_halfwidth_y = window_halfwidth_x
    window_aspect_ratio = 1  # unused at the moment, keep record of it
elif window_aspect_ratio is not None:
    window_halfwidth_y = window_halfwidth_x / window_aspect_ratio

# window half widths as string for filenames
if metanames:
    if window_aspect_ratio is None:
        # x and y sizes
        windows_halfwidths_str = "x{:d}_y{:d}_".format(
            round(window_halfwidth_x), round(window_halfwidth_y))
    else:
        # only x size
        windows_halfwidths_str = "x{:d}_".format(
            round(window_halfwidth_x))
else:
    windows_halfwidths_str = ""

# edge width
if edge_width_x_i is None and edge_width_y_i is None:
    edge_width_x_i = 0
    edge_width_y_i = 0
elif edge_width_x_i is None and edge_width_y_i is not None:
    edge_width_x_i = 0
elif edge_width_y_i is None:
    edge_width_y_i = edge_width_x_i

# global regression: not implemented yet
if global_flag:
    raise NotImplementedError("Global regression is not implemented.")
    # TODO: implement global regression
    # TODO: if global_flag and metanames, add 'global' in filenames

# output path for output grids
out_dirname = A_filename.resolve().parents[0]
# prefix of output file: basename of grid A filename
out_basename_prefix = A_filename.stem
# filenames for output grids
out_filename = {}
# returned by regression:
out_filename["c0"] = out_dirname / (
    out_basename_prefix + "_linreg_" + windows_halfwidths_str + "c0.grd")
out_filename["c1"] = out_dirname / (
    out_basename_prefix + "_linreg_" + windows_halfwidths_str + "c1.grd")
out_filename["rv"] = out_dirname / (
    out_basename_prefix + "_linreg_" + windows_halfwidths_str + "rv.grd")
out_filename["pv"] = out_dirname / (
    out_basename_prefix + "_linreg_" + windows_halfwidths_str + "pv.grd")
out_filename["se"] = out_dirname / (
    out_basename_prefix + "_linreg_" + windows_halfwidths_str + "se.grd")
out_filename["ie"] = out_dirname / (
    out_basename_prefix + "_linreg_" + windows_halfwidths_str + "ie.grd")
# number and ratio of non-nan points in window:
out_filename["np"] = out_dirname / (
    out_basename_prefix + "_linreg_" + windows_halfwidths_str + "np.grd")
out_filename["nr"] = out_dirname / (
    out_basename_prefix + "_linreg_" + windows_halfwidths_str + "nr.grd")
# title fields for output grids
out_titles = {
    "c0": "Regression intercept",
    "c1": "Regression slope",
    "rv": "R-value of regression",
    "pv": "p-value of regression",
    "se": "Standard error of slope",
    "ie": "Standard error of intercept",
    "np": "Number of non-nan points in rolling regression window",
    "nr": "Ratio of non-nan points and total number of points in window",
}
# history field for output grids
out_history = args_string

# using 'load' instead of 'open'
# application-specific choice: usually InSAR unwrap/dem grids are relatively small
# resulting in no need for lazy-loading
A = xr.load_dataarray(A_filename)
B = xr.load_dataarray(B_filename)

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

# store grid intervals
x_int = A_x_int_range[0]
y_int = A_y_int_range[0]

# dict of xr.DataArray for each of regression results
# same shape as data in A
# NaN-filled -> not assigned nodes will stay NaN
# TO DO: use a single dataset with all the output variables
# it can be then used in gmt using the filename?variable notation
out = {}
for key in out_filename.keys():
    out[key] = xr.Dataset(
        data_vars={key: (["y", "x"], np.full(A.shape, np.nan))},
        coords=dict(
            x=(["x"], A.x.values),
            y=(["y"], A.y.values),
        ),
        attrs={"title": out_titles[key], "history": out_history},
    )

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
    valid_elements_idx = valid_elements_idx[:small_scale_n]

# number of elements in window, used for ratio of valid elements
window_size = (window_halfwidth_x_i * 2 + 1) * (window_halfwidth_y_i * 2 + 1)


# temp here
def wrap_linregress(a, b, e_i, hw_x_i, hw_y_i):
    w_a = a[
        e_i[0] - hw_x_i : e_i[0] + hw_x_i + 1,
        e_i[1] - hw_y_i : e_i[1] + hw_y_i + 1]
    w_b = b[
        e_i[0] - hw_x_i : e_i[0] + hw_x_i + 1,
        e_i[1] - hw_y_i : e_i[1] + hw_y_i + 1]
    w_no_nan = np.logical_not(
        np.logical_or(np.isnan(w_a), np.isnan(w_b))
        )
    w_no_nan_count = np.count_nonzero(w_no_nan)
    # w_no_nan_ratio = w_no_nan_count / window_size  # TODO: move outside

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
