#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides what is required to call decorrelategrids.rollinglingreg as a CLI tool (argument parser and related)
"""
import argparse
import shlex
import sys


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
