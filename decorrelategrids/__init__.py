#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Computes a windowed (or 'rolling') regression on 2-D windows of 2-D arrays
with the aim of estimating the correlated component between the two
('decorrelation': estimating the correlated component and removing it,
as a 'data reduction' step to isolate the non-correlated component)
"""
from pathlib import Path

import xarray as xr

from .cliwrapper import parse_arguments, window_halfwidths_from_args, record_arguments, \
    define_out_filenames, save_results
from .rollinglinreg import regression


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
