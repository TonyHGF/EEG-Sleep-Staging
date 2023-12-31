"""Utility functions for time-related operations."""

__author__ = "Jonas Van Der Donckt"

from typing import Union

import pandas as pd


def timedelta_to_str(td: pd.Timedelta) -> str:
    """Construct a tight string representation for the given timedelta arg.

    Parameters
    ----------
    td: pd.Timedelta
        The timedelta for which the string representation is constructed

    Returns
    -------
    str:
        The tight string bounds of format '$d-$h$m$s.$ms'.

    """
    out_str = ""

    # Edge case if we deal with negative
    if td < pd.Timedelta(seconds=0):
        td *= -1
        out_str += "NEG"

    # Note: this must happen after the *= -1
    c = td.components
    if c.days > 0:
        out_str += f"{c.days}D"
    if c.hours > 0 or c.minutes > 0 or c.seconds > 0 or c.milliseconds > 0:
        out_str += "_" if len(out_str) else ""

    if c.hours > 0:
        out_str += f"{c.hours}h"
    if c.minutes > 0:
        out_str += f"{c.minutes}m"
    if c.seconds > 0 or c.milliseconds > 0:
        out_str += f"{c.seconds}"
        if c.milliseconds:
            out_str += f".{str(c.milliseconds / 1000).split('.')[-1].rstrip('0')}"
        out_str += "s"
    return out_str


def parse_time_arg(arg: Union[str, pd.Timedelta]) -> pd.Timedelta:
    """Parse the `window`/`stride` arg into a fixed set of types.

    Parameters
    ----------
    arg : Union[float, str, pd.Timedelta]
        The arg that will be parsed. \n
        * If the type is a `pd.Timedelta`, nothing will happen.
        * If the type is a `str`, `arg` should represent a time-string, and will be
          converted to a `pd.Timedelta`.

    Returns
    -------
    pd.Timedelta
        The parsed time arg

    Raises
    ------
    TypeError
        Raised when `arg` is not an instance of `float`, `int`, `str`, or
        `pd.Timedelta`.

    """
    if isinstance(arg, pd.Timedelta):
        return arg
    elif isinstance(arg, str):
        if arg.isnumeric():
            raise ValueError(f"time-string arg {arg} must contain a unit")
        return pd.Timedelta(arg)
    raise TypeError(f"arg type {type(arg)} is not supported!")
