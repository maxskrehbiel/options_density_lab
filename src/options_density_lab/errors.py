"""Exception types raised by the package for bad input data."""

from __future__ import annotations


class OptionsDensityError(Exception):
    """Base class for errors the command line reports as bad input (exit code 2)."""


class QuoteDataError(OptionsDensityError, ValueError):
    """The quote table is empty, lacks a column, or holds values that cannot be used."""


class CleaningError(OptionsDensityError, ValueError):
    """An expiry's quotes cannot support a forward and discount estimate."""
