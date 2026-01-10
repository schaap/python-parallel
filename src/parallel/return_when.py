# SPDX-FileContributor: Thomas Schaap
#
# SPDX-License-Identifier: MIT

"""
A more strongly typed version of the `return_when` parameter to `Future.wait()`.
"""

from enum import Enum


class ReturnWhen(Enum):
    """When to return while waiting for a set of `Future`s to finish."""

    # Note that the values are compatible with the API design of `Future.wait()`

    FIRST_COMPLETED = "FIRST_COMPLETED"
    """Return as soon as one `Future` instance has completed."""
    # FIRST_COMPLETED is not actually used, currently, but included for completeness with respect to `Future.wait()`

    FIRST_EXCEPTION = "FIRST_EXCEPTION"
    """Return as soon as one `Future` instance has completed with an exception."""

    ALL_COMPLETED = "ALL_COMPLETED"
    """Only return when all the `Future` instances have completed (with a result or an exception)."""
