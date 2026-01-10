# SPDX-FileContributor: Thomas Schaap
#
# SPDX-License-Identifier: MIT

# This is just a copy of the example in the README.md, to ensure its correctness

# No formatting: the example is manually formatted for readability
# fmt: off

# ruff: noqa: E302,E305  # No minimal spacing between functions etc, for readability
# ruff: noqa: I001  # Imports are organized for readibility
# ruff: noqa: PLR2004  # Magic values are permissible in this example
# ruff: noqa: S101  # Yes, asserts are used for demonstration purposes

from collections.abc import Generator
from contextlib import contextmanager

from parallel.parallel import parallel

def to_string(number: int) -> str:
    return f"{number}"

three, four, five = (
    parallel(lambda: 1 + 2)()
    .parallel(to_string)(4)
    .parallel(lambda: 1 + 4)()
).results()

assert three == 3
assert four == "four"
assert five == 5

@contextmanager
def one_ctx() -> Generator[int, None, None]:
    yield 1

@contextmanager
def two_ctx() -> Generator[str, None, None]:
    yield "two"

with parallel(one_ctx()).parallel(two_ctx()) as (one, two):
    # one and two will have the correct type: int and str
    assert one == 1
    assert two == "two"
