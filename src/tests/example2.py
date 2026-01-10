# SPDX-FileContributor: Thomas Schaap
#
# SPDX-License-Identifier: MIT

# This is just a copy of the example in the README.md, to ensure its correctness

# No formatting: the example is manually formatted for readability
# fmt: off

# ruff: noqa: PERF401  # Not interested in performance in the example

from concurrent.futures import Future, wait

from parallel.parallel_executor import ParallelExecutor

futures: list[Future[int]] = []
for i in range(10):
    futures.append(ParallelExecutor.execute_one(lambda i: i, i))
done, not_done = wait(futures)
