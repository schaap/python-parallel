# SPDX-FileContributor: Thomas Schaap
#
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 Thomas Schaap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Wrapper around `ThreadPoolExecutor` that allows submitting tasks which are guaranteed to immediately start on a free
thread.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Semaphore
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    ParamSpec,
    TypeVar,
)

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
P = ParamSpec("P")

WORKERS_PER_POOL = 32


class _ParallelExecutorPool:
    """
    `ThreadPoolExecutor` wrapper that only submits calls if a thread from the pool is readily available.
    """

    def __init__(self) -> None:
        """
        Create a new `_ParallelExecutorPool`.

        The pool will have a set number of threads available. Submit calls to those threads using `run()` or
        `run_one()`.
        """
        self.pool = ThreadPoolExecutor(max_workers=WORKERS_PER_POOL)
        self.semaphore = Semaphore(WORKERS_PER_POOL)

    def run(self, *funcs: Callable[[], Any]) -> list[Future[Any]]:
        """
        Attempt to submit a number of calls to the `ThreadPoolExecutor`.

        Calls will only be submitted if threads are currently available on the `ThreadPoolExecutor`, making sure that
        the calls can indeed start running immediately. If less threads are available than there are calls, calls will
        be submitted in the order they are listed in `funcs`. The remainder of the calls will not be submitted at all.

        :param funcs: The calls to be submitted.
        :returns: The `Future` instances for the calls that were successfully submitted, in the same order as the calls
                  were listed.
        """
        requested = len(funcs)

        # Reserve threads
        reserved = 0
        while reserved < requested:
            if not self.semaphore.acquire(blocking=False):
                break
            reserved += 1

        # Submit functions
        futures = [self.pool.submit(func) for func in funcs[:reserved]]

        # Add callbacks to release the threads again
        for future in futures:
            future.add_done_callback(lambda _: self.semaphore.release())

        return futures

    def run_one(self, f: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Future[T] | None:
        """
        Attempt to submit a single call to the `ThreadPoolExecutor`.

        The call will only be submitted if a thread is currently available on the `ThreadPoolExecutor`.

        :param f: The function to call.
        :param args: Arguments to `f`.
        :param kwargs: Key-word arguments to `f`.
        :returns: A `Future` for `f` if the call was successfully submitted, or `None` if no thread was available.
        """
        # Reserve a thread, immediately bail if no threads are available
        if not self.semaphore.acquire(blocking=False):
            return None

        # Submit the function
        future: Future[T]
        try:
            future = self.pool.submit(f, *args, **kwargs)
        except:
            self.semaphore.release()
            raise

        # Add callback to release the thread again
        future.add_done_callback(lambda _: self.semaphore.release())

        return future


class ParallelExecutor:
    """
    Helper class for executing calls in parallel.

    This class governs a list of `_ParallelExecutorPool` instances and ensures that all requested calls will be
    submitted to readily available threads.
    """

    _pools: ClassVar[list[_ParallelExecutorPool]] = []
    _new_pool_mutex: ClassVar[Semaphore] = Semaphore(1)

    @classmethod
    def _add_pool(cls) -> _ParallelExecutorPool:
        """
        Thread-safely add a new `_ParallelExecutorPool`.

        :returns: The newly create pool.
        """
        with cls._new_pool_mutex:
            pool = _ParallelExecutorPool()
            cls._pools.append(pool)
            return pool

    @classmethod
    def _reset_pools(cls) -> None:
        """
        Thread-safely remove all `_ParallelExectorPool` instances.

        This *can* break any concurrent calls on `execute` or `execute_one`, because those have been optimized not to
        need the mutex when iterating the pools.

        This is for testing only.
        """
        with cls._new_pool_mutex:
            cls._pools.clear()

    @classmethod
    def _pool_count(cls) -> int:
        """
        Thread-safely count the number of `_ParallelExecutorPool` instances.

        This is for testing only.
        """
        with cls._new_pool_mutex:
            return len(cls._pools)

    @classmethod
    def execute(cls, f1: Callable[[], Any], /, *funcs: Callable[[], Any]) -> list[Future[Any]]:
        """
        Execute one or more calls in parallel.

        :param f1: The first call to execute in parallel.
        :param funcs: More calls to execute in parallel.
        :returns: The `Future` instances for the calls in `f1` and `funcs`, in the same order as the calls were listed.
        """
        remaining_funcs = [f1, *funcs]
        pool_index = 0
        futures: list[Future[Any]] = []

        # Try and submit functions to existing pools
        while remaining_funcs and pool_index < len(cls._pools):
            new_futures = cls._pools[pool_index].run(*remaining_funcs)
            remaining_funcs = remaining_funcs[len(new_futures) :]
            futures.extend(new_futures)
            pool_index += 1

        # Submit the rest to new pools
        while remaining_funcs:
            pool = cls._add_pool()
            new_futures = pool.run(*remaining_funcs)
            remaining_funcs = remaining_funcs[len(new_futures) :]
            futures.extend(new_futures)

        return futures

    @classmethod
    def execute_one(cls, f: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        """
        Execute a single call in parallel.

        :param f: The function to call.
        :param args: The arguments to `f`.
        :param kwargs: The keyword argument to `f`.
        :returns: The `Future` instance for calling `f`.
        """
        pool_index = 0

        # Try and submit function to an existing pool
        while pool_index < len(cls._pools):
            future = cls._pools[pool_index].run_one(f, *args, **kwargs)
            if future:
                return future
            pool_index += 1

        # Submit the function to a new pool
        pool = cls._add_pool()
        future = pool.run_one(f, *args, **kwargs)
        if future is None:
            raise RuntimeError("Could not acquire sempaphore on newly created pool")
        return future
