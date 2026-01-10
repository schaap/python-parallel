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

# ruff: noqa: S101  # assert is allowed in tests
# ruff: noqa: S311  # cryptographic random is not needed in tests

"""
Tests for `_ParallelExecutorPool` and `ParallelExecutor`.
"""

from __future__ import annotations

from concurrent.futures import Future, wait
from functools import partial
from random import Random
from threading import Barrier
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import pytest

from parallel.parallel_executor import WORKERS_PER_POOL, ParallelExecutor
from parallel.parallel_executor import (
    _ParallelExecutorPool as ParallelExecutorPool,  # noqa: PLC2701  # Private import allowed for its own tests
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

T = TypeVar("T")


JOB_COUNT_LISTS = (
    # Jobs are accepted one by one
    [*[1 for _ in range(WORKERS_PER_POOL)], 1],
    # Jobs are accepted in pairs
    [*[2 for _ in range((WORKERS_PER_POOL + 1) // 2)], 1],
    # Jobs are accepted in threes
    [*[3 for _ in range((WORKERS_PER_POOL + 2) // 3)], 1],
    # The entire capacity is accepted in one call
    [WORKERS_PER_POOL, 1],
    # The last slot can be filled by a single matching request
    [WORKERS_PER_POOL - 1, 1, 1],
    # The last two slots can be filled by a single matching request
    [WORKERS_PER_POOL - 2, 2, 1],
    # The last slot can be filled by a partially fulfulled request
    [WORKERS_PER_POOL - 1, 2],
    # The last two slots can be filled by a partially fulfilled request
    [WORKERS_PER_POOL - 2, 3],
    # The entire capacity can be filled by a single request that is too large
    [WORKERS_PER_POOL + 1],
    # The entire capacity can be filled by a single request that is far too large
    [WORKERS_PER_POOL + 2],
    # More than one job from a request can be dropped
    [WORKERS_PER_POOL - 2, 4],
    # Requests of different sizes can be dropped entirely
    [WORKERS_PER_POOL, 1, 1, 2, 3, WORKERS_PER_POOL, WORKERS_PER_POOL + 1],
)
"""
Lists of numbers of jobs to run in paralell. The selection is based on the capabilities and limits of
`_ParallelExecutorPool`, designed primarily for `test_parallel_executor_pool_run_scheduling_basics_work`.
"""

ARGS_LISTS: list[tuple[Any, ...]] = [
    ((),),
    ((1,),),
    ((1, 2, 3),),
    (((),),),
    (((1,),),),
    (({"one": 1}),),
]
"""Parametrization options for parametrization of *args."""

KWARGS_LISTS: list[dict[str, Any]] = [
    {},
    {"one": 1},
    {"one": 1, "two": 2.0},
    {"args": ()},
    {"args": ("different")},
    {"kwargs": ("different")},
    {"args": {"one": 1}},
    {"kwargs": {"one": 1}},
]
"""Parametrization options for parametrization of **kwargs."""


@pytest.fixture
def clear_parallel_executor() -> Generator[None, None, None]:
    """
    Clears the static pool of pools of `ParallelExecutor`, before and after a test.
    """
    try:
        ParallelExecutor._reset_pools()  # noqa: SLF001  # Private access allowed from test
        yield
    finally:
        ParallelExecutor._reset_pools()  # noqa: SLF001  # Private access allowed from test


def wait_for_signal(barrier: Barrier, result: T) -> T:
    """
    Simple function to be scheduled. This will wait for a signal before finishing.

    :param barrier: The `Barrier` to wait for.
    :param result: The value that will be returned when finishing.
    :returns: `result`
    """
    barrier.wait(timeout=2)
    return result


# We allow Any here, because we need actually need it
def return_args_and_kwargs(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:  # noqa: ANN401
    """
    Simple function that returns its arguments and keyword arguments.

    :returns: The arguments and keyword arguments passed to this function.
    """
    return (args, kwargs)


def schedule_with_run_one(pool: ParallelExecutorPool, *jobs: Callable[[], Any]) -> list[Future[Any]]:
    """
    Schedule multiple jobs on a pool using `run_one`.

    :param pool: The pool to schedule the jobs on.
    :param jobs: The jobs to schedule.
    :returns: The created `Future`s.
    """
    have_none = False
    futures: list[Future[Any]] = []
    for job in jobs:
        future = pool.run_one(job)
        if future is None:
            have_none = True
        else:
            assert not have_none
            futures.append(future)
    return futures


def schedule_with_execute_one(*jobs: Callable[[], Any]) -> list[Future[Any]]:
    """
    Schedule multiple jobs on `ParallelPool` using `execute_one`.

    :param jobs: The jobs to schedule.
    :returns: The created `Future`s.
    """
    futures: list[Future[Any]] = []
    for job in jobs:
        future = ParallelExecutor.execute_one(job)
        assert future is not None
        futures.append(future)
    return futures


def finish_jobs(futures: list[Future[Any]], *, barrier: Barrier | None = None) -> None:
    """
    Helper for finishing a number of jobs.

    :param futures: If set, the list of futures to wait for. The result of each future must be None.
    :param barriers: If set, the barrier to wait for.
    """
    if barrier:
        barrier.wait(timeout=2)

    done, not_done = wait(futures, timeout=2, return_when="ALL_COMPLETED")
    assert len(done) == len(futures)
    assert not not_done

    assert all(future.result() is None for future in futures)


@pytest.mark.parametrize("job_counts", JOB_COUNT_LISTS)
def test_parallel_executor_pool_run_scheduling_basics_work(job_counts: list[int]) -> None:
    """
    Verify that a parallel executor pool's `run` accepts jobs until it's completely full.

    Also verifies that dropped requests are not added after all.
    Also verifies that threads are given back after all requests have been handled.

    :param job_counts: The list of requests to be made to `run`, in number of jobs per request.
    """
    assert sum(job_counts) > WORKERS_PER_POOL, "Please provide too many jobs, so saturation can be detected"

    pool = ParallelExecutorPool()
    barrier = Barrier(WORKERS_PER_POOL + 1)  # All jobs will wait until we trigger the barrier from the test

    scheduled_job_count = 0
    all_futures: list[Future[Any]] = []
    unschedule_job_count = 0

    # Attempt to add all the jobs as per job_counts
    for job_count in job_counts:
        # Create the jobs for one call
        jobs = [lambda: wait_for_signal(barrier, None) for _ in range(job_count)]

        # Attempt to schedule them
        futures = pool.run(*jobs)
        all_futures += futures

        # Verify the call's result
        expected_future_count = min(scheduled_job_count + job_count, WORKERS_PER_POOL) - scheduled_job_count
        assert len(futures) == expected_future_count

        # Track the totals
        scheduled_job_count += expected_future_count
        expected_unscheduled = job_count - expected_future_count
        unschedule_job_count += expected_unscheduled

    # Verify result of all calls
    assert len(all_futures) == WORKERS_PER_POOL
    assert unschedule_job_count == sum(job_counts) - WORKERS_PER_POOL
    assert not any(future.done() for future in all_futures)

    # Trigger the barrier and wait for all jobs to have finished
    finish_jobs(all_futures, barrier=barrier)

    # Wait a short moment to allow the unscheduled jobs to 'somehow' be scheduled.
    sleep(0.2)

    # Add maximum number of jobs. This should work as the threads have been returned, and no unscheduled jobs should
    # 'somehow' have become scheduled.
    barrier = Barrier(WORKERS_PER_POOL)  # The jobs only wait for each other
    all_futures = pool.run(*(lambda: wait_for_signal(barrier, None) for _ in range(WORKERS_PER_POOL)))
    assert len(all_futures) == WORKERS_PER_POOL

    # Wait for all jobs to have finished
    finish_jobs(all_futures)


@pytest.mark.parametrize("first_job", range(WORKERS_PER_POOL))
@pytest.mark.parametrize("use_run_one", [False, True])
def test_parallel_executor_pool_run_threads_immediately_returned(*, first_job: int, use_run_one: bool) -> None:
    """
    Verify that a parallel executor pool schedules jobs such that the threads are
    returned to the pool as soon as they become available.

    :param first_job: The index of the job that will finish first.
    :param use_run_one: Whether to use `run_one` instead of `run` to schedule the jobs.
    """
    assert 0 <= first_job < WORKERS_PER_POOL

    pool = ParallelExecutorPool()
    barrier_one = Barrier(2)  # We want the one job to wait until we trigger it from the test
    barrier_rest = Barrier(WORKERS_PER_POOL + 1)  # We'll have all other jobs wait until we trigger them from the test
    schedule_call = partial(schedule_with_run_one, pool) if use_run_one else pool.run

    # Set up the jobs. We can just replace one job with a special one, as they're all (unexecuted) lambda's.
    jobs = [lambda: wait_for_signal(barrier_rest, None) for _ in range(WORKERS_PER_POOL)]
    jobs[first_job] = lambda: wait_for_signal(barrier_one, None)

    # Schedule all the jobs
    all_futures = schedule_call(*jobs)

    # Separate the special job's result from the result
    first_future = all_futures[first_job]
    all_futures.remove(all_futures[first_job])

    # Trigger the single barrier to finish the one job
    barrier_one.wait(timeout=2)
    assert first_future.result(timeout=2) is None
    assert all(not future.done() for future in all_futures)

    # Add a new job to show that the thread was immediately returned, add it to the rest of the jobs
    futures = schedule_call(lambda: wait_for_signal(barrier_rest, None))
    assert len(futures) == 1
    assert not futures[0].done()
    all_futures.append(futures[0])

    # Trigger the big barrier to finish all jobs
    finish_jobs(all_futures, barrier=barrier_rest)


@pytest.mark.parametrize("use_run_one", [False, True])
def test_parallel_executor_pool_run_threads_never_overcommit(*, use_run_one: bool) -> None:
    """
    Verify that a parallel executor pool's `run` never returns too many threads.

    This can't truly be proven (we're proving a negative). A statistical test is performed, instead.

    :param use_run_one: Whether to use `run_one` instead of `run` to schedule the jobs.
    """

    pool = ParallelExecutorPool()
    random = Random()
    schedule_call = partial(schedule_with_run_one, pool) if use_run_one else pool.run

    class Job:
        def __init__(self) -> None:
            self._barrier = Barrier(2)  # Each job waits until it's triggered from the test
            self.future: Future[None] | None = None

        def wait_for_signal(self) -> None:
            return wait_for_signal(self._barrier, None)

        def finish(self) -> None:
            assert self.future
            self._barrier.wait(timeout=2)
            return self.future.result(timeout=2)

    def add_jobs(count: int, expect_success: int) -> list[Job]:
        """
        Create and schedule `Job`s on the `pool`.

        :param count: The number of jobs to schedule. This may be more than fit in `pool`.
        :param expect_success: The exact expected number of jobs to successfully be scheduled.
        """
        jobs = [Job() for _ in range(count)]
        futures = schedule_call(*[job.wait_for_signal for job in jobs])

        # Track the futures of the jobs
        assert len(futures) == expect_success
        for index in range(expect_success):
            jobs[index].future = futures[index]
        return jobs[:expect_success]

    # Schedule the initial jobs
    jobs = add_jobs(WORKERS_PER_POOL, WORKERS_PER_POOL)

    for finish_job_count in range(1, WORKERS_PER_POOL + 1):
        # Choose a number of random jobs to finish
        random.shuffle(jobs)
        jobs_to_finish = jobs[:finish_job_count]
        jobs = jobs[finish_job_count:]

        # Finish the chosen jobs by triggering their barriers
        for job in jobs_to_finish:
            job.finish()

        # Add new jobs. This adds one too many, verifying that the correct number of threads was released.
        jobs += add_jobs(finish_job_count + 1, finish_job_count)

    # Finish all the jobs as cleanup
    for job in jobs:
        job.finish()


def test_parallel_executor_pool_run_returned_futures_match_request() -> None:
    """
    Verify that a parallel executor pool's `run` returns its `Future`s in the same order as the request.

    This can't truly be proven (it's effectively proving a negative), so we provide a statistical test, instead.
    """
    pool = ParallelExecutorPool()
    random = Random()
    job_counts = random.choices(range(1, WORKERS_PER_POOL + 1), k=32)

    for job_count in job_counts:
        # Set up the jobs, shuffle them for added randomization
        barrier = Barrier(job_count)  # The jobs only wait for each other
        jobs = [
            (index, lambda barrier=barrier, index=index: wait_for_signal(barrier, index)) for index in range(job_count)
        ]
        random.shuffle(jobs)

        # Schedule the jobs
        futures = pool.run(*[job[1] for job in jobs])

        # Wait for the jobs to have finished
        done, not_done = wait(futures, timeout=2, return_when="ALL_COMPLETED")
        assert len(done) == job_count
        assert not not_done

        # Verify that the futures match the jobs: the original index of the job has to match the future's result
        assert all(
            job_with_future[0][0] == job_with_future[1].result() for job_with_future in zip(jobs, futures, strict=True)
        )


def test_parallel_executor_pool_run_mixed_returns() -> None:
    """
    Verify that a single call to parallel executor pool's `run` can handle mixed return types in its requests.
    """
    # NOTE: The jobs in this test are written out instead of being constructed from a list. This is done to make sure
    # that the typing of each scheduled job is specific and not Callable[[], Any]
    value1 = 42
    value2 = "second"
    value3 = 3
    value4 = 4.4

    job_count = 4

    pool = ParallelExecutorPool()
    barrier = Barrier(job_count)  # The jobs only wait for each other

    # Add some jobs with different result types
    futures = pool.run(
        lambda: wait_for_signal(barrier, value1),
        lambda: wait_for_signal(barrier, value2),
        lambda: wait_for_signal(barrier, value3),
        lambda: wait_for_signal(barrier, value4),
    )

    # Wait for the jobs to have finished
    done, not_done = wait(futures, timeout=2, return_when="ALL_COMPLETED")
    assert len(done) == job_count
    assert not not_done

    # Check that the futures contain the correct values
    assert futures[0].result() == value1
    assert futures[1].result() == value2
    assert futures[2].result() == value3
    assert futures[3].result() == value4


@pytest.mark.parametrize("args", ARGS_LISTS)
@pytest.mark.parametrize("kwargs", KWARGS_LISTS)
def test_parallel_executor_pool_run_one_passes_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """
    Verify that a parallel executor pool's `run_one` passes arguments to the call.

    :param args: The positional arguments to pass.
    :param kwargs: The keyword arguments to pass.
    """
    pool = ParallelExecutorPool()

    future = pool.run_one(return_args_and_kwargs, *args, **kwargs)
    assert future is not None

    result = future.result(timeout=2)
    assert result[0] == args
    assert result[1] == kwargs


def test_parallel_executor_pool_run_one_fills_all_slots() -> None:
    """
    Verify that a parallel executor pool's `run_one` can fill all slots of the pool.
    """
    pool = ParallelExecutorPool()
    barrier = Barrier(WORKERS_PER_POOL + 1)  # All jobs will wait until they're triggered from the test

    # Add jobs to fill all slots
    futures: list[Future[None]] = []
    for _ in range(WORKERS_PER_POOL):
        future = pool.run_one(wait_for_signal, barrier, None)
        assert future is not None
        futures.append(future)

    # All slots have been used
    assert pool.run_one(wait_for_signal, barrier, None) is None

    # Trigger the barrier to finish all jobs again
    finish_jobs(futures, barrier=barrier)


@pytest.mark.parametrize("job_counts", JOB_COUNT_LISTS)
@pytest.mark.parametrize("use_execute_one", [False, True])
def test_parallel_executor_execute_schedules_all(
    *,
    job_counts: list[int],
    use_execute_one: bool,
    clear_parallel_executor: None,  # noqa: ARG001  # Fixture
) -> None:
    """
    Verify that a parallel executor will schedule all jobs, even though they're too many for one pool.

    :param job_counts: The list of requests to schedule, in number of jobs per request.
    :param use_execute_one: Whether to use `execute_one` to schedule jobs instead of `execute`.
    :param clear_parallel_executor: Ensures a clear state to test with.
    """
    total_jobs = sum(job_counts)
    barrier = Barrier(total_jobs + 1)  # All jobs will wait until triggered by the test
    schedule_call = schedule_with_execute_one if use_execute_one else ParallelExecutor.execute

    # Create the jobs in batches as stated in job_counts
    all_futures: list[Future[Any]] = []
    for job_count in job_counts:
        futures = schedule_call(*[lambda: wait_for_signal(barrier, None) for _ in range(job_count)])
        assert len(futures) == job_count
        all_futures += futures

    # None of the jobs should have finished yet
    assert not any(future.done() for future in all_futures)

    # Trigger the barrier and finish all jobs
    finish_jobs(all_futures, barrier=barrier)


@pytest.mark.parametrize(
    "job_count",
    [
        WORKERS_PER_POOL - 1,
        WORKERS_PER_POOL,
        WORKERS_PER_POOL + 1,
        WORKERS_PER_POOL * 2 - 1,
        WORKERS_PER_POOL * 2,
        WORKERS_PER_POOL * 2 + 1,
        WORKERS_PER_POOL * 5 - 1,
        WORKERS_PER_POOL * 5,
        WORKERS_PER_POOL * 5 + 1,
    ],
)
@pytest.mark.parametrize("use_execute_one", [False, True])
def test_parallel_executor_execute_adds_pools(
    *,
    job_count: int,
    use_execute_one: bool,
    clear_parallel_executor: None,  # noqa: ARG001  # Fixture
) -> None:
    """
    Verify that a parallel executor will add pools as needed to schedule jobs, but no more than needed.

    Also verifies that jobs are correctly scheduled on existing pools.
    Also verifies that the returned `Future`s are in the same order as the request.

    :param job_count: The number of jobs to schedule.
    :param use_execute_one: Whether to use `execute_one` to schedule jobs instead of `execute`.
    :param clear_parallel_executor: Ensures a clear state to test with.
    """
    expected_pool_count = (job_count + (WORKERS_PER_POOL - 1)) // WORKERS_PER_POOL
    schedule_call = schedule_with_execute_one if use_execute_one else ParallelExecutor.execute

    # Repeat the test a few times to make sure that the pool count doesn't go up after the first round
    for _ in range(3):
        barrier = Barrier(job_count + 1)  # All jobs will wait until triggered from the test

        # Create the jobs
        futures = schedule_call(*[lambda barrier=barrier: wait_for_signal(barrier, None) for _ in range(job_count)])
        assert len(futures) == job_count

        # Verify the number of pools
        assert ParallelExecutor._pool_count() == expected_pool_count  # noqa: SLF001  # Private access allowed from test

        # Trigger the barrier and wait for the jobs to finish
        finish_jobs(futures, barrier=barrier)


@pytest.mark.parametrize("args", ARGS_LISTS)
@pytest.mark.parametrize("kwargs", KWARGS_LISTS)
def test_parallel_executor_execute_one_passes_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """
    Verify that a parallel executor's `execute_one` correctly passes the arguments to the call.

    :param args: The positional arguments to pass.
    :param kwargs: The keyword arguments to pass.
    """
    future = ParallelExecutor.execute_one(return_args_and_kwargs, *args, **kwargs)
    assert future is not None

    result = future.result(timeout=2)
    assert result[0] == args
    assert result[1] == kwargs
