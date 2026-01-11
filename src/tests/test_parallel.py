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

# ruff: noqa: B904  # Lots of exception-translation (raise in except without from) for testing
# ruff: noqa: PT012  # Lots of pytest.raises have seemingly complex contents, which boil down to one statement
# ruff: noqa: S101  # assert is allowed in tests
# ruff: noqa: S311  # cryptographic random is not needed in tests
# ruff: noqa: SIM117  # Nested with-statements are used for readability
# ruff: noqa: TRY301  # lots of inline raises here, for testing

"""
Tests for `FutureCollection`, `FutureContextCollection` and their setup call `parallel()`.
"""

from __future__ import annotations

import sys
from concurrent.futures import wait
from contextlib import contextmanager
from enum import Enum
from random import Random
from threading import Barrier, BrokenBarrierError, Semaphore
from typing import TYPE_CHECKING, Any, Final, Generic, TypeVar, cast

import pytest

from parallel.parallel import FutureCollection, FutureContextCollection, parallel
from parallel.return_when import ReturnWhen

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from types import TracebackType

if sys.version_info <= (3, 10):  # noqa: UP036  # Backwards compatibility
    from typing_extensions import ParamSpec
else:  # pragma: no cover  # Backwards compatibility
    from typing import ParamSpec

if sys.version_info >= (3, 11):  # noqa: UP036  # Backwards compatibility
    from typing import assert_type
else:  # pragma: no cover  # Backwards compatibility

    def assert_type(value: Any, expected_type: Any) -> None:  # noqa: ANN401
        pass


T = TypeVar("T")
P = ParamSpec("P")

MULTI_JOB_COUNT = 8
"""Number of parallel jobs needed to test all execption handling."""

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


if sys.version_info < (3, 11):  # noqa: UP036  # Backwards compatibility is a feature

    class BaseExceptionGroup(BaseException):  # noqa: A001  # ruff fails to see this is pre-3.11 code
        """Naive and oversimplified implementation of BaseExceptionGroup for easier testing."""

        def __init__(self, msg: str, exceptions: Sequence[BaseException]) -> None:
            super().__init__(msg)
            self.exceptions = exceptions

    class ExceptionGroup(BaseExceptionGroup, Exception):  # noqa: A001, N818  # ruff fails to see this is pre-3.11 code
        """Naive and oversimplified implementation of ExceptionGroup for easier testing."""

        def __init__(self, msg: str, exceptions: Sequence[Exception]) -> None:
            super().__init__(msg, exceptions)


class Sentinel(Enum):
    """Sentinel values"""

    NO_VALUE = 1
    """Sentinel default value for optional parameters where `None` is a significant value."""


class Woopsie(Exception):  # noqa: N818  # Error suffix would be less readable for tests
    """An `Exception` to test with."""

    def __init__(self, identifier: int = 0) -> None:
        """
        Create a new `Woopsie`.

        :param id: Identifier of this exception.
        """
        super().__init__(f"Test exception Woopsie, id {identifier}")
        self.identifier = identifier

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Woopsie) and self.identifier == other.identifier

    def __ne__(self, other: object) -> bool:  # pragma: no cover  # Included for completeness' sake, only
        return not self.__eq__(other)

    def __hash__(self) -> int:  # pragma: no cover  # Included for completeness' sake, only
        return hash((self.__class__.__qualname__, self.identifier))


class BaseWoopsie(BaseException):
    """A `BaseException` to test with."""

    def __init__(self, identifier: int = 0) -> None:
        """
        Create a new `BaseWoopsie`.

        :param id: Identifier of this exception.
        """
        super().__init__(f"Test exception BaseWoopsie, id {identifier}")
        self.identifier = identifier

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BaseWoopsie) and self.identifier == other.identifier

    def __ne__(self, other: object) -> bool:  # pragma: no cover  # Included for completeness' sake, only
        return not self.__eq__(other)

    def __hash__(self) -> int:  # pragma: no cover  # Included for completeness' sake, only
        return hash((self.__class__.__qualname__, self.identifier))


def wait_for_signal(barrier: Barrier, result: T, raises: BaseException | None = None) -> T:
    """
    Simple function to be scheduled. This will wait for a signal before finishing.

    :param barrier: The `Barrier` to wait for.
    :param result: The value that will be returned when finishing.
    :param raises: If not None, the exception that will be raised instead of returning `result`.
    :returns: `result`
    """
    barrier.wait(timeout=2)
    if raises is not None:
        raise raises
    return result


def raise_exception(exc: Any) -> None:  # noqa: ANN401
    """
    Simple function to be scheduled. Will immediately raise an exception.

    :param exc: The exception to raise. Purposefully untyped.
    """
    raise exc


def return_arguments(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:  # noqa: ANN401
    """
    Return the arguments passed as a list and dictionary.
    """
    return (args, kwargs)


class SignalledContext(Generic[T]):
    """Context manager with customizable value and exceptions that can wait for a signal."""

    # Despite being a simplistic context manager, this must be a full class: @contextmanager has several corner cases
    # that would be hit by the tests (e.g. entering the context manager again after it has been exit-ed).

    def __init__(  # noqa: PLR0913  # Highly configurable utility
        self,
        *,
        yields: T,
        barrier_enter: Barrier | None = None,
        barrier_enter_timeout: float = 2.0,
        raises_enter: BaseException | None = None,
        barrier_exit: Barrier | None = None,
        raises_exit: BaseException | None = None,
        suppress_exception: bool | Woopsie = False,
    ) -> None:
        """
        Create a SignalledContext.

        :param yields: The value to yield, i.e. the result of `__enter__`.
        :param barrier_enter: The barrier to wait for at the beginning of `__enter__`.
        :param barrier_enter_timeout: The timeout on waiting for `barrier_enter`.
        :param raises_enter: The exception to raise during `__enter__`.
        :param barrier_exit: The barrier to wait for during `__exit__`.
        :param raises_exit: The exception to raise during `__exit__`.
        :param suppress_exception: Whether to suppress the exception during `__exit__`.
        """
        assert barrier_enter_timeout > 0
        self._yields = yields
        self._barrier_enter = barrier_enter
        self._barrier_enter_timeout = barrier_enter_timeout
        self._raises_enter = raises_enter
        self._barrier_exit = barrier_exit
        self._raises_exit = raises_exit
        self._suppress_exception = suppress_exception

    def __enter__(self) -> T:
        if self._barrier_enter:
            self._barrier_enter.wait(timeout=self._barrier_enter_timeout)
        if self._raises_enter:
            raise self._raises_enter
        return self._yields

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: TracebackType | None
    ) -> bool | None:
        if self._barrier_exit:
            self._barrier_exit.wait(timeout=2)
        if self._raises_exit:
            raise self._raises_exit
        if exc_type is not None and self._suppress_exception is True:
            return True
        return None


class ExpectExceptionInExit:
    """Context manager that expects a specific exception in `__exit__` and suppresses it."""

    def __init__(self, exception: Woopsie, *, suppress: bool = False) -> None:
        """
        Create a new `SuppressException`.

        :param exception: The exception to expect.
        :param suppress: If True, the exception will be suppressed.
        """
        self._exception = exception
        self._suppress = suppress

    def __enter__(self) -> None:
        pass

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: TracebackType | None
    ) -> bool | None:
        assert exc_value == self._exception
        return self._suppress


class RaiseExceptionInExit:
    """Context manager that raises an exception in its `__exit__`."""

    def __init__(self, exception: Woopsie) -> None:
        """
        Create a new `SuppressException`.

        :param exception: The exception to raise.
        """
        self._exception = exception

    def __enter__(self) -> None:
        pass

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: TracebackType | None
    ) -> bool | None:
        assert exc_type is None
        raise self._exception


@contextmanager
def will_raise(  # noqa: C901  # Somewhat complex, but that's a necessity
    exc: BaseException | type[BaseException],
    context: BaseException | type[BaseException] | Sentinel | None = Sentinel.NO_VALUE,
) -> Generator[None, None, None]:
    """
    Verify that an exception will be raised, possibly with the given context.

    This is basically a more specific `pytest.raises`.

    :param exc: The (type of) exception to should be raised. If a specific exception is given, a naive `!=` comparison
                is used to check equality. `ExceptionGroup` and `BaseExceptionGroup` are explicitly supported with their
                contained exceptions being compared (recursively, unordered).
    :param context: The (type of) exception that is expected to be the `__context__` of the raised exception. If a
                    specific exception is given, a naive `==` comparison is used to check equality. `ExceptionGroup` and
                    `BaseExceptionGroup` are explicitly supported with their contained exceptions being compared
                    (recursively, unordered).
    """

    def check_sub_exceptions(
        caught: BaseExceptionGroup | ExceptionGroup, expected: BaseExceptionGroup | ExceptionGroup
    ) -> bool:
        """
        Check that the exceptions in a BaseExceptionGroup or an ExceptionGroup are the same.

        Uses naive `==` comparison for exceptions, but recursively calls this function for `BaseExceptiongroup` and
        `ExceptionGroup`.

        :param caught: The caught (base) exception group.
        :param expected: The expected (base) exception group.
        :returns: Whether `caught` and `expected` contain the same exceptions.
        """
        if len(caught.exceptions) != len(expected.exceptions):
            return False
        expected_exceptions = list(expected.exceptions)
        caught_exceptions = list(caught.exceptions)
        for expected_exception in expected_exceptions:
            found = False
            for caught_exception in caught_exceptions:
                if exception_is_expected(expected_exception, caught_exception):
                    found = True
                    caught_exceptions.remove(cast(BaseException, caught_exception))
                    break
            if not found:
                return False
        return len(caught_exceptions) == 0

    def exception_is_expected(expected: BaseException | type[BaseException], caught: BaseException) -> bool:
        """
        Test whether a caught exception is as expected.

        :param expected: The expected (type of) exception.
        :param caught: The caught exception.
        :returns: Whether the caught exception matches the expected exception.
        """
        if isinstance(expected, type):
            return caught.__class__ == expected
        if isinstance(expected, BaseExceptionGroup) and isinstance(caught, BaseExceptionGroup):
            return check_sub_exceptions(cast(BaseExceptionGroup, expected), cast(BaseExceptionGroup, caught))
        return caught == expected

    try:
        yield
    except BaseException as caught:
        if not exception_is_expected(exc, caught):
            raise

        if context is not Sentinel.NO_VALUE:
            if context is None:
                assert caught.__context__ is None  # noqa: PT017  # Can't use pytest.raises here
            else:
                assert caught.__context__ is not None  # noqa: PT017  # Can't use pytest.raises here
                assert exception_is_expected(context, caught.__context__)  # noqa: PT017  # Can't use pytest.raises here


def test_will_raise() -> None:  # noqa: PLR0915  # Many statements, sure
    """
    Basic verification that `will_raise` works as expected.
    """
    # Correctly suppress expected type
    with will_raise(Woopsie):
        raise Woopsie(1)

    # Do not suppress exceptions of other types
    with pytest.raises(FileNotFoundError):
        with will_raise(Woopsie):
            raise FileNotFoundError

    # Strict check for expected type
    with pytest.raises(Woopsie):
        with will_raise(Exception):
            raise Woopsie(1)

    # Correctly suppress specific instance
    with will_raise(Woopsie(1)):
        raise Woopsie(1)

    # Do not suppress different instance
    with pytest.raises(Woopsie):
        with will_raise(Woopsie(1)):
            raise Woopsie(2)

    # Correctly suppress with explicitly no context
    with will_raise(Woopsie(1), context=None):
        raise Woopsie(1)

    # Assert if context is set when no context expected
    with pytest.raises(AssertionError):
        with will_raise(Woopsie(2), context=None):
            try:
                raise BaseWoopsie(1)
            finally:
                raise Woopsie(2)

    # Correctly suppress with correctly typed context
    with will_raise(Woopsie(2), context=BaseWoopsie):
        try:
            raise BaseWoopsie(1)
        finally:
            raise Woopsie(2)

    # Assert if type of context does not match expected type of context
    with pytest.raises(AssertionError):
        with will_raise(Woopsie(2), context=BaseWoopsie):
            try:
                raise Woopsie(1)
            finally:
                raise Woopsie(2)

    # Correctly suppress with specific context
    with will_raise(Woopsie(2), context=BaseWoopsie(1)):
        try:
            raise BaseWoopsie(1)
        finally:
            raise Woopsie(2)

    # Assert if type of context does not match specific context
    with pytest.raises(AssertionError):
        with will_raise(Woopsie(2), context=BaseWoopsie(1)):
            try:
                raise BaseWoopsie(2)
            finally:
                raise Woopsie(2)

    # Correctly suppresses specific ExceptionGroup
    with will_raise(ExceptionGroup("", [Woopsie(1)])):
        try:
            raise Woopsie(1)
        except Woopsie as exc:
            raise ExceptionGroup("", [exc])

    # Does not suppress incorrect specific ExceptionGroup
    with pytest.raises(ExceptionGroup):
        with will_raise(ExceptionGroup("", [Woopsie(1)])):
            try:
                raise Woopsie(2)
            except Woopsie as exc:
                raise ExceptionGroup("", [exc])

    with pytest.raises(ExceptionGroup):
        with will_raise(ExceptionGroup("", [Woopsie(2), Woopsie(2)])):
            try:
                raise Woopsie(2)
            except Woopsie as exc:
                raise ExceptionGroup("", [exc])

    # Correctly suppresses specific BaseExceptionGroup
    with will_raise(BaseExceptionGroup("", [BaseWoopsie(1)])):
        try:
            raise BaseWoopsie(1)
        except BaseWoopsie as exc:
            raise BaseExceptionGroup("", [exc])

    # Does not suppress incorrect specific BaseExceptionGroup
    with pytest.raises(BaseExceptionGroup):
        with will_raise(BaseExceptionGroup("", [BaseWoopsie(1)])):
            try:
                raise BaseWoopsie(2)
            except BaseWoopsie as exc:
                raise BaseExceptionGroup("", [exc])


def test_future_collection_results_immediately_raises_exception_on_first_exception() -> None:
    """
    Verify that FutureCollection's `results` raises the first exception to occur when called with
    `ReturnWhen.FIRST_EXCEPTION`.
    """
    barrier = Barrier(5)  # The additional jobs will wait for a trigger from the test
    collection = (
        FutureCollection
        .create(wait_for_signal)(barrier, None)
        .parallel(wait_for_signal)(barrier, None, Woopsie(1))
        .parallel(wait_for_signal)(barrier, None)
        .parallel(raise_exception)(Woopsie(2))  # This is the exception that will be caught
        .parallel(wait_for_signal)(barrier, None)
    )
    try:
        with will_raise(Woopsie(2)):
            collection.results(timeout=2, return_when=ReturnWhen.FIRST_EXCEPTION)
    finally:
        barrier.wait(timeout=2)  # Clean up waiting threads


def test_future_collection_results_raises_only_one_exception_on_first_exception() -> None:
    """
    Verify that FutureCollection's `results` raises only one exception when called with `ReturnWhen.FIRST_EXCEPTION`,
    even if multiple are already available when `results` is called.
    """
    collection = FutureCollection.create(raise_exception)(Woopsie(1)).parallel(raise_exception)(Woopsie(2))
    # Wait for all futures to be done (raised their exceptions) before asking the collection for its results
    _, not_done = wait(collection.futures, timeout=2)
    assert not not_done

    with will_raise(Woopsie, context=None):
        collection.results(timeout=2, return_when=ReturnWhen.FIRST_EXCEPTION)


@pytest.mark.parametrize("baseexception_job", [0, 1])
def test_future_collection_results_prefers_baseexception_on_first_exception(baseexception_job: int) -> None:
    """
    Verify that FutureCollection's `results` prefers to raise a BaseException when called with
    `ReturnWhen.FIRST_EXCEPTION`.

    :param baseexception_job: The index of the job that will raise the BaseException.
    """
    excs: list[Woopsie | BaseWoopsie] = [
        Woopsie(1),
        Woopsie(1),
    ]
    excs[baseexception_job] = BaseWoopsie(2)

    collection = FutureCollection.create(raise_exception)(excs[0]).parallel(raise_exception)(excs[1])
    # Wait for all futures to be done (raised their exceptions) before asking the collection for its results
    _, not_done = wait(collection.futures, timeout=2)
    assert not not_done

    with will_raise(BaseWoopsie(2), context=None):
        collection.results(timeout=2, return_when=ReturnWhen.FIRST_EXCEPTION)


def test_future_collection_results_does_not_support_first_completed() -> None:
    """
    Verify that FutureCollection's `results` does not complete `ReturnWhen.FIRST_COMPLETED`.
    """
    with pytest.raises(RuntimeError):
        FutureCollection.create(raise_exception)(Woopsie(1)).results(return_when=ReturnWhen.FIRST_COMPLETED)


def test_future_collection_results_can_timeout() -> None:
    """
    Verify that FutureCollection's `results` will raise a `TimeoutError` if at least one future takes too long.
    """
    barrier = Barrier(2)  # The job waits for a trigger from the test

    with pytest.raises(TimeoutError):
        FutureCollection.create(wait_for_signal)(barrier, None).results(timeout=0.25)

    barrier.wait(timeout=2)  # Cleanup


@pytest.mark.parametrize(
    "exception",
    [Woopsie(42), BaseWoopsie(42), ExceptionGroup("", [Woopsie(42)]), BaseExceptionGroup("", [BaseWoopsie(42)])],
)
@pytest.mark.parametrize("failing_job", range(MULTI_JOB_COUNT))
def test_future_collection_results_raises_the_one_exception(exception: Any, failing_job: int) -> None:  # noqa: ANN401
    """
    Verify that FutureCollection's `result` correctly raises the one `Exception` or `BaseException` that occurs when
    called with `ReturnWhen.ALL_COMPLETED`.

    :param exception: The one exception that will be raised.
    :param failing_job: The index of the job that will raise the exception.
    """
    barrier = Barrier(MULTI_JOB_COUNT)  # All jobs wait for each other

    args: list[tuple[Barrier, None, BaseException | None]] = [(barrier, None, None) for _ in range(MULTI_JOB_COUNT)]
    args[failing_job] = (barrier, None, exception)

    collection = (
        FutureCollection
        .create(wait_for_signal)(*args[0])
        .parallel(wait_for_signal)(*args[1])
        .parallel(wait_for_signal)(*args[2])
        .parallel(wait_for_signal)(*args[3])
        .parallel(wait_for_signal)(*args[4])
        .parallel(wait_for_signal)(*args[5])
        .parallel(wait_for_signal)(*args[6])
        .parallel(wait_for_signal)(*args[7])
    )

    with will_raise(exception, context=None):
        collection.results()


@pytest.mark.parametrize(
    "exception",
    [
        pytest.param(
            Woopsie, marks=pytest.mark.skipif(sys.version_info >= (3, 11), reason="Python version"), id="Python <3.11"
        ),
        pytest.param(
            ExceptionGroup("", [Woopsie(idx) for idx in range(MULTI_JOB_COUNT)]),
            marks=pytest.mark.skipif(sys.version_info < (3, 11), reason="Python version"),
            id="Python >=3.11",
        ),
    ],
)
def test_future_collection_results_raises_exception_on_multiple_exceptions(
    exception: type[Exception] | ExceptionGroup,
) -> None:
    """
    Verify that FutureCollection's `result` correctly raises when multiple exceptions occurred when called with
    `ReturnWhen.ALL_COMPLETE`.

    :param exception: The exception that is expected to be raised. This depends on the Python version.
    """
    barrier = Barrier(MULTI_JOB_COUNT)  # All jobs wait for each other

    args: list[tuple[Barrier, None, BaseException | None]] = [
        (barrier, None, Woopsie(idx)) for idx in range(MULTI_JOB_COUNT)
    ]

    collection = (
        FutureCollection
        .create(wait_for_signal)(*args[0])
        .parallel(wait_for_signal)(*args[1])
        .parallel(wait_for_signal)(*args[2])
        .parallel(wait_for_signal)(*args[3])
        .parallel(wait_for_signal)(*args[4])
        .parallel(wait_for_signal)(*args[5])
        .parallel(wait_for_signal)(*args[6])
        .parallel(wait_for_signal)(*args[7])
    )

    with will_raise(exception, context=None):
        collection.results()


@pytest.mark.parametrize(
    ("baseexception_job", "exception"),
    [
        *[
            pytest.param(
                baseexception_job,
                BaseWoopsie,
                marks=pytest.mark.skipif(sys.version_info >= (3, 11), reason="Python version"),
                id="Python <3.11",
            )
            for baseexception_job in range(MULTI_JOB_COUNT)
        ],
        *[
            pytest.param(
                baseexception_job,
                BaseExceptionGroup(
                    "",
                    [Woopsie(idx) if idx != baseexception_job else BaseWoopsie(1) for idx in range(MULTI_JOB_COUNT)],
                ),
                marks=pytest.mark.skipif(sys.version_info < (3, 11), reason="Python version"),
                id="Python >=3.11",
            )
            for baseexception_job in range(MULTI_JOB_COUNT)
        ],
    ],
)
def test_future_collection_results_prefers_baseexception_on_multiple_exceptions(
    baseexception_job: int, exception: type[BaseException] | ExceptionGroup
) -> None:
    """
    Verify that FutureCollection's `result` correctly prefers the `BaseException` over other exceptions when multiple
    exceptions occur when called with `ReturnWhen.ALL_COMPLETE`.

    :param baseexception_job: The index of the job that will raise the `BaseException`.
    :param exception: The exception that is expected to be raised. This depends on the Python version.
    """
    barrier = Barrier(MULTI_JOB_COUNT)  # All jobs wait for each other

    args: list[tuple[Barrier, None, BaseException | None]] = [
        (barrier, None, Woopsie(idx)) for idx in range(MULTI_JOB_COUNT)
    ]
    args[baseexception_job] = (barrier, None, BaseWoopsie(1))

    collection = (
        FutureCollection
        .create(wait_for_signal)(*args[0])
        .parallel(wait_for_signal)(*args[1])
        .parallel(wait_for_signal)(*args[2])
        .parallel(wait_for_signal)(*args[3])
        .parallel(wait_for_signal)(*args[4])
        .parallel(wait_for_signal)(*args[5])
        .parallel(wait_for_signal)(*args[6])
        .parallel(wait_for_signal)(*args[7])
    )

    with will_raise(exception, context=None):
        collection.results()


@pytest.mark.parametrize("seed", [Random().randbytes(8) for _ in range(16)])
def test_future_collection_results_in_order(seed: bytes) -> None:
    """
    Verify that FutureCollection's `results` returns the results in the same order as the calls were given.

    This test is randomized and repeated several times.

    :param seed: The seed for `Random` to use.
    """
    job_count: Final[int] = 8

    barriers = [Barrier(2) for _ in range(job_count)]  # Each job will wait for a trigger from the test

    # Note that the calls are explicitly filled in below. This is required for testing the typing.
    collection = (
        FutureCollection
        .create(wait_for_signal)(barriers[0], 1)
        .parallel(wait_for_signal)(barriers[1], 2)
        .parallel(wait_for_signal)(barriers[2], "two")
        .parallel(wait_for_signal)(barriers[3], 3)
        .parallel(wait_for_signal)(barriers[4], 4)
        .parallel(wait_for_signal)(barriers[5], cast(tuple[int, int, int], (1, 2, 3)))
        .parallel(wait_for_signal)(barriers[6], 6)
        .parallel(wait_for_signal)(barriers[7], 7)
    )

    # Trigger all jobs in a randomized order
    Random(seed).shuffle(barriers)
    for barrier in barriers:
        barrier.wait(timeout=2)

    results = collection.results()

    assert_type(results, tuple[int, int, str, int, int, tuple[int, int, int], int, int])
    assert len(results) == job_count
    assert results == (1, 2, "two", 3, 4, (1, 2, 3), 6, 7)


def test_future_collection_parallel_breaks_future_collection() -> None:
    """
    Verify that a FutureCollection is broken by calling the result of `parallel`.
    """
    collection = FutureCollection.create(lambda: None)()

    # Set up an incomplete call: this does *not* break the collection yet. (NOTE: Never do it like this!)
    incomplete_parallel = collection.parallel(lambda: 13)

    # Break the original collection, as well as the incomplete parallel call above.
    collection2 = collection.parallel(lambda: 1)()

    # Verify that everything of the original collection is broken
    with pytest.raises(RuntimeError):
        incomplete_parallel()
    with pytest.raises(RuntimeError):
        _ = collection.futures
    with pytest.raises(RuntimeError):
        collection.parallel(lambda: 2)
    with pytest.raises(RuntimeError):
        collection.results()

    # The superseeding collection still works correctly
    assert collection2.results() == (None, 1)


@pytest.mark.parametrize("args", ARGS_LISTS)
@pytest.mark.parametrize("kwargs", KWARGS_LISTS)
def test_future_collection_parallel_passes_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """
    Verify that FutureCollection's `parallel` correctly passes the arguments to the call.

    :param args: The positional arguments to pass.
    :param kwargs: The keyword arguments to pass.
    """
    results = FutureCollection.create(lambda: None)().parallel(return_arguments)(*args, **kwargs).results()

    assert results == (None, (args, kwargs))


@pytest.mark.parametrize("args", ARGS_LISTS)
@pytest.mark.parametrize("kwargs", KWARGS_LISTS)
def test_future_collection_create_passes_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """
    Verify that FutureCollection's `create`, as well as plain `parallel`, create a usable `FutureCollection` and
    correctly pass the arguments to the first call in a FutureCollection.

    :param args: The positional arguments to pass.
    :param kwargs: The keyword arguments to pass.
    """
    # Note that this test is NOT parametrized on the creation function. This is not doable in a Python-version
    # independent way without explicitly breaking and/or exempting all kinds of typing checks. The code is short, so
    # copy-paste is a nicer way to fix this.

    # First version: plain parallel
    collection = parallel(return_arguments)(*args, **kwargs)
    assert isinstance(collection, FutureCollection)

    results = collection.results()
    assert results == ((args, kwargs),)

    # Second version: FutureCollection.create
    collection = FutureCollection.create(return_arguments)(*args, **kwargs)
    assert isinstance(collection, FutureCollection)

    results = collection.results()
    assert results == ((args, kwargs),)


def test_future_context_collection_parallel_breaks_collection() -> None:
    """
    Verify that FutureContextCollection's `parallel` breaks the existing collection.
    """
    yields_one: Final[int] = 1
    yields_two: Final[int] = 2
    yields_three: Final[int] = 3

    collection = FutureContextCollection.create(SignalledContext(yields=yields_one))

    # Break the original collection by calling parallel
    collection2 = collection.parallel(SignalledContext(yields=yields_two))

    # Check that everything of the original collection is now broken
    with pytest.raises(RuntimeError):
        with collection:
            pytest.fail("Unreachable")
    with pytest.raises(RuntimeError):
        collection.parallel(SignalledContext(yields=yields_three))

    # New collection still works
    with collection2 as (one, two):
        assert one == yields_one
        assert two == yields_two


def test_future_context_collection_cannot_be_entered_recursively() -> None:
    """
    Verify that FutureContextCollection's context can't be entered recursively.
    """
    collection = FutureContextCollection.create(SignalledContext(yields=1))

    with collection:
        with pytest.raises(RuntimeError):
            with collection:
                pytest.fail("Unreachable")

    with collection:
        pass


@pytest.mark.parametrize("seed", [Random().randbytes(8) for _ in range(16)])
def test_future_context_collection_enter_is_parallel(seed: bytes) -> None:
    """
    Verify that FutureContextCollection's `__enter__` enters its contained context managers in parallel.

    This test is randomized and repeated several times.

    :param seed: The seed for `Random` to use.
    """
    barriers = [Barrier(2) for _ in range(MULTI_JOB_COUNT)]  # Each job waits for a signal from the test

    kwargs: list[dict[str, Any]] = [{"yields": idx, "barrier_enter": barriers[idx]} for idx in range(MULTI_JOB_COUNT)]

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]))
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    def signal_barriers() -> None:
        """Signal the barriers in a random order."""
        Random(seed).shuffle(barriers)
        for barrier in barriers:
            barrier.wait(timeout=2)

    # While trying to __enter__, in parallel signal each barrier in random order
    results, _ = FutureCollection.create(collection.__enter__)().parallel(signal_barriers)().results(timeout=2)
    assert results == tuple(range(MULTI_JOB_COUNT))


def test_future_context_collection_enter_can_timeout() -> None:
    """
    Verify that FutureContextCollection's `__enter__` can cause a timeout error.
    """
    barrier = Barrier(2)  # Job will wait for a trigger from the test
    collection = FutureContextCollection.create(SignalledContext(yields=None, barrier_enter=barrier), timeout=0.25)

    with pytest.raises(TimeoutError):
        with collection:
            pytest.fail("Unreachable")

    # Clean up the waiting thread
    barrier.wait(timeout=2)


@pytest.mark.parametrize(
    "exception",
    [Woopsie(42), BaseWoopsie(42), ExceptionGroup("", [Woopsie(42)]), BaseExceptionGroup("", [BaseWoopsie(42)])],
)
@pytest.mark.parametrize("failing_job", range(MULTI_JOB_COUNT))
def test_future_context_collection_enter_raises_the_one_exception(
    exception: Any,  # noqa: ANN401
    failing_job: int,
) -> None:
    """
    Verify that FutureContextCollection's `__enter__` correctly raises the one `Exception` or `BaseException` that
    occurs.

    :param exception: The one exception that will be raised.
    :param failing_job: The index of the job that will raise the exception.
    """
    barrier = Barrier(MULTI_JOB_COUNT)  # All jobs wait for each other

    kwargs: list[dict[str, Any]] = [{"yields": None, "barrier_enter": barrier} for _ in range(MULTI_JOB_COUNT)]
    kwargs[failing_job]["raises_enter"] = exception

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]))
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    with will_raise(exception, context=None):
        with collection:
            pytest.fail("Not reachable")


@pytest.mark.parametrize(
    "exception",
    [
        pytest.param(
            Woopsie, marks=pytest.mark.skipif(sys.version_info >= (3, 11), reason="Python version"), id="Python <3.11"
        ),
        pytest.param(
            ExceptionGroup("", [Woopsie(idx) for idx in range(MULTI_JOB_COUNT)]),
            marks=pytest.mark.skipif(sys.version_info < (3, 11), reason="Python version"),
            id="Python >=3.11",
        ),
    ],
)
def test_future_context_collection_enter_raises_exception_on_multiple_exceptions(
    exception: type[Exception] | ExceptionGroup,
) -> None:
    """
    Verify that FutureContextCollection's `__enter__` correctly raises when multiple exceptions occur.

    :param exception: The exception that is expected to be raised. This depends on the Python version.
    """
    barrier = Barrier(MULTI_JOB_COUNT)  # All jobs wait for each other

    kwargs: list[dict[str, Any]] = [
        {"yields": None, "barrier_enter": barrier, "raises_enter": Woopsie(idx)} for idx in range(MULTI_JOB_COUNT)
    ]

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]))
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    with will_raise(exception, context=None):
        with collection:
            pytest.fail("Not reachable")


@pytest.mark.parametrize(
    ("baseexception_job", "exception"),
    [
        *[
            pytest.param(
                baseexception_job,
                BaseWoopsie,
                marks=pytest.mark.skipif(sys.version_info >= (3, 11), reason="Python version"),
                id="Python <3.11",
            )
            for baseexception_job in range(MULTI_JOB_COUNT)
        ],
        *[
            pytest.param(
                baseexception_job,
                BaseExceptionGroup(
                    "",
                    [Woopsie(idx) if idx != baseexception_job else BaseWoopsie(1) for idx in range(MULTI_JOB_COUNT)],
                ),
                marks=pytest.mark.skipif(sys.version_info < (3, 11), reason="Python version"),
                id="Python >=3.11",
            )
            for baseexception_job in range(MULTI_JOB_COUNT)
        ],
    ],
)
def test_future_context_collection_enter_prefers_baseexception_on_multiple_exceptions(
    baseexception_job: int, exception: type[BaseException] | ExceptionGroup
) -> None:
    """
    Verify that FutureContextCollection's `__enter__` correctly prefers the `BaseException` over other exceptions when
    multiple exceptions occur.

    :param baseexception_job: The index of the job that will raise the `BaseException`.
    :param exception: The exception that is expected to be raised. This depends on the Python version.
    """
    barrier = Barrier(MULTI_JOB_COUNT)  # All jobs wait for each other

    kwargs: list[dict[str, Any]] = [
        {"yields": None, "barrier_enter": barrier, "raises_enter": Woopsie(idx)} for idx in range(MULTI_JOB_COUNT)
    ]
    kwargs[baseexception_job]["raises_enter"] = BaseWoopsie(1)

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]))
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    with will_raise(exception, context=None):
        with collection:
            pytest.fail("Not reachable")


def test_future_context_collection_enter_exits_entered_contexts_in_reverse_order_on_exception() -> None:
    """
    Verify that FutureContextCollection's `__enter__` will exit the already entered contained context managers in
    reverse order if an exception occurs while entering one of the contained context managers.
    """
    exception_job: Final[int] = 5
    normal_jobs: Final[list[int]] = [idx for idx in range(MULTI_JOB_COUNT) if idx != exception_job]

    exit_barriers = [Barrier(2) for _ in range(MULTI_JOB_COUNT)]  # Each job will wait for a trigger from the test

    kwargs: list[dict[str, Any]] = [
        {"yields": None, "barrier_exit": exit_barriers[idx]} for idx in range(MULTI_JOB_COUNT)
    ]
    kwargs[exception_job] = {"yields": None, "raises_enter": Woopsie(1), "barrier_exit": exit_barriers[exception_job]}

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]), timeout=2)
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    def trigger_jobs() -> None:
        """Trigger the barriers for all the jobs."""
        # Entering the collection will wait until all jobs have finished their __enter__, either with success or an
        # exception.

        # Trigger each exit barrier in the expected order
        normal_jobs.reverse()
        for idx in normal_jobs:
            exit_barriers[idx].wait(timeout=2)

    def enter_collection() -> None:
        """Enter the collection."""
        with pytest.raises(Woopsie):
            with collection:
                pytest.fail("Unreachable")

    parallel(enter_collection)().parallel(trigger_jobs)().results(timeout=2)


def test_future_context_collection_enter_does_not_exit_timed_out_contexts_on_exception() -> None:
    """
    Verify that FutureContextCollection's `__enter__` will only exit the already entered contained context managers if
    an exception occurs while entering one of the contained context managers.

    Also verifies that a timeout in `__enter__` correctly raises the exception that occurred in one of the already
    entered contained context managers, with the TimeoutError as its context.
    """
    exception_job: Final[int] = 5
    slow_job: Final[int] = 7
    normal_jobs: Final[list[int]] = [idx for idx in range(MULTI_JOB_COUNT) if idx not in {exception_job, slow_job}]

    slow_enter_barrier = Barrier(2)  # The slow job will wait in __enter__ for a trigger from the test
    exit_barriers = [Barrier(2) for _ in range(MULTI_JOB_COUNT)]  # Each job will wait for a trigger from the test

    kwargs: list[dict[str, Any]] = [
        {"yields": None, "barrier_exit": exit_barriers[idx]} for idx in range(MULTI_JOB_COUNT)
    ]
    kwargs[exception_job] = {"yields": None, "raises_enter": Woopsie(1), "barrier_exit": exit_barriers[exception_job]}
    kwargs[slow_job] = {
        "yields": None,
        "barrier_enter": slow_enter_barrier,
        "barrier_enter_timeout": 5,
        "barrier_exit": exit_barriers[slow_job],
    }

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]), timeout=0.25)
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    def trigger_jobs() -> None:
        """Trigger the barriers for all the jobs."""
        # Entering the collection will wait until all jobs have finished their __enter__, either with success or an
        # exception.

        # At this point the slow job is still waiting. A timeout will occur.

        # Trigger each exit barrier in the expected order
        normal_jobs.reverse()
        for idx in normal_jobs:
            exit_barriers[idx].wait(timeout=2)

    def enter_collection() -> None:
        """Enter the collection."""
        # The actual exception will be the Woopsie, but it should list the TimeoutError (due to the slow job) as its
        # context
        with will_raise(Woopsie, context=TimeoutError):
            with collection:
                pytest.fail("Unreachable")

    parallel(enter_collection)().parallel(trigger_jobs)().results(timeout=5)

    # Only now trigger the slow job's __enter__, triggering its __exit__ should fail
    slow_enter_barrier.wait(0.25)
    with pytest.raises(BrokenBarrierError):
        exit_barriers[slow_job].wait(0.25)


def test_future_context_collection_enter_does_not_exit_timed_out_context_managers() -> None:
    """
    Verify that FutureContextCollection's `__enter__` will only exit the contained context managers that were already
    entered before the timeout occurred.
    """
    slow_job: Final[int] = 7
    normal_jobs: Final[list[int]] = [idx for idx in range(MULTI_JOB_COUNT) if idx != slow_job]

    slow_enter_barrier = Barrier(2)  # The slow job will wait in __enter__ for a trigger from the test
    exit_barriers = [Barrier(2) for _ in range(MULTI_JOB_COUNT)]  # Each job will wait for a trigger from the test

    kwargs: list[dict[str, Any]] = [
        {"yields": None, "barrier_exit": exit_barriers[idx]} for idx in range(MULTI_JOB_COUNT)
    ]
    kwargs[slow_job] = {"yields": None, "barrier_enter": slow_enter_barrier, "barrier_enter_timeout": 5}

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]), timeout=0.25)
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    def trigger_jobs() -> None:
        """Trigger the barriers for all the jobs."""
        # Entering the collection will wait until all jobs have finished their __enter__, either with success or an
        # exception.

        # At this point the slow job is still waiting. A timeout will occur.

        # Trigger each exit barrier in the expected order
        normal_jobs.reverse()
        for idx in normal_jobs:
            exit_barriers[idx].wait(timeout=2)

    def enter_collection() -> None:
        """Enter the collection."""
        with pytest.raises(TimeoutError):
            with collection:
                pytest.fail("Unreachable")

    parallel(enter_collection)().parallel(trigger_jobs)().results()

    # Only now trigger the slow job's __enter__, triggering its __exit__ should fail
    slow_enter_barrier.wait(0.25)
    with pytest.raises(BrokenBarrierError):
        exit_barriers[slow_job].wait(0.25)


@pytest.mark.parametrize(
    ("enter_exception", "exit_exception", "expected_exception", "expected_context"),
    [
        (Woopsie(1), Woopsie(2), Woopsie(2), Woopsie(1)),
        (Woopsie(1), None, Woopsie(1), None),
        (None, Woopsie(2), Woopsie(2), None),
    ],
)
def test_future_context_collection_enter_does_not_allow_suppressing_exception(
    enter_exception: Woopsie | None,
    exit_exception: Woopsie | None,
    expected_exception: Woopsie,
    expected_context: Woopsie | None,
) -> None:
    """
    Verify that it is not possible to suppress the exception that occurred in FutureContextCollection's `__enter__`
    when entering one of the contained context managers or exiting it again.

    Also verifies that when an exception occurs in FutureContextCollection's `__enter__` when entering one of the
    contained context managers, and then an exception occurs in one of the already entered contained context manager's
    `__exit__`, that those exceptions are chained correctly.

    :param enter_exception: The exception to raise in a job during `__enter__`.
    :param exit_exception: The exception to raise in a job during `__exit__`.
    :param expected_exception: The exception that is expected to be raised.
    :param expected_context: The context that the raised exception is expected to have.
    """
    collection = (
        FutureContextCollection
        .create(SignalledContext(yields=None, suppress_exception=True))
        .parallel(SignalledContext(yields=None, raises_enter=enter_exception, suppress_exception=True))
        .parallel(SignalledContext(yields=None, suppress_exception=True))
        .parallel(SignalledContext(yields=None, raises_exit=exit_exception, suppress_exception=True))
        .parallel(SignalledContext(yields=None, suppress_exception=True))
    )

    with will_raise(expected_exception, context=expected_context):
        with collection:
            pytest.fail("Unreachable")


@pytest.mark.parametrize("seed", [Random().randbytes(8) for _ in range(16)])
def test_future_context_collection_enter_returns_results_in_order(seed: bytes) -> None:
    """
    Verify that FutureContextCollection's `__enter__` returns results for the contained  context managers in the order
    they were added, and works with mixed type results.

    :param seed: The seed for `Random` to use.
    """
    job_count: Final[int] = 4
    barriers = [Barrier(2) for _ in range(job_count)]

    # Note that the calls are explicitly filled in below. This is required for testing the typing.
    collection = (
        FutureContextCollection
        .create(SignalledContext(yields=1, barrier_enter=barriers[0]))
        .parallel(SignalledContext(yields="two", barrier_enter=barriers[1]))
        .parallel(SignalledContext(yields=3, barrier_enter=barriers[2]))
        .parallel(SignalledContext(yields=cast(tuple[int, int, int], (1, 2, 3)), barrier_enter=barriers[3]))
    )

    def trigger_jobs() -> None:
        """Trigger all jobs in a randomized order."""
        Random(seed).shuffle(barriers)
        for barrier in barriers:
            barrier.wait(timeout=2)

    def enter_collection() -> None:
        """Enter the collection and verify the results."""
        with collection as results:
            assert_type(results, tuple[int, str, int, tuple[int, int, int]])
            assert results == (1, "two", 3, (1, 2, 3))

    parallel(enter_collection)().parallel(trigger_jobs)().results(timeout=2)


def test_future_context_collection_exit_is_idempotent_when_not_entered() -> None:
    """
    Verify that FutureContextCollection's `__exit__` can be called even if the context has not been entered yet.
    """
    barrier = Barrier(2)  # The exit of the job is triggered from the test

    collection = FutureContextCollection.create(SignalledContext(yields=None, barrier_exit=barrier))

    # Never entered
    collection.__exit__(None, None, None)

    # Enter and exit correctly
    collection.__enter__()  # noqa: PLC2801  # Calling the dunder directly is better for the test
    parallel(collection.__exit__)(None, None, None).parallel(barrier.wait)(2).results(timeout=2)

    # After having been entered and exitted
    collection.__exit__(None, None, None)


def test_future_context_collection_exit_is_idempotent_when_enter_fails() -> None:
    """
    Verify that FutureContextCollection's `__exit__` can be called even when entering the context failed.
    """
    barrier = Barrier(2)  # Should never be waited upon; the test won't trigger it

    collection = FutureContextCollection.create(
        SignalledContext(yields=None, raises_enter=Woopsie(1), barrier_exit=barrier)
    )

    with pytest.raises(Woopsie):
        collection.__enter__()  # noqa: PLC2801  # Calling the dunder directly, here, is better for the test

    collection.__exit__(None, None, None)


def test_future_context_collection_exit_is_idempotent_when_exit_fails() -> None:
    """
    Verify that FutureContextCollection's `__exit__` can be called even when a previous call to `__exit__` failed.
    """
    collection = FutureContextCollection.create(SignalledContext(yields=None, raises_exit=Woopsie(1)))

    collection.__enter__()  # noqa: PLC2801  # Calling the dunder directly, here, is better for the test
    with pytest.raises(Woopsie):
        collection.__exit__(None, None, None)

    collection.__exit__(None, None, None)

    collection.__enter__()  # noqa: PLC2801  # Calling the dunder directly, here, is better for the test
    with pytest.raises(Woopsie):
        collection.__exit__(None, None, None)

    collection.__enter__()  # noqa: PLC2801  # Calling the dunder directly, here, is better for the test
    with pytest.raises(Woopsie):
        collection.__exit__(None, None, None)

    collection.__exit__(None, None, None)


def test_future_context_collection_exit_sequential_exits_in_reverse_order() -> None:
    """
    Verify that FutureContextCollection's `__exit__` leaves the contained context managers in reverse order when not
    using parallel exit.
    """
    exit_barriers = [Barrier(2) for _ in range(MULTI_JOB_COUNT)]  # Each job will wait for a trigger from the test

    kwargs: list[dict[str, Any]] = [
        {"yields": None, "barrier_exit": exit_barriers[idx]} for idx in range(MULTI_JOB_COUNT)
    ]

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]), timeout=2)
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    collection.__enter__()  # noqa: PLC2801  # Calling the dunder directly is better for the test

    def trigger_jobs() -> None:
        """Trigger the exit barriers for all the jobs."""
        # Trigger each exit barrier in the expected order
        exit_barriers.reverse()
        for barrier in exit_barriers:
            barrier.wait(timeout=2)

    parallel(collection.__exit__)(None, None, None).parallel(trigger_jobs)().results(timeout=2)


def test_future_context_collection_exit_sequential_exceptions_bubble_into_exit() -> None:
    """
    Verify that FutureContextCollection's `__exit__` passes the outer exception into the contained context managers'
    `__exit__`, and correctly passes exceptions between those as well, when not using parallel exit.

    Also verify that FutureContextCollection's `__exit__` allows suppressing the outer exception in the contained
    context managers' `__exit__`, as well as exceptions that are passed between those.
    """
    collection = (
        FutureContextCollection
        .create(ExpectExceptionInExit(Woopsie(2), suppress=True))
        .parallel(RaiseExceptionInExit(Woopsie(2)))
        .parallel(ExpectExceptionInExit(Woopsie(1), suppress=True))
        .parallel(ExpectExceptionInExit(Woopsie(1)))
    )

    with collection:
        raise Woopsie(1)


def test_future_context_collection_exit_handles_all_exit_signatures() -> None:  # noqa: C901  # Not actually complex
    """
    Verify that FutureContextCollection's `__exit__` correctly handles all valid signatures for `__exit__` on contained
    context managers.

    Also verifies that FutureContextCollection's `create` and `parallel`, as well as plain `parallel`, correctly
    recognize and handle context managers with those signatures.
    """
    semaphore = Semaphore(0)  # Counter for tracking how many __exit__s have been called

    class NoneExit:
        """Context manager for which `__exit__` returns `None`."""

        def __enter__(self) -> None:
            pass

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            exc_traceback: TracebackType | None,
        ) -> None:
            semaphore.release()

    class BoolExit:
        """Context manager for which `__exit__` returns `bool`."""

        def __init__(self, *, returns: bool) -> None:
            """
            Initialize a `BoolExit` with a specific return value for `__exit__`.

            :param returns: The value `__exit__` will return.
            """
            self._returns = returns

        def __enter__(self) -> None:
            pass

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            exc_traceback: TracebackType | None,
        ) -> bool:
            semaphore.release()
            return self._returns

    class FullExit:
        """Context manager for which `__exit__` returns `bool | None`."""

        def __init__(self, *, returns: bool | None) -> None:
            """
            Initialize a `FullExit` with a specific return value for `__exit__`.

            :param returns: The value `__exit__` will return.
            """
            self._returns = returns

        def __enter__(self) -> None:
            pass

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            exc_traceback: TracebackType | None,
        ) -> bool | None:
            semaphore.release()
            return self._returns

    class ArgsExit:
        """Context manager with `*args` as parameters for `__exit__`."""

        def __enter__(self) -> None:
            pass

        def __exit__(self, *args: Any) -> None:  # noqa: ANN401  # Specifically testing for this signature
            semaphore.release()

    # The collections are made in many different ways. This is also a test for the typing information.
    collections: list[Any] = [
        FutureContextCollection.create(NoneExit()),
        FutureContextCollection.create(BoolExit(returns=False)),
        FutureContextCollection.create(FullExit(returns=None)),
        FutureContextCollection.create(FullExit(returns=False)),
        FutureContextCollection.create(ArgsExit()),
        parallel(NoneExit()),
        parallel(BoolExit(returns=False)),
        parallel(FullExit(returns=None)),
        parallel(FullExit(returns=False)),
        parallel(ArgsExit()),
        FutureContextCollection.create(NoneExit(), parallel_exit=True),
        FutureContextCollection.create(BoolExit(returns=False), parallel_exit=True),
        FutureContextCollection.create(FullExit(returns=None), parallel_exit=True),
        FutureContextCollection.create(FullExit(returns=False), parallel_exit=True),
        FutureContextCollection.create(ArgsExit(), parallel_exit=True),
        parallel(NoneExit(), parallel_exit=True),
        parallel(BoolExit(returns=False), parallel_exit=True),
        parallel(FullExit(returns=None), parallel_exit=True),
        parallel(FullExit(returns=False), parallel_exit=True),
        parallel(ArgsExit(), parallel_exit=True),
        FutureContextCollection.create(NoneExit(), parallel_exit=False),
        FutureContextCollection.create(BoolExit(returns=False), parallel_exit=False),
        FutureContextCollection.create(FullExit(returns=None), parallel_exit=False),
        FutureContextCollection.create(FullExit(returns=False), parallel_exit=False),
        FutureContextCollection.create(ArgsExit(), parallel_exit=False),
        parallel(NoneExit(), parallel_exit=False),
        parallel(BoolExit(returns=False), parallel_exit=False),
        parallel(FullExit(returns=None), parallel_exit=False),
        parallel(FullExit(returns=False), parallel_exit=False),
        parallel(ArgsExit(), parallel_exit=False),
    ]

    # Verify that each exit is correctly called and correctly bubbles the exception
    for collection in collections:
        with pytest.raises(Woopsie):
            with collection:
                raise Woopsie(1)
        semaphore.acquire(timeout=0.25)

    # The collections are made in many different ways. This is also a test for the typing information.
    collections = [
        FutureContextCollection.create(BoolExit(returns=True)),
        FutureContextCollection.create(FullExit(returns=True)),
        parallel(BoolExit(returns=True)),
        parallel(FullExit(returns=True)),
        FutureContextCollection.create(BoolExit(returns=True), parallel_exit=True),
        FutureContextCollection.create(FullExit(returns=True), parallel_exit=True),
        parallel(BoolExit(returns=True), parallel_exit=True),
        parallel(FullExit(returns=True), parallel_exit=True),
        FutureContextCollection.create(BoolExit(returns=True), parallel_exit=False),
        FutureContextCollection.create(FullExit(returns=True), parallel_exit=False),
        parallel(BoolExit(returns=True), parallel_exit=False),
        parallel(FullExit(returns=True), parallel_exit=False),
    ]

    # Verify that each exit is correctly called and correctly suppresses the exception
    for collection in collections:
        with collection:
            raise Woopsie(1)
        semaphore.acquire(timeout=0.25)

    # The collections are made in many different ways. This is also a test for the typing information.
    complex_collections = [
        parallel(NoneExit()).parallel(NoneExit()),
        parallel(NoneExit()).parallel(BoolExit(returns=False)),
        parallel(NoneExit()).parallel(FullExit(returns=False)),
        parallel(NoneExit()).parallel(ArgsExit()),
        parallel(NoneExit(), parallel_exit=True).parallel(NoneExit()),
        parallel(NoneExit(), parallel_exit=True).parallel(BoolExit(returns=False)),
        parallel(NoneExit(), parallel_exit=True).parallel(FullExit(returns=False)),
        parallel(NoneExit(), parallel_exit=True).parallel(ArgsExit()),
        parallel(NoneExit(), parallel_exit=False).parallel(NoneExit()),
        parallel(NoneExit(), parallel_exit=False).parallel(BoolExit(returns=False)),
        parallel(NoneExit(), parallel_exit=False).parallel(FullExit(returns=False)),
        parallel(NoneExit(), parallel_exit=False).parallel(ArgsExit()),
    ]

    # Verify that each exit is correctly called and correctly bubbles the exception
    for collection in complex_collections:
        with pytest.raises(Woopsie):
            with collection:
                raise Woopsie(1)
        semaphore.acquire(timeout=0.25)

    # The collections are made in many different ways. This is also a test for the typing information.
    complex_collections = [
        parallel(NoneExit()).parallel(BoolExit(returns=True)),
        parallel(NoneExit()).parallel(FullExit(returns=True)),
        parallel(NoneExit(), parallel_exit=True).parallel(BoolExit(returns=True)),
        parallel(NoneExit(), parallel_exit=True).parallel(FullExit(returns=True)),
        parallel(NoneExit(), parallel_exit=False).parallel(BoolExit(returns=True)),
        parallel(NoneExit(), parallel_exit=False).parallel(FullExit(returns=True)),
    ]

    # Verify that each exit is correctly called and correctly suppresses the exception
    for collection in collections:
        with collection:
            raise Woopsie(1)
        semaphore.acquire(timeout=0.25)


def test_future_context_collection_exit_parallel_exits_in_parallel() -> None:
    """
    Verify that FutureContextCollection's `__exit__` in parallel mode exits the contained context managers in parallel.
    """
    barrier = Barrier(MULTI_JOB_COUNT)  # The jobs wait for each other

    kwargs: list[dict[str, Any]] = [{"yields": None, "barrier_exit": barrier} for _ in range(MULTI_JOB_COUNT)]

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]), timeout=2, parallel_exit=True)
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    with collection:
        pass


def test_future_context_collection_exit_parallel_only_passes_outer_exception() -> None:
    """
    Verify that FutureContextCollection's `__exit__` in parallel mode passes the outer exception to the contained
    context managers' `__exit__`, but does not pass exceptions between those.
    """
    collection = (
        FutureContextCollection
        .create(ExpectExceptionInExit(Woopsie(1), suppress=True), parallel_exit=True)
        .create(SignalledContext(yields=None, raises_exit=Woopsie(2)))
        .create(ExpectExceptionInExit(Woopsie(1), suppress=True))
    )

    with will_raise(Woopsie(2), context=None):
        with collection:
            raise Woopsie(1)


@pytest.mark.parametrize(
    "suppress_jobs",
    [
        *[[idx] for idx in range(MULTI_JOB_COUNT)],
        *[[idx1, idx2] for idx1 in range(MULTI_JOB_COUNT) for idx2 in range(MULTI_JOB_COUNT) if idx1 != idx2],
    ],
)
def test_future_context_collection_exit_parallel_allows_any_exit_to_suppress_outer_exception(
    suppress_jobs: list[int],
) -> None:
    """
    Verify that FutureContextCollection's `__exit__` in parallel mode allows suppressing the outer exception by any of
    the context managers' `__exit__`.

    :param suppress_jobs: Indices of the jobs that will attempt to suppress the exception.
    """
    context_managers = [
        ExpectExceptionInExit(Woopsie(1), suppress=idx in suppress_jobs) for idx in range(MULTI_JOB_COUNT)
    ]

    collection = (
        FutureContextCollection
        .create(context_managers[0], parallel_exit=True)
        .parallel(context_managers[1])
        .parallel(context_managers[2])
        .parallel(context_managers[3])
        .parallel(context_managers[4])
        .parallel(context_managers[5])
        .parallel(context_managers[6])
        .parallel(context_managers[7])
    )

    with collection:
        raise Woopsie(1)


@pytest.mark.parametrize(
    "exception",
    [Woopsie(42), BaseWoopsie(42), ExceptionGroup("", [Woopsie(42)]), BaseExceptionGroup("", [BaseWoopsie(42)])],
)
@pytest.mark.parametrize("failing_job", range(MULTI_JOB_COUNT))
def test_future_context_collection_exit_parallel_raises_the_one_exception(
    exception: Any,  # noqa: ANN401
    failing_job: int,
) -> None:
    """
    Verify that FutureContextCollection's `__exit__` correctly raises the one `Exception` or `BaseException` that
    occurs.

    :param exception: The one exception that will be raised.
    :param failing_job: The index of the job that will raise the exception.
    """
    barrier = Barrier(MULTI_JOB_COUNT)  # All jobs wait for each other

    kwargs: list[dict[str, Any]] = [{"yields": None, "barrier_exit": barrier} for _ in range(MULTI_JOB_COUNT)]
    kwargs[failing_job] = {"yields": None, "barrier_exit": barrier, "raises_exit": exception}

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]), parallel_exit=True)
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    enter_succeeded = False
    with will_raise(exception, context=None):
        with collection:
            enter_succeeded = True
    assert enter_succeeded


@pytest.mark.parametrize(
    "exception",
    [
        pytest.param(
            Woopsie, marks=pytest.mark.skipif(sys.version_info >= (3, 11), reason="Python version"), id="Python <3.11"
        ),
        pytest.param(
            ExceptionGroup("", [Woopsie(idx) for idx in range(MULTI_JOB_COUNT)]),
            marks=pytest.mark.skipif(sys.version_info < (3, 11), reason="Python version"),
            id="Python >=3.11",
        ),
    ],
)
def test_future_context_collection_exit_parallel_raises_exception_on_multiple_exceptions(
    exception: type[Exception] | ExceptionGroup,
) -> None:
    """
    Verify that FutureContextCollection's `__enter__` correctly raises when multiple exceptions occur.

    :param exception: The exception that is expected to be raised. This depends on the Python version.
    """
    barrier = Barrier(MULTI_JOB_COUNT)  # All jobs wait for each other

    kwargs: list[dict[str, Any]] = [
        {"yields": None, "barrier_exit": barrier, "raises_exit": Woopsie(idx)} for idx in range(MULTI_JOB_COUNT)
    ]

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]), parallel_exit=True)
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    enter_succeeded = False
    with will_raise(exception, context=None):
        with collection:
            enter_succeeded = True
    assert enter_succeeded


@pytest.mark.parametrize(
    ("baseexception_job", "exception"),
    [
        *[
            pytest.param(
                baseexception_job,
                BaseWoopsie,
                marks=pytest.mark.skipif(sys.version_info >= (3, 11), reason="Python version"),
                id="Python <3.11",
            )
            for baseexception_job in range(MULTI_JOB_COUNT)
        ],
        *[
            pytest.param(
                baseexception_job,
                BaseExceptionGroup(
                    "",
                    [Woopsie(idx) if idx != baseexception_job else BaseWoopsie(1) for idx in range(MULTI_JOB_COUNT)],
                ),
                marks=pytest.mark.skipif(sys.version_info < (3, 11), reason="Python version"),
                id="Python >=3.11",
            )
            for baseexception_job in range(MULTI_JOB_COUNT)
        ],
    ],
)
def test_future_context_collection_exit_parallel_prefers_baseexception_on_multiple_exceptions(
    baseexception_job: int, exception: type[BaseException] | ExceptionGroup
) -> None:
    """
    Verify that FutureContextCollection's `__enter__` correctly prefers the `BaseException` over other exceptions when
    multiple exceptions occur.

    :param baseexception_job: The index of the job that will raise the `BaseException`.
    :param exception: The exception that is expected to be raised. This depends on the Python version.
    """
    barrier = Barrier(MULTI_JOB_COUNT)  # All jobs wait for each other

    kwargs: list[dict[str, Any]] = [
        {"yields": None, "barrier_exit": barrier, "raises_exit": Woopsie(idx)} for idx in range(MULTI_JOB_COUNT)
    ]
    kwargs[baseexception_job] = {"yields": None, "barrier_exit": barrier, "raises_exit": BaseWoopsie(1)}

    collection = (
        FutureContextCollection
        .create(SignalledContext(**kwargs[0]), parallel_exit=True)
        .parallel(SignalledContext(**kwargs[1]))
        .parallel(SignalledContext(**kwargs[2]))
        .parallel(SignalledContext(**kwargs[3]))
        .parallel(SignalledContext(**kwargs[4]))
        .parallel(SignalledContext(**kwargs[5]))
        .parallel(SignalledContext(**kwargs[6]))
        .parallel(SignalledContext(**kwargs[7]))
    )

    enter_succeeded = False
    with will_raise(exception, context=None):
        with collection:
            enter_succeeded = True
    assert enter_succeeded


def test_future_context_collection_no_parallel_while_entered() -> None:
    """
    Verify that chaining a `parallel` call on a `FutureContextCollection` that has already been entered is not possible.
    """
    collection = parallel(SignalledContext(yields=1)).parallel(SignalledContext(yields=2))
    with collection:
        with pytest.raises(RuntimeError):
            collection.parallel(SignalledContext(yields=3))

    # After the context manager has been left again, .parallel is allowed again
    new_collection = collection.parallel(SignalledContext(yields=3))
    with new_collection:
        pass


def test_future_context_collection_create_passes_on_arguments() -> None:
    """
    Verify that plain `parallel` create a new `FutureContextCollection` with the arguments passed on.
    """
    yields_one: Final[int] = 1
    yields_two: Final[int] = 2

    # First test that timeout is correctly passed on by parallel()
    barrier = Barrier(2)
    collection_with_timeout = parallel(SignalledContext(yields=yields_one, barrier_enter=barrier), timeout=0.2)

    with pytest.raises(TimeoutError):
        with collection_with_timeout:
            pytest.fail("Unreachable")

    barrier.wait(timeout=0.25)

    # Secondly test that the parallel exit parameter is correctly passed on by parallel(), as well as the actual context
    # managers
    barrier = Barrier(2)
    collection_with_parallel_exit = parallel(
        SignalledContext(yields=yields_one, barrier_exit=barrier), parallel_exit=True
    ).parallel(SignalledContext(yields=yields_two, barrier_exit=barrier))

    with collection_with_parallel_exit as (one, two):
        assert one == yields_one
        assert two == yields_two


def test_parallel_with_context_manager_enter_creates_future_collection() -> None:
    """
    Verify that plain `parallel` will create a `FutureCollection` when directly given an `AbstractContextManager`'s
    `__enter__`.
    """
    context_manager = SignalledContext(yields=None)

    assert parallel(context_manager.__enter__)().results() == (None,)


def test_parallel_with_context_manager_creates_future_context_collection() -> None:
    """
    Verify that plain `parallel` will create a `FutureContextCollection` when given an `AbstractContextManager`, even if
    that context manager implements `__call__`.
    """
    barrier = Barrier(2)

    class ContextManagerWithCall:
        def __call__(self) -> None:
            pytest.fail("Should never be called")

        def __enter__(self) -> None:
            barrier.wait(timeout=2)

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            exc_traceback: TracebackType | None,
        ) -> bool | None:  # pragma: no cover  # __exit__ is only implemented to make this a context manager
            return None

    collection = parallel(ContextManagerWithCall())

    FutureCollection.create(collection.__enter__)().parallel(barrier.wait)(timeout=2)


def test_parallel_with_contextmanager_creates_future_context_collection() -> None:
    """
    Verify that plain `parallel` will create a `FutureContextCollection` when given an `@contextmanager` decorated
    function.
    """
    barrier = Barrier(2)

    @contextmanager
    def context_manager() -> Generator[None, None, None]:
        """Very simple `@contextmanager`"""
        barrier.wait(timeout=2)
        yield None

    collection = parallel(context_manager())

    FutureCollection.create(collection.__enter__)().parallel(barrier.wait)(timeout=2)


def test_parallel_with_context_manager_init_creates_future_collection() -> None:
    """
    Verify that plain `parallel` will create a `FutureCollection` when given an `AbstractContextManager`'s `__init__`.
    """
    (cm,) = parallel(SignalledContext)(yields=None).results()

    assert isinstance(cm, SignalledContext)


def test_parallel_parallel_preserves_typing() -> None:
    """
    Verify that chaining `parallel` calls preserves the typing information.

    This test is nothing special. Its true value is when type checkers like mypy are used: they should not complain
    about the typing.
    """
    results = parallel(lambda: "bla")().parallel(lambda: 12)().results()
    assert_type(results, tuple[str, int])
    assert results == ("bla", 12)

    with parallel(SignalledContext(yields="bla")).parallel(SignalledContext(yields=12)) as with_results:
        assert_type(with_results, tuple[str, int])
        assert with_results == ("bla", 12)
