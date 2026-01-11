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
Helpers for calling functions or entering context managers in parallel, using threading.

This module's contents can be used to conveniently have some calls run in parallel, with minimal boilerplate while
retaining all typing information and (almost) all semantics that a non-parallel version has.

Usage example: ::

    connection, file_contents = (
        parallel(setup_secure_connection)(server, certificate).parallel(read_file_contents)(big_file)
    ).results()
    with parallel(TapeWriter(tape_device, seek=tape_offset)).parallel(
        DatabaseConnection(db_name, db_user, db_pass)
    ) as (tape, db):
        (
            parallel(tape.write_records)({url: connection.download(url), big_file: file_contents})
            .parallel(db.write_file)(url, connection.download(url))
            .parallel(db.write_file)(big_file, file_contents)
        ).results()

Note that exception groups will be used if available, so `except*` is preferred for exception handling.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from concurrent.futures import Future, wait
from contextlib import AbstractContextManager, ExitStack
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    final,
    overload,
)

from parallel.parallel_executor import ParallelExecutor
from parallel.return_when import ReturnWhen

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

if sys.version_info <= (3, 10):  # noqa: UP036  # Backwards compatibility
    from typing_extensions import ParamSpec
else:  # pragma: no cover  # Backwards compatibility
    from typing import ParamSpec


T = TypeVar("T")
P = ParamSpec("P")


# NOTE: Parts of this file are further typed in the accompanying .pyi file


def _extract_exceptions(futures_with_exception: list[Future[Any]]) -> list[BaseException]:
    """
    Extract the exceptions from a number of finished `Future` objects which are known to have an exception.

    :param futures_with_exception: The `Future` objects to extract the exceptions from.
    :returns: The extracted exceptions. `BaseException` instances will be listed first.
    """
    exceptions: list[BaseException] = []
    for future in futures_with_exception:
        try:
            future.result()
        except BaseException as exc:  # noqa: BLE001
            # BaseException might seem like overkill (Exception is usually enough), but if we can be
            # transparent, why not be transparent? The BaseExceptionGroup constructor will actually construct an
            # ExceptionGroup if only Exception instances are given, too, so we basically get best of all worlds.
            exceptions.append(exc)
    # Sort BaseExceptions before other Exceptions, so naive unpacking of group.exceptions[0] gets the most
    # important exception
    exceptions.sort(key=lambda x: isinstance(x, Exception))
    return exceptions


def _wait_for_futures(
    futures: tuple[Future[T], ...],
    /,
    *,
    timeout: float | None = None,
    return_when: ReturnWhen = ReturnWhen.ALL_COMPLETED,
    on_exception: Callable[
        [set[Future[T]], type[BaseException] | None, BaseException | None, TracebackType | None], None
    ]
    | None = None,
) -> tuple[Any]:
    """
    Helper for waiting until a list of futures has finished.

    If an exception occurs, including the locally generated `TimeoutError` when the `futures` take more than `timeout`
    seconds, `on_exception` will be called with the set of futures that are done. These include any futures that have
    an exception.

    :param futures: The futures to wait for
    :param timeout: If set, the amount of time the `futures` may take to finish.
    :param return_when: When to return a result. `ReturnWhen.FIRST_COMPLETED` is not supported.
    :param on_exception: The callable for handling the finished futures when an exception is being handled. The
                         exception will be reraised right after this function returns.
    :returns: The results of the `futures`, in that same order.
    """
    if return_when == ReturnWhen.FIRST_COMPLETED:
        raise RuntimeError("The semantics of FIRST_COMPLETED can't be supported by results.")

    # Wait for the futures to finish
    done, not_done = wait(cast(Iterable[Future[T]], futures), timeout=timeout, return_when=return_when.value)
    futures_with_exception = [future for future in done if future.exception()]
    futures_with_baseexception = [
        future for future in futures_with_exception if not isinstance(future.exception(), Exception)
    ]

    # Handle FIRST_EXCEPTION before timeout
    if return_when == ReturnWhen.FIRST_EXCEPTION and futures_with_exception:
        # Prefer a BaseException over a normal Exception
        if futures_with_baseexception:
            futures_with_baseexception[0].result()
        futures_with_exception[0].result()

    try:
        try:
            # Detect timeout
            if not_done:
                raise TimeoutError(f"Not all futures completed within {timeout} seconds")
        finally:
            # Handle exceptions
            if futures_with_exception:
                # Note that exception extraction is done using .result(): this call has specific optimizations over
                # .exception() that should be taken advantage of for normal program flow.

                if len(futures_with_exception) == 1 or sys.version_info < (3, 11):
                    # Prefer a BaseException over a normal Exception
                    if futures_with_baseexception:
                        futures_with_baseexception[0].result()
                    futures_with_exception[0].result()
                    raise RuntimeError(
                        "Finished future's exception was NOT raised by retrieving the future's result"
                    )  # pragma: no cover  # Impossible to hit unless concurrent.future has a severe bug

                # Multiple exceptions: extract them all and reraise as a group
                exceptions = _extract_exceptions(futures_with_exception)
                raise BaseExceptionGroup(f"{len(exceptions)} futures yielded an exception", exceptions)
    except:
        if on_exception:
            on_exception(done, *sys.exc_info())
        raise

    # Extract and return all results. Use original futures list to preserve order.
    return cast(tuple[Any], tuple(future.result() for future in futures))


@final
class FutureCollection:
    """
    Helper class for running a number of calls in parallel.

    `FutureCollection` allows a stronger expressivity without loosing the result types of parallel calls: ::

        futures = (
            FutureCollection
            .create(gives_int)(an_argument, key=word_argument)
            .parallel(gives_float)()
            .parallel(gives_list_of_str)(*list_of_arguments)
        )
        an_int, a_float, a_list_of_str = futures.results()
        # These result variables have their types correctly recognized by the typing system.

    Using the `parallel()` method, another call can be started in parallel.

    The `results()` method can be used to obtain the (correctly typed) results of all the parallel calls. The
    `futures()` method can be used to obtain the `Future` instances for the call, although these are not correctly typed
    due to a limitation in the typing system.

    Note that using `parallel()` is destructive for the instance it is called upon. This is by design: if you were to
    chain several calls in parallel but would results from an intermediate object, things could get very confusing. Only
    use the resulting `FutureCollection` from the last `parallel()` call.

    `FutureCollection` is itself not thread-safe.
    """

    def __init__(self) -> None:
        """
        Create a new, empty `FutureCollection`.
        """
        self._futures: tuple[Future[Any], ...] | None = ()

    def _pass_futures(self, *futures: Future[Any]) -> None:
        """
        Pass the futures of one `FutureCollection` to another.
        """
        self._futures = futures

    @property
    def futures(self) -> tuple[Future[Any], ...]:
        """
        The `Future` instances of all the parallel calls, in the same order as they were chained.

        The encapsulated types of the returned `Future` instances match those of `results()`, but can't be typed as such
        due to a limitation of the typing system.
        """
        if self._futures is None:
            raise RuntimeError("The futures from this collection have been passed on by a call to parallel().")
        return self._futures

    def results(
        self, *, timeout: float | None = None, return_when: ReturnWhen = ReturnWhen.ALL_COMPLETED
    ) -> tuple[Any]:
        """
        Obtain the results of the calls.

        Python <3.11: If an exception occurred in one of the threads, the first encountered exception is raised. Other
        exceptions are not raised.

        :param timeout: The amount of time to wait for the results to become available.
        :param return_when: When `results` will return. If set to `ReturnWhen.FIRST_EXCEPTION`, the first exception to
                            occur will be raised immediately, even when the other threads haven't finished yet.
        :returns: The results of all the parallel calls, in the order they were chained.
        :raises TimeoutError: If `timeout` was set and the result were not available before that time.
        :raises Exception: If exactly one of the calls finished with an `Exception`, or at least one of the calls
                           finished with an exception and none of the calls finished with a `BaseException` and
                           `return_when` was set to `ReturnWhen.FIRST_EXCEPTION`.
        :raises ExceptionGroup: If multiple calls finished with an `Exception`, none of which was a `BaseException`.
        :raises BaseException: If exactly one of the calls finished with a BaseException, or at least one of the calls
                               finished with an exception and `return_when` was set to `ReturnWhen.FIRST_EXCEPTION`.
        :raises BaseExceptionGroup: If multiple calls finished with an exception and at least one of them was a
                                    `BaseException`.
        """
        if self._futures is None:
            raise RuntimeError("The futures from this collection have been passed on by a call to parallel().")
        return _wait_for_futures(self._futures, timeout=timeout, return_when=return_when)

    def parallel(self, func: Callable[P, T], /) -> Callable[P, FutureCollection]:
        """
        Call another function in parallel.

        This will create a new `FutureCollection` instance, which replaces this instance. All calls on this instance
        become invalid after the function object returned by `parallel()` has been called.

        :param func: The function to call.
        :returns: A function object with the same signature as `func`, except that it returns a new `FutureCollection`
                  for all the `Future` instances already in this instance, and the `Future` for the call to `func`
                  appended to the end.
        """
        if self._futures is None:
            raise RuntimeError("The futures from this collection have been passed on by a call to parallel().")

        @wraps(func)
        def wrapped_parallel_func(*args: P.args, **kwargs: P.kwargs) -> FutureCollection:
            if self._futures is None:
                raise RuntimeError("The futures from this collection have been passed on by a call to parallel().")

            extended_collection = FutureCollection()
            extended_collection._pass_futures(*self._futures, ParallelExecutor.execute_one(func, *args, **kwargs))
            self._futures = None
            return extended_collection

        return wrapped_parallel_func

    @staticmethod
    def create(func: Callable[P, T], /) -> Callable[P, FutureCollection]:
        """
        Call a function in parallel.

        :param func: The function to call.
        :returns: A function object with the same signature as `func`, except that it returns a `FutureCollection` with
                  only the `Future` for the call to `func`.
        """

        @wraps(func)
        def wrapped_parallel_func(*args: P.args, **kwargs: P.kwargs) -> FutureCollection:
            collection = FutureCollection()
            collection._pass_futures(ParallelExecutor.execute_one(func, *args, **kwargs))
            return collection

        return wrapped_parallel_func


@final
class FutureContextCollection:
    """
    Helper class for entering a number of context managers in parallel.

    `FutureContextCollection` allows a stronger expressivity without loosing the result types of parallel calls: ::

        with (
            FutureContextCollection.create(enters_with_int)
            .parallel(enters_with_float)
            .parallel(enters_with_list_of_str)
        ) as (an_int, a_float, a_list_of_str):
            # These result variables have their types correctly recognized by the typing system.

    Using the `parallel()` method, another context can be entered in parallel.

    Note that using `parallel()` is destructive for the instance it is called upon. This is by design: if you were to
    chain several calls in parallel but would results from an intermediate object, things could get very confusing. Only
    use the resulting `FutureContextCollection` from the last `parallel()` call.

    `FutureContextCollection` is itself not thread-safe.
    """

    def __init__(self, *, timeout: float | None = None, parallel_exit: bool = False) -> None:
        """
        Create a new, empty `FutureContextCollection`.

        :param timeout: The amount of seconds that the context managers may take to `__enter__` their contexts. Note
                        that this timeout is a last resort, as a context managers that does not finish `__enter__`
                        within this time will never have its `__exit__` called, even when its `__enter__` eventually
                        does finish.
        :param parallel_exit: If `True`, leaving the `FutureContextCollection`'s context will leave the contained
                              context managers in parallel, as well. Note that this does not apply when an exception
                              occurs while entering the contained context managers and the already entered contained
                              context managers need to be left again.
        """
        self._timeout = timeout
        self._context_managers: tuple[AbstractContextManager[Any], ...] | None = ()
        self._entered_contexts: tuple[AbstractContextManager[Any], ...] | None = None
        self._parallel_exit = parallel_exit

    def _pass_context_managers(self, *context_managers: AbstractContextManager[Any]) -> None:
        """
        Pass the context managers of one `FutureContextCollection` to another.
        """
        self._context_managers = context_managers

    def __enter__(self) -> tuple[Any]:
        """
        Enter all the context managers in parallel.

        If one or more of the context managers fail to enter, or fail to do so within the timeout, then those context
        managers that have successfully been entered will be exited again before the exception is raised. This mimics
        the semantics of `with`, with the exception that `FutureContextCollection` behaves as if the successfully
        parallel entered context managers were entered before those that failed regardless of the order in which they
        were chained. Additionally, the return value of `__exit__` will be ignored when handling an exception that
        occurred during `__enter__`.

        Python <3.11: If an exception occurred in one of the threads, the first encountered exception is raised. Other
        exceptions are not raised.

        :returns: The results of entering all the context managers (in parallel fashion), in the order they were
                  chained.
        :raises TimeoutError: If `timeout` was set and the result were not available before that time.
        :raises Exception: If entering exactly one of the context managers finished with an Exception.
        :raises ExceptionGroup: If entering multiple context managers finished with an Exception, none of which was a
                                BaseException.
        :raises BaseException: If entering exactly one of the context managers finished with a BaseException.
        :raises BaseExceptionGroup: If entering multiple context managers finished with an exception and at least one of
                                    them was a BaseException.
        """
        if self._context_managers is None:
            raise RuntimeError("The context managers from this collection have been passed on by a call to parallel().")
        if self._entered_contexts is not None:
            raise RuntimeError("The FutureContextCollection's context has already been entered.")

        def enter_context_parallel(ctx: AbstractContextManager[T]) -> tuple[AbstractContextManager[T], T]:
            """
            Simple wrapper for entering a context manager but adding the original context manager to the result. This
            facilitates setting up the `__exit__` handling.

            :param ctx: The context manager to enter.
            :returns: The context manager and the result of entering it.
            """
            return (ctx, ctx.__enter__())

        futures = tuple(
            ParallelExecutor.execute_one(enter_context_parallel, context_manager)
            for context_manager in self._context_managers
        )

        class NoSuppressExit:
            """
            Simple wrapper class with a valid __exit__ call that will not suppress exceptions (even if the wrapped
            __exit__ call would attempt to suppress it).
            """

            def __init__(self, cm: AbstractContextManager[Any, bool | None]) -> None:
                """
                Create a new NoSuppressExit.

                :param cm: The already entered context manager to exit.
                """
                self._cm = cm

            def __enter__(self) -> None:  # pragma: no cover  # This is a wrapper for exiting context managers, only
                raise NotImplementedError

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_value: BaseException | None,
                exc_traceback: TracebackType | None,
            ) -> bool | None:
                self._cm.__exit__(exc_type, exc_value, exc_traceback)
                return None

        def leave_entered_contexts_on_exception(
            done: set[Future[tuple[AbstractContextManager[T], T]]],
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            exc_traceback: TracebackType | None,
        ) -> None:
            """
            Handler for cleanup when an exception occurs while the context managers are being entered.

            This will attempt to `__exit__` the context managers that have finished `__enter__`. This mimics the
            semantics of with: if you enter two contexts and the second fails to `__enter__`, the first has already been
            `__enter__`-ed and hence will be `__exit__`-ed.

            :param done: The set of `Future`s that have finished.
            :param exc_type: The type of the exception being handled.
            :param exc_value: The exception being handled.
            :param exc_traceback: The traceback of the exception being handled.
            """
            futures_to_exit = [
                future for future in futures if future in done and not future.exception() and not future.cancelled()
            ]
            # Use an ExitStack to support chaining exceptions raised in the __exit__ of the already entered context
            # managers
            # When one of the __enter__ calls fails, it is not supported to swallow exceptions of the failed __enter__
            # calls or any failed __exit__ calls (for the context managers that did succeed to __enter__). The context
            # managers are wrapped in a NoSuppressExit to ignore any attempts to suppress exceptions.
            exit_stack = ExitStack().__enter__()
            for future in futures_to_exit:
                exit_stack.push(NoSuppressExit(future.result()[0]))
            exit_stack.__exit__(exc_type, exc_value, exc_traceback)

        # Wait until all context managers have been entered.
        results = _wait_for_futures(futures, timeout=self._timeout, on_exception=leave_entered_contexts_on_exception)

        # Track the entered contexts
        self._entered_contexts = tuple(result[0] for result in results)

        # Extract and return all the actual results of the __enter__ calls.
        return tuple(result[1] for result in results)

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: TracebackType | None
    ) -> bool | None:
        """
        Leave the context of this `FutureContextCollection`. All previously entered context managers inside this object
        will be left, as well.

        The context managers' `__exit__` will be called normally, exactly as if the context managers were entered one
        by using nested with-statements.

        If parallel exit is requested, the `__exit__` of all context managers will be called in parallel. Exceptions
        from one `__exit__` method will not be passed to another. Exceptions will be grouped similar to `__enter__`.

        :param exc_type: The type of the exception being handled.
        :param exc_value: The exception being handled.
        :param exc_traceback: The traceback of the exception being handled.
        :returns: True iff the exception being handled is to be suppressed.
        """
        if self._entered_contexts is None:
            return None

        try:
            if self._parallel_exit:
                # Use a FutureCollection to do the heavy lifting of parallelizing the __exit__ calls, and handling all
                # the parallel exceptions
                exc_info = (exc_type, exc_value, exc_traceback)
                futures = tuple(
                    ParallelExecutor.execute_one(context_manager.__exit__, *exc_info)
                    for context_manager in self._entered_contexts
                )
                return any(_wait_for_futures(futures))

            # Push all the entered contexts on an ExitStack. ExitStack can do the heavy lifting of correct cleanup,
            # including all the corner cases and nifty stack manipulations they have for better exception handling
            # and propagation.
            exit_stack = ExitStack().__enter__()
            for context_manager in self._entered_contexts:
                exit_stack.push(context_manager)
            return exit_stack.__exit__(exc_type, exc_value, exc_traceback)
        finally:
            self._entered_contexts = None

    def parallel(self, context_manager: AbstractContextManager[Any], /) -> FutureContextCollection:
        """
        Set up another context manager for parallel entering.

        This will create a new `FutureContextCollection` instance, which replaces this instance. All calls on this
        instance become invalid after `parallel()` has been called.

        :param func: The function to call.
        :returns: A `FutureContextCollection` with the same context managers in this object and the new
                  `context_manager` appended to the end.
        """
        if self._context_managers is None:
            raise RuntimeError("The context managers from this collection have been passed on by a call to parallel().")
        if self._entered_contexts:
            raise RuntimeError("The FutureContextCollection's context has already been entered.")

        extended_collection = FutureContextCollection(timeout=self._timeout, parallel_exit=self._parallel_exit)
        extended_collection._pass_context_managers(*self._context_managers, context_manager)
        self._context_managers = None
        return extended_collection

    @staticmethod
    def create(
        context_manager: AbstractContextManager[Any], /, *, timeout: float | None = None, parallel_exit: bool = False
    ) -> FutureContextCollection:
        """
        Set up a context manager for parallel entering.

        :param context_manager: The context manager to enter.
        :param timeout: The amount of time that all the context managers may take to `__enter__` their contexts. Note
                        that this timeout is a last resort, as a context manager that does not finish `__enter__`
                        within this time will never have its `__exit__` called, even when its `__enter__` eventually
                        does finish.
        :param parallel_exit: If `True`, leaving the `FutureContextCollection`'s context will leave the contained
                              context managers in parallel, as well. Note that this does not apply when an exception
                              occurs while entering the contained context managers and the already entered contained
                              context managers need to be left again.
        :returns: A `FutureContextCollection` with the one `context_manager` in it.
        """
        collection = FutureContextCollection(timeout=timeout, parallel_exit=parallel_exit)
        collection._pass_context_managers(context_manager)
        return collection


@overload
def parallel(
    context_manager: AbstractContextManager[T], /, *, timeout: float | None = None, parallel_exit: bool = False
) -> FutureContextCollection:
    """
    Enter one or more context managers in parallel.

    This function sets up the first context manager for parallel entering. Additional context managers can be added to
    the result object. All context managers will be entered at the same time, in parallel and on their own threads, when
    the result object is being entered.

    Usage: ::

        with (
            parallel(open("file1", "r"))
            .parallel(yields_an_int())
            .parallel(ExpensiveObjectToEnter())
        ) as (the_file, an_int, expensive_object):
            # The result variables have their types correctly recognized by the typing system.

    Except for some details regarding leaving the context if an exception occurs, this is equivalent to:

        with open("file1", "r") as the_file, yields_an_int() as an_int, ExpensiveObjectToEnter() as expensive_object:
            # Same result.

    Note on signature preference: `parallel` prefers `AbstractContextManager` over `Callable`. To force your context
    manager to be considered a `Callable`, instead, please use a lambda (do not use `cast`: the check is performed at
    runtime).

    :param context_manager: The context manager to enter.
    :param timeout: The amount of time that all the context managers may take to `__enter__` their contexts. Note that
                    this timeout is a last resort, as a context manager that does not finish `__enter__` within this
                    time will never have its `__exit__` called, even when its `__enter__` eventually does finish.
    :param parallel_exit: If `True`, leaving the `FutureContextCollection`'s context will leave the contained context
                          managers in parallel, as well. Note that this does not apply when an exception occurs while
                          entering the contained context managers and the already entered contained context managers
                          need to be left again.
    :returns: A `FutureContextCollection` that will enter the given context manager. Use the `FutureContextCollection`
              to add additional context managers to be entered in parallel, then enter the `FutureContextCollection` to
              enter all those context managers in parallel.
    """


@overload
def parallel(func: Callable[P, T], /) -> Callable[P, FutureCollection]:
    """
    Call one or more functions in parallel.

    This function sets up the first parallel call. Additional calls can be chained on the result object. All calls will
    start to execute immediately on their own threads.

    Usage: ::

        futures = (
            parallel(gives_int)(an_argument, key=word_argument)
            .parallel(gives_float)()
            .parallel(gives_list_of_str)(*list_of_arguments)
        )
        an_int, a_float, a_list_of_str = futures.results()
        # The result variables have their types correctly recognized by the typing system.

    Note on signature preference: `parallel` prefers `AbstractContextManager` over `Callable`. To force your context
    manager to be considered a `Callable`, instead, please use a lambda (do not use `cast`: the check is performed at
    runtime).

    :param func: The function to call in parallel.
    :returns: A function object with the same signature as `func`, except that it returns a `FutureCollection` with the
              `Future` instance for calling `func`. Use the `FutureCollection` to add additional parallel calls (e.g. in
              chain calling fashion) or to obtain the results of the parallel calls.
    """


def parallel(
    func_or_ctx: AbstractContextManager[T] | Callable[P, T],
    /,
    *,
    timeout: float | None = None,
    parallel_exit: bool = False,
) -> FutureContextCollection | Callable[P, FutureCollection]:
    # Do *not* check whether func_or_ctx is callable, instead: an AbstractContextManager is not unlikely to also be
    # callable. In fact, every ContextDecorator (e.g. a @contextmanager-decorated function) is callable, but is
    # *intended* to be a context manager.
    if isinstance(func_or_ctx, AbstractContextManager):
        return FutureContextCollection.create(
            cast(AbstractContextManager[T], func_or_ctx), timeout=timeout, parallel_exit=parallel_exit
        )

    return FutureCollection.create(func_or_ctx)
