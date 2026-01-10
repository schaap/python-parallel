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

import sys
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import AbstractContextManager
from types import TracebackType
from typing import (
    Any,
    Generic,
    Literal,
    NoReturn,
    ParamSpec,
    TypeVar,
    final,
    overload,
)

from parallel.return_when import ReturnWhen

T1 = TypeVar("T1")
P = ParamSpec("P")

if sys.version_info >= (3, 11):  # noqa: UP036
    from typing import (
        TypeVarTuple,
        Unpack,
    )

    Results = TypeVarTuple("Results")

    @final
    class FutureCollection(Generic[Unpack[Results]]):
        def __init__(self) -> None: ...
        @property
        def futures(self) -> tuple[Future[Any], ...]: ...
        @overload
        def results(
            self,
            *,
            timeout: float | None = None,
            return_when: Literal[ReturnWhen.ALL_COMPLETED] = ReturnWhen.ALL_COMPLETED,
        ) -> tuple[Unpack[Results]]: ...
        @overload
        def results(
            self, *, timeout: float | None = None, return_when: Literal[ReturnWhen.FIRST_EXCEPTION]
        ) -> tuple[Unpack[Results]]: ...
        @overload
        def results(
            self, *, timeout: float | None = None, return_when: Literal[ReturnWhen.FIRST_COMPLETED]
        ) -> NoReturn:
            """
            The semantics of FIRST_COMPLETED can't be supported by results.
            """

        def parallel(self, func: Callable[P, T1], /) -> Callable[P, FutureCollection[Unpack[Results], T1]]: ...
        @staticmethod
        def create(func: Callable[P, T1], /) -> Callable[P, FutureCollection[T1]]: ...

    @final
    class FutureContextCollection(Generic[Unpack[Results]]):
        def __init__(self, *, timeout: float | None = None, parallel_exit: bool = False) -> None: ...
        def __enter__(self) -> tuple[Unpack[Results]]: ...
        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            exc_traceback: TracebackType | None,
        ) -> bool | None: ...
        def parallel(
            self, context_manager: AbstractContextManager[T1], /
        ) -> FutureContextCollection[Unpack[Results], T1]: ...
        @staticmethod
        def create(
            context_manager: AbstractContextManager[T1], /, *, timeout: float | None = None, parallel_exit: bool = False
        ) -> FutureContextCollection[T1]: ...

    # Be sure to keep the AbstractContextManager variant up front. This ensures correct detection of this signature
    # when using ContextDecorator objects, like @contextmanager, which are callable by default.
    @overload
    def parallel(
        context_manager: AbstractContextManager[T1], /, *, timeout: float | None = None, parallel_exit: bool = False
    ) -> FutureContextCollection[T1]: ...
    @overload
    def parallel(func: Callable[P, T1], /) -> Callable[P, FutureCollection[T1]]: ...

else:
    @final
    class FutureCollection:
        def __init__(self) -> None: ...
        @property
        def futures(self) -> tuple[Future[Any], ...]: ...
        @overload
        def results(
            self,
            *,
            timeout: float | None = None,
            return_when: Literal[ReturnWhen.ALL_COMPLETED] = ReturnWhen.ALL_COMPLETED,
        ) -> tuple[Any, ...]: ...
        @overload
        def results(
            self, *, timeout: float | None = None, return_when: Literal[ReturnWhen.FIRST_EXCEPTION]
        ) -> tuple[Any, ...]: ...
        @overload
        def results(
            self, *, timeout: float | None = None, return_when: Literal[ReturnWhen.FIRST_COMPLETED]
        ) -> NoReturn:
            """
            The semantics of FIRST_COMPLETED can't be supported by results.
            """

        def parallel(self, func: Callable[P, T1], /) -> Callable[P, FutureCollection]: ...
        @staticmethod
        def create(func: Callable[P, T1], /) -> Callable[P, FutureCollection]: ...

    @final
    class FutureContextCollection:
        def __init__(self, *, timeout: float | None = None, parallel_exit: bool = False) -> None: ...
        def __enter__(self) -> tuple[Any, ...]: ...
        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            exc_traceback: TracebackType | None,
        ) -> bool | None: ...
        def parallel(self, context_manager: AbstractContextManager[T1], /) -> FutureContextCollection: ...
        @staticmethod
        def create(
            context_manager: AbstractContextManager[T1], /, *, timeout: float | None = None, parallel_exit: bool = False
        ) -> FutureContextCollection: ...

    # Be sure to keep the AbstractContextManager variant up front. This ensures correct detection of this signature
    # when using ContextDecorator objects, like @contextmanager, which are callable by default.
    @overload
    def parallel(
        context_manager: AbstractContextManager[T1], /, *, timeout: float | None = None, parallel_exit: bool = False
    ) -> FutureContextCollection: ...
    @overload
    def parallel(func: Callable[P, T1], /) -> Callable[P, FutureCollection]: ...
