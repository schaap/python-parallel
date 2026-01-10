# python-parallel
Call python functions in parallel using threads.

# Features
- Strong expressivity
- Support for parallel entering multiple context managers in a `with` statement, with semantics as close to regular `with` as possible
- Fully typed parallel calls in Python 3.11 and later
- Guaranteed parallel execution, with each call on its own thread
- Uses exception groups in Python 3.11 and later, as defined in [PEP 654](https://peps.python.org/pep-0654/)
- Backwards compatible to Python 3.8

# How To Use
Below is a very simple example that shows the basic use of the chained `parallel` calls, both for calling functions and for entering context managers.

<!-- This example is also in tests/example1.py -->
```python
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
```

## Parallel Function Calls
The first part performs a few function calls in parallel. Each call to `parallel()` prepares the function to be called in parallel and returns a function object with the signature of the function to be called. Calling that function object will start the parallel call. The returned function object is fully typed and IDEs should have full parameter documentation support as if you were calling the original function.

The `.results()` call will block until the parallel calls have finished. It will raise exceptions from those calls or return the values of each call in the same order as they were called with `parallel`.

It is not required to chain all parallel function calls immediately, as in the example. You can store them to a variable, add more calls later on, and finally call `.results()` when you're ready for it. This allows, for example, starting a long lasting calculation early on while the rest of your code continues.

For better handling of the results `.results()` has a number of options, and you can even extract the underlying `Future` objects directly for even greater control.

## Parallel Context Managers
The second part sets up a few context managers which will be entered in parallel. The `parallel()` calls in the `with` statement set up a specialized context manager that contains the context managers provided to `parallel()`. When entering this specialized context manager, the contained context managers are entered in parallel.

The parallel context managers feature mimics the way `with` would normally work with multiple context managers, including for things like exception handling. There are some differences, due to the calls being in parallel instead of in sequence, especially when it comes to the specifics of exception handling. One obvious difference is of course that the context managers that are called in parallel can't use each other's results.

The initial `parallel()` call has a number of additional parameters to better configure the behavior, such as a timeout and the option to also exit the context managers in parallel.

## Dynamic Parallel Calls
Due to their typing intricacies, the `parallel()` calls can't be called dynamically, for example in a loop. The following would grealy confuse the typing system:

<!-- This example is not in tests/, as it's not valid -->
```python
from parallel.parallel import parallel

calls = parallel(lambda: 1)
for i in range(1, 10):
    calls = calls.parallel(lambda: i)
call.results()
```

For cases like this, you can use the underlying untyped system, but you'll have to do the `Future` handling yourself.

<!-- This example is also in tests/example2.py -->
```python
from concurrent.futures import Future, wait
from parallel.parallel_executor import ParallelExecutor

futures: list[Future[int]] = []
for i in range(10):
    futures.append(ParallelExecutor.execute_one(lambda: i))
done, not_done = wait(futures)
```

# Development
Development environments depend on `uv`. To create a virtual environment for python-parallel, run the following command:

    uv sync --group lint --group test

You can run code linters as follows (development was done using the `--preview` flags for `ruff`):

    uv run mypy .
    uv run ruff format  # --preview
    uv run ruff check  # --preview

You can run the tests as follows:

    uv run pytest .

You will notice that some tests are always skipped. This is correct: some tests are parametrized on the Python version.

By default the Python version is set to 3.11 in `.python-version`. Feel free to change this to the Python version you need, e.g. 3.8 on older systems.
