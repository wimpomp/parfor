from __future__ import annotations

import logging
import os
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from functools import wraps
from importlib.metadata import version
from io import StringIO
from multiprocessing.shared_memory import SharedMemory
from traceback import format_exc
from typing import (
    Any,
    Callable,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    NoReturn,
    Optional,
    Protocol,
    Sequence,
    Sized,
)

import numpy as np
import ray
from numpy.typing import ArrayLike, DTypeLike
from ray.remote_function import RemoteFunction
from ray.types import ObjectRef
from tqdm.auto import tqdm

from .pickler import dumps, loads

__version__ = version("parfor")
cpu_count = int(os.cpu_count())


class Bar(Protocol):
    def update(self, n: int = 1) -> None: ...


class SharedArray(np.ndarray):
    """Numpy array whose memory can be shared between processes, so that memory use is reduced and changes in one
    process are reflected in all other processes. Changes are not atomic, so protect changes with a lock to prevent
    race conditions!
    """

    def __new__(
        cls,
        shape: int | Sequence[int],
        dtype: DTypeLike = float,
        shm: str | SharedMemory = None,
        offset: int = 0,
        strides: tuple[int, int] = None,
        order: str = None,
    ) -> SharedArray:
        if isinstance(shm, str):
            shm = SharedMemory(shm)
        elif shm is None:
            shm = SharedMemory(create=True, size=int(np.prod(shape) * np.dtype(dtype).itemsize))  # type: ignore
        new = super().__new__(cls, shape, dtype, shm.buf, offset, strides, order)
        new.shm = shm
        return new  # type: ignore

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[int | Sequence[int], DTypeLike, str], SharedArray],
        tuple[int | tuple[int, ...], np.dtype, str],
    ]:
        return self.__class__, (self.shape, self.dtype, self.shm.name)

    def __enter__(self) -> SharedArray:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self, "shm"):
            self.shm.close()
            self.shm.unlink()

    def __del__(self) -> None:
        if hasattr(self, "shm"):
            self.shm.close()

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        if isinstance(obj, np.ndarray) and not isinstance(obj, SharedArray):
            raise TypeError("view casting to SharedArray is not implemented because right now we need to make a copy")

    @classmethod
    def from_array(cls, array: ArrayLike) -> SharedArray:
        """copy existing array into a SharedArray"""
        array = np.asarray(array)
        new = cls(array.shape, array.dtype)
        new[:] = array[:]
        return new


class Chunks(Iterable):
    """Yield successive chunks from lists.
    Usage: chunks(list0, list1, ...)
           chunks(list0, list1, ..., size=s)
           chunks(list0, list1, ..., number=n)
           chunks(list0, list1, ..., ratio=r)
    size:   size of chunks, might change to optimize division between chunks
    number: number of chunks, coerced to 1 <= n <= len(list0)
    ratio:  number of chunks / number of cpus, coerced to 1 <= n <= len(list0)
    both size and number or ratio are given: use number or ratio, unless the chunk size would be bigger than size
    both ratio and number are given: use ratio
    """

    def __init__(
        self,
        *iterables: Iterable[Any] | Sized,
        size: int = None,
        number: int = None,
        ratio: float = None,
        length: int = None,
    ) -> None:
        if length is None:
            try:
                length = min(*[len(iterable) for iterable in iterables]) if len(iterables) > 1 else len(iterables[0])
            except TypeError:
                raise TypeError(
                    "Cannot determine the length of the iterables(s), so the length must be provided as an argument."
                )
        if size is not None and (number is not None or ratio is not None):
            if number is None:
                number = int(cpu_count * ratio)
            if length >= size * number:
                number = round(length / size)
        elif size is not None:  # size of chunks
            number = round(length / size)
        elif ratio is not None:  # number of chunks
            number = int(cpu_count * ratio)
        self.iterators = [iter(arg) for arg in iterables]
        self.number_of_items = length
        self.length = min(length, number)
        self.lengths = [
            ((i + 1) * self.number_of_items // self.length) - (i * self.number_of_items // self.length)
            for i in range(self.length)
        ]

    def __iter__(self) -> Iterator[Any]:
        for i in range(self.length):
            p, q = (
                (i * self.number_of_items // self.length),
                ((i + 1) * self.number_of_items // self.length),
            )
            if len(self.iterators) == 1:
                yield [next(self.iterators[0]) for _ in range(q - p)]
            else:
                yield [[next(iterator) for _ in range(q - p)] for iterator in self.iterators]

    def __len__(self) -> int:
        return self.length


class ExternalBar(Iterable):
    def __init__(
        self,
        iterable: Iterable = None,
        callback: Callable[[int], None] = None,
        total: int = 0,
    ) -> None:
        self.iterable = iterable
        self.callback = callback
        self.total = total
        self._n = 0

    def __enter__(self) -> ExternalBar:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        return

    def __iter__(self) -> Iterator[Any]:
        for n, item in enumerate(self.iterable):
            yield item
            self.n = n + 1

    def update(self, n: int = 1) -> None:
        self.n += n

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, n: int) -> None:
        if n != self._n:
            self._n = n
            if self.callback is not None:
                self.callback(n)


def get_worker(n_processes) -> RemoteFunction:
    n_processes = n_processes or PoolSingleton.cpu_count
    num_cpus = None if n_processes is None else cpu_count / n_processes

    if not ray.is_initialized():
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        ray.init(logging_level=logging.ERROR, log_to_driver=False)

    def worker(task):
        try:
            with ExitStack() as stack:  # noqa
                if task.allow_output:
                    out = StringIO()
                    err = StringIO()
                    stack.enter_context(redirect_stdout(out))
                    stack.enter_context(redirect_stderr(err))
                else:
                    stack.enter_context(redirect_stdout(open(os.devnull, "w")))
                    stack.enter_context(redirect_stderr(open(os.devnull, "w")))
                try:
                    task()
                    task.status = ("done",)
                except Exception:  # noqa
                    task.status = "task_error", format_exc()
                if task.allow_output:
                    task.out = out.getvalue()
                    task.err = err.getvalue()
        except KeyboardInterrupt:  # noqa
            pass

        return task

    if num_cpus:
        return ray.remote(num_cpus=num_cpus)(worker)  # type: ignore
    else:
        return ray.remote(worker)  # type: ignore


class Task:
    def __init__(
        self,
        handle: Hashable,
        fun: Callable[[Any, ...], Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] = None,
        allow_output: bool = False,
    ) -> None:
        self.handle = handle
        self.fun = fun
        self.args = args
        self.kwargs = kwargs or {}
        self.name = fun.__name__ if hasattr(fun, "__name__") else None
        self.done = False
        self.result = None
        self.out = None
        self.err = None
        self.future = None
        self.status = "starting"
        self.allow_output = allow_output

    @staticmethod
    def get(item: tuple[bool, Any]) -> Any:
        if item[0]:
            return loads(ray.get(item[1]))
        else:
            return ray.get(item[1])

    @staticmethod
    def put(item: Any) -> tuple[bool, Any]:
        try:
            return False, ray.put(item)
        except Exception:  # noqa
            return True, ray.put(dumps(item, recurse=True))

    @property
    def fun(self) -> Callable[[Any, ...], Any]:
        return self.get(self._fun)

    @fun.setter
    def fun(self, fun: Callable[[Any, ...], Any]):
        self._fun = self.put(fun)

    @property
    def args(self) -> tuple[Any, ...]:
        return tuple([self.get(arg) for arg in self._args])

    @args.setter
    def args(self, args: tuple[Any, ...]) -> None:
        self._args = [self.put(arg) for arg in args]

    @property
    def kwargs(self) -> dict[str, Any]:
        return {key: self.get(value) for key, value in self._kwargs.items()}

    @kwargs.setter
    def kwargs(self, kwargs: dict[str, Any]) -> None:
        self._kwargs = {key: self.put(value) for key, value in kwargs.items()}

    @property
    def result(self) -> Any:
        return self.get(self._result)

    @result.setter
    def result(self, result: Any) -> None:
        self._result = self.put(result)

    def __call__(self) -> Task:
        if not self.done:
            self.result = self.fun(*self.args, **self.kwargs)  # noqa
            self.done = True
        return self

    def __repr__(self) -> str:
        if self.done:
            return f"Task {self.handle}, result: {self.result}"
        else:
            return f"Task {self.handle}"


class ParPool:
    """Parallel processing with addition of iterations at any time and request of that result any time after that.
    The target function and its argument can be changed at any time.
    """

    def __init__(
        self,
        fun: Callable[[Any, ...], Any] = None,
        args: tuple[Any] = None,
        kwargs: dict[str, Any] = None,
        n_processes: int = None,
        bar: Bar = None,
        allow_output: bool = False,
    ):
        self.handle = 0
        self.tasks = {}
        self.bar = bar
        self.bar_lengths = {}
        self.allow_output = allow_output
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self.n_processes = n_processes or PoolSingleton.cpu_count
        self.worker = get_worker(self.n_processes)

    def __getstate__(self) -> NoReturn:
        raise RuntimeError(f"Cannot pickle {self.__class__.__name__} object.")

    def __enter__(self) -> ParPool:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, n: Any, handle: Hashable = None, barlength: int = 1) -> None:
        self.add_task(
            args=(n, *(() if self.args is None else self.args)),
            handle=handle,
            barlength=barlength,
        )

    def close(self) -> None:
        pass

    def add_task(
        self,
        fun: Callable[[Any, ...], Any] = None,
        args: tuple[Any, ...] = None,
        kwargs: dict[str, Any] = None,
        handle: Hashable = None,
        barlength: int = 1,
        allow_output: bool = False,
    ) -> Optional[int]:
        if handle is None:
            new_handle = self.handle
            self.handle += 1
        else:
            new_handle = handle
        if new_handle in self:
            raise ValueError(f"handle {new_handle} already present")
        task = Task(
            new_handle,
            fun or self.fun,
            args or self.args,
            kwargs or self.kwargs,
            allow_output or self.allow_output,
        )
        self.block_until_space_available()
        task.future = self.worker.remote(task)
        self.tasks[new_handle] = task
        self.bar_lengths[new_handle] = barlength
        if handle is None:
            return new_handle
        else:
            return None

    def __setitem__(self, handle: Hashable, n: Any) -> None:
        """Add new iteration."""
        self(n, handle=handle)

    def __getitem__(self, handle: Hashable) -> Any:
        """Request result and delete its record. Wait if result not yet available."""
        if handle not in self:
            raise KeyError(f"No task with handle: {handle} in pool")
        task = self.finalize_task(self.tasks[handle])
        self.tasks.pop(task.handle)
        return task.result

    def __contains__(self, handle: Hashable) -> bool:
        return handle in self.tasks

    def __delitem__(self, handle: Hashable) -> None:
        self.tasks.pop(handle)

    def finalize_task(self, future: ObjectRef | Task) -> Task:
        if isinstance(future, Task):
            task: Task = future
            future = task.future
        else:
            task = None  # type: ignore

        if future is not None:
            task: Task = ray.get(future)  # type: ignore
            code, *args = task.status
            if task.out:
                if hasattr(self.bar, "write"):
                    self.bar.write(task.out, end="")
                else:
                    print(task.out, end="")
            if task.err:
                if hasattr(self.bar, "write"):
                    self.bar.write(task.err, end="")
                else:
                    print(task.err, end="")
            getattr(self, code)(task, *args)
            self.tasks[task.handle] = task
        return task

    def block_until_space_available(self) -> None:
        if len(self.tasks) < 3 * self.n_processes:
            return
        while True:
            if self.tasks:
                futures = [task.future for task in self.tasks.values() if task.future is not None]
                done, busy = ray.wait(futures, num_returns=1, timeout=0.01)
                for d in done:
                    self.finalize_task(d)  # type: ignore
                if len(busy) < 3 * self.n_processes:
                    return

    def get_newest(self) -> Any:
        """Request the newest handle and result and delete its record. Wait if result not yet available."""
        if self.tasks:
            done = [task for task in self.tasks.values() if task.future is None]
            if done:
                task = done[0]
                self.tasks.pop(task.handle)
                return task.handle, task.result
            while True:
                futures = [task.future for task in self.tasks.values() if task.future is not None]
                done, _ = ray.wait(futures, num_returns=1, timeout=0.01)
                if done:
                    task = self.finalize_task(done[0])
                    self.tasks.pop(task.handle)
                    return task.handle, task.result
        raise StopIteration

    def task_error(self, task: Task, error: Exception) -> None:
        if task.handle in self:
            task = self.tasks[task.handle]
            print(f"Error from process working on iteration {task.handle}:\n")
            print(error)
            print("Retrying in main process...")
            task()
            raise Exception(f"Function '{task.name}' cannot be executed by parfor, amend or execute in serial.")

    def done(self, task: Task) -> None:
        if task.handle in self:  # if not, the task was restarted erroneously
            self.tasks[task.handle] = task
            if hasattr(self.bar, "update"):
                self.bar.update(self.bar_lengths.pop(task.handle))


class PoolSingleton:
    cpu_count: int = os.cpu_count()


class Worker:
    nested: bool = False


def gmap(
    fun: Callable[[Any, ...], Any],
    iterable: Iterable[Any] = None,
    args: tuple[Any, ...] = None,
    kwargs: dict[str, Any] = None,
    total: int = None,
    desc: str = None,
    bar: Bar | bool = True,
    serial: bool = None,
    n_processes: int = None,
    yield_ordered: bool = True,
    yield_index: bool = False,
    allow_output: bool = False,
    **bar_kwargs: Any,
) -> Generator[Any, None, None]:
    """map a function fun to each iteration in iterable
    use as a function: pmap
    use as a decorator: parfor
    best use: iterable is a generator and length is given to this function as 'total'

    required:
        fun:    function taking arguments: iteration from  iterable, other arguments defined in args & kwargs
        iterable: iterable or iterator from which an item is given to fun as a first argument
    optional:
        args:   tuple with other unnamed arguments to fun
        kwargs: dict with other named arguments to fun
        total:  give the length of the iterator in cases where len(iterator) results in an error
        desc:   string with description of the progress bar
        bar:    bool enable progress bar,
                    or a callback function taking the number of passed iterations as an argument
        serial: execute in series instead of parallel if True, None (default): let pmap decide
        n_processes: number of processes to use,
            the parallel pool will be restarted if the current pool does not have the right number of processes
        yield_ordered: return the result in the same order as the iterable
        yield_index: return the index of the result too
        allow_output: allow output from subprocesses
        **bar_kwargs: keywords arguments for tqdm.tqdm

    output:
        list (pmap) or generator (gmap) with results from applying the function \'fun\' to each iteration
         of the iterable / iterator

    examples:
        << from time import sleep
        <<
        @parfor(range(10), (3,))
        def fun(i, a):
            sleep(1)
            return a * i ** 2
        fun
        >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]

        <<
        def fun(i, a):
            sleep(1)
            return a * i ** 2
        pmap(fun, range(10), (3,))
        >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]

        equivalent to using the deco module:
        <<
        @concurrent
        def fun(i, a):
            time.sleep(1)
            return a * i ** 2

        @synchronized
        def run(iterator, a):
            res = []
            for i in iterator:
                res.append(fun(i, a))
            return res
        run(range(10), 3)
        >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]

        all equivalent to the serial for-loop:
        <<
        a = 3
        fun = []
        for i in range(10):
            sleep(1)
            fun.append(a * i ** 2)
        fun
        >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]
    """
    is_chunked = isinstance(iterable, Chunks)
    if is_chunked:
        chunk_fun = fun
    else:
        iterable = Chunks(iterable, ratio=5, length=total)

        @wraps(fun)
        def chunk_fun(iterable: Iterable, *args: Any, **kwargs: Any) -> list[Any]:  # noqa
            return [fun(iteration, *args, **kwargs) for iteration in iterable]

    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    if "total" not in bar_kwargs:
        bar_kwargs["total"] = sum(iterable.lengths)
    if "desc" not in bar_kwargs:
        bar_kwargs["desc"] = desc
    if "disable" not in bar_kwargs:
        bar_kwargs["disable"] = not bar
    if serial is True or (serial is None and len(iterable) < min(cpu_count, 4)):  # serial case

        def tqdm_chunks(chunks: Chunks, *args, **kwargs) -> Iterable[Any]:  # noqa
            with tqdm(*args, **kwargs) as b:
                for chunk, length in zip(chunks, chunks.lengths):  # noqa
                    yield chunk
                    b.update(length)

        iterable = (
            ExternalBar(iterable, bar, sum(iterable.lengths)) if callable(bar) else tqdm_chunks(iterable, **bar_kwargs)  # type: ignore
        )
        if is_chunked:
            if yield_index:
                for i, c in enumerate(iterable):
                    yield i, chunk_fun(c, *args, **kwargs)
            else:
                for c in iterable:
                    yield chunk_fun(c, *args, **kwargs)
        else:
            if yield_index:
                for i, c in enumerate(iterable):
                    for q in chunk_fun(c, *args, **kwargs):
                        yield i, q
            else:
                for c in iterable:
                    yield from chunk_fun(c, *args, **kwargs)

    else:  # parallel case
        with ExitStack() as stack:  # noqa
            if callable(bar):
                bar = stack.enter_context(ExternalBar(callback=bar))  # noqa
            else:
                bar = stack.enter_context(tqdm(**bar_kwargs))
            with ParPool(chunk_fun, args, kwargs, n_processes, bar, allow_output) as p:  # type: ignore
                for i, (j, l) in enumerate(zip(iterable, iterable.lengths)):  # add work to the queue
                    p(j, handle=i, barlength=l)
                    if bar.total is None or bar.total < i + 1:
                        bar.total = i + 1

                if is_chunked:
                    if yield_ordered:
                        if yield_index:
                            for i in range(len(iterable)):
                                yield i, p[i]
                        else:
                            for i in range(len(iterable)):
                                yield p[i]
                    else:
                        if yield_index:
                            for _ in range(len(iterable)):
                                yield p.get_newest()
                        else:
                            for _ in range(len(iterable)):
                                yield p.get_newest()[1]
                else:
                    if yield_ordered:
                        if yield_index:
                            for i in range(len(iterable)):
                                for q in p[i]:
                                    yield i, q
                        else:
                            for i in range(len(iterable)):
                                yield from p[i]
                    else:
                        if yield_index:
                            for _ in range(len(iterable)):
                                i, n = p.get_newest()
                                for q in n:
                                    yield i, q
                        else:
                            for _ in range(len(iterable)):
                                yield from p.get_newest()[1]


def pmap(*args, **kwargs) -> list[Any]:
    return list(gmap(*args, **kwargs))


def parfor(*args: Any, **kwargs: Any) -> Callable[[Callable[[Any, ...], Any]], list[Any]]:
    def decfun(fun: Callable[[Any, ...], Any]) -> list[Any]:
        return pmap(fun, *args, **kwargs)

    return decfun


try:
    parfor.__doc__ = pmap.__doc__ = gmap.__doc__
    pmap.__annotations__ = gmap.__annotations__ | pmap.__annotations__
    parfor.__annotations__ = {key: value for key, value in pmap.__annotations__.items() if key != "fun"}
except AttributeError:
    pass
