from __future__ import annotations

import sys
from contextlib import ExitStack
from functools import wraps
from importlib.metadata import version
from typing import Any, Callable, Generator, Iterable, Iterator, Sized, TypeVar
from warnings import warn

from tqdm.auto import tqdm

from . import gil, nogil
from .common import Bar, cpu_count

if hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled():  # noqa
    from .nogil import ParPool, PoolSingleton, Task, Worker
else:
    from .gil import ParPool, PoolSingleton, Task, Worker


__version__ = version('parfor')


Result = TypeVar('Result')
Iteration = TypeVar('Iteration')


class Chunks(Iterable):
    """ Yield successive chunks from lists.
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

    def __init__(self, *iterables: Iterable[Any] | Sized[Any], size: int = None, number: int = None,
                 ratio: float = None, length: int = None) -> None:
        if length is None:
            try:
                length = min(*[len(iterable) for iterable in iterables]) if len(iterables) > 1 else len(iterables[0])
            except TypeError:
                raise TypeError('Cannot determine the length of the iterables(s), so the length must be provided as an'
                                ' argument.')
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
        self.lengths = [((i + 1) * self.number_of_items // self.length) - (i * self.number_of_items // self.length)
                        for i in range(self.length)]

    def __iter__(self) -> Iterator[Any]:
        for i in range(self.length):
            p, q = (i * self.number_of_items // self.length), ((i + 1) * self.number_of_items // self.length)
            if len(self.iterators) == 1:
                yield [next(self.iterators[0]) for _ in range(q - p)]
            else:
                yield [[next(iterator) for _ in range(q - p)] for iterator in self.iterators]

    def __len__(self) -> int:
        return self.length


class ExternalBar(Iterable):
    def __init__(self, iterable: Iterable = None, callback: Callable[[int], None] = None, total: int = 0) -> None:
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


def gmap(fun: Callable[[Iteration, Any, ...], Result], iterable: Iterable[Iteration] = None,
         args: tuple[Any, ...] = None, kwargs: dict[str, Any] = None, total: int = None, desc: str = None,
         bar: Bar | bool = True, terminator: Callable[[], None] = None, serial: bool = None, length: int = None,
         n_processes: int = None, yield_ordered: bool = True, yield_index: bool = False,
         **bar_kwargs: Any) -> Generator[Result, None, None] | list[Result]:
    """ map a function fun to each iteration in iterable
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
            length: deprecated alias for total
            n_processes: number of processes to use,
                the parallel pool will be restarted if the current pool does not have the right number of processes
            yield_ordered: return the result in the same order as the iterable
            yield_index: return the index of the result too
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
    if total is None and length is not None:
        total = length
        warn('parfor: use of \'length\' is deprecated, use \'total\' instead', DeprecationWarning, stacklevel=2)
        warn('parfor: use of \'length\' is deprecated, use \'total\' instead', DeprecationWarning, stacklevel=3)
    if terminator is not None:
        warn('parfor: use of \'terminator\' is deprecated, workers are terminated automatically',
             DeprecationWarning, stacklevel=2)
        warn('parfor: use of \'terminator\' is deprecated, workers are terminated automatically',
             DeprecationWarning, stacklevel=3)
    is_chunked = isinstance(iterable, Chunks)
    if is_chunked:
        chunk_fun = fun
    else:
        iterable = Chunks(iterable, ratio=5, length=total)

        @wraps(fun)
        def chunk_fun(iterable: Iterable, *args: Any, **kwargs: Any) -> list[Result]:  # noqa
            return [fun(iteration, *args, **kwargs) for iteration in iterable]

    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    if 'total' not in bar_kwargs:
        bar_kwargs['total'] = sum(iterable.lengths)
    if 'desc' not in bar_kwargs:
        bar_kwargs['desc'] = desc
    if 'disable' not in bar_kwargs:
        bar_kwargs['disable'] = not bar
    if serial is True or (serial is None and len(iterable) < min(cpu_count, 4)) or Worker.nested:  # serial case

        def tqdm_chunks(chunks: Chunks, *args, **kwargs) -> Iterable[Iteration]:  # noqa
            with tqdm(*args, **kwargs) as b:
                for chunk, length in zip(chunks, chunks.lengths):  # noqa
                    yield chunk
                    b.update(length)

        iterable = (ExternalBar(iterable, bar, sum(iterable.lengths)) if callable(bar)
                    else tqdm_chunks(iterable, **bar_kwargs))
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

    else:   # parallel case
        with ExitStack() as stack:
            if callable(bar):
                bar = stack.enter_context(ExternalBar(callback=bar))
            else:
                bar = stack.enter_context(tqdm(**bar_kwargs))
            with ParPool(chunk_fun, args, kwargs, n_processes, bar) as p:  # type: ignore
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


@wraps(gmap)
def pmap(*args, **kwargs) -> list[Result]:
    return list(gmap(*args, **kwargs))


@wraps(gmap)
def parfor(*args: Any, **kwargs: Any) -> Callable[[Callable[[Iteration, Any, ...], Result]], list[Result]]:
    def decfun(fun: Callable[[Iteration, Any, ...], Result]) -> list[Result]:
        return pmap(fun, *args, **kwargs)
    return decfun
