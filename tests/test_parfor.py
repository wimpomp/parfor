from __future__ import annotations

import sys
from dataclasses import dataclass
from os import getpid
from time import sleep
from typing import Any, Iterator, Optional, Sequence

import pytest

from parfor import Chunks, ParPool, parfor, pmap

try:
    if sys._is_gil_enabled():  # noqa
        gil = True
    else:
        gil = False
except Exception:  # noqa
    gil = True


class SequenceIterator:
    def __init__(self, sequence: Sequence) -> None:
        self._sequence = sequence
        self._index = 0

    def __iter__(self) -> SequenceIterator:
        return self

    def __next__(self) -> Any:
        if self._index < len(self._sequence):
            item = self._sequence[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __len__(self) -> int:
        return len(self._sequence)


class Iterable:
    def __init__(self, sequence: Sequence) -> None:
        self.sequence = sequence

    def __iter__(self) -> SequenceIterator:
        return SequenceIterator(self.sequence)


def iterators() -> tuple[Iterator, Optional[int]]:
    yield range(10), None
    yield list(range(10)), None
    yield (i for i in range(10)), 10
    yield SequenceIterator(range(10)), None
    yield Iterable(range(10)), 10


@pytest.mark.parametrize('iterator', iterators())
def test_chunks(iterator: tuple[Iterator, Optional[int]]) -> None:
    chunks = Chunks(iterator[0], size=2, length=iterator[1])
    assert list(chunks) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def test_parpool() -> None:
    def fun(i, j, k) -> int:  # noqa
        return i * j * k

    with ParPool(fun, (3,), {'k': 2}) as pool:  # noqa
        for i in range(10):
            pool[i] = i

        assert [pool[i] for i in range(10)] == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]


def test_parfor() -> None:
    @parfor(range(10), (3,), {'k': 2})
    def fun(i, j, k):
        return i * j * k

    assert fun == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]


@pytest.mark.parametrize('serial', (True, False))
def test_pmap(serial) -> None:
    def fun(i, j, k):
        return i * j * k

    assert pmap(fun, range(10), (3,), {'k': 2}, serial=serial) == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]


@pytest.mark.parametrize('serial', (True, False))
def test_pmap_with_idx(serial) -> None:
    def fun(i, j, k):
        return i * j * k

    assert (pmap(fun, range(10), (3,), {'k': 2}, serial=serial, yield_index=True) ==
            [(0, 0), (1, 6), (2, 12), (3, 18), (4, 24), (5, 30), (6, 36), (7, 42), (8, 48), (9, 54)])


@pytest.mark.parametrize('serial', (True, False))
def test_pmap_chunks(serial) -> None:
    def fun(i, j, k):
        return [i_ * j * k for i_ in i]

    chunks = Chunks(range(10), size=2)
    assert pmap(fun, chunks, (3,), {'k': 2}, serial=serial) == [[0, 6], [12, 18], [24, 30], [36, 42], [48, 54]]


@pytest.mark.skipif(not gil, reason='test if gil enabled only')
def test_id_reuse() -> None:
    def fun(i):
        return i[0].a

    @dataclass
    class T:
        a: int = 3

    def gen(total):
        for i in range(total):
            t = T(i)
            yield t
            del t

    a = pmap(fun, Chunks(gen(1000), size=1, length=1000), total=1000)  # noqa
    assert all([i == j for i, j in enumerate(a)])


@pytest.mark.skipif(not gil, reason='test if gil enabled only')
@pytest.mark.parametrize('n_processes', (2, 4, 6))
def test_n_processes(n_processes) -> None:

    @parfor(range(12), n_processes=n_processes)
    def fun(i):  # noqa
        sleep(0.25)
        return getpid()

    assert len(set(fun)) == n_processes
