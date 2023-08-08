import pytest
from parfor import Chunks, parfor, Parpool, pmap


class SequenceIterator:
    def __init__(self, sequence):
        self._sequence = sequence
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._sequence):
            item = self._sequence[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __len__(self):
        return len(self._sequence)


class Iterable:
    def __init__(self, sequence):
        self.sequence = sequence

    def __iter__(self):
        return SequenceIterator(self.sequence)


def iterators():
    yield range(10), None
    yield list(range(10)), None
    yield (i for i in range(10)), 10
    yield SequenceIterator(range(10)), None
    yield Iterable(range(10)), 10


@pytest.mark.parametrize('iterator', iterators())
def test_chunks(iterator):
    chunks = Chunks(iterator[0], size=2, length=iterator[1])
    assert list(chunks) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def test_parpool():
    def fun(i, j, k):
        return i * j * k

    with Parpool(fun, (3,), {'k': 2}) as pool:
        for i in range(10):
            pool[i] = i

        assert [pool[i] for i in range(10)] == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]


def test_parfor():
    @parfor(range(10), (3,), {'k': 2})
    def fun(i, j, k):
        return i * j * k

    assert fun == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]


def test_pmap():
    def fun(i, j, k):
        return i * j * k

    assert pmap(fun, range(10), (3,), {'k': 2}) == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]