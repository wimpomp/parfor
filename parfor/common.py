from __future__ import annotations

import os
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

cpu_count = int(os.cpu_count())


class Bar(Protocol):
    def update(self, n: int = 1) -> None: ...


class SharedArray(np.ndarray):
    """ Numpy array whose memory can be shared between processes, so that memory use is reduced and changes in one
        process are reflected in all other processes. Changes are not atomic, so protect changes with a lock to prevent
        race conditions!
    """

    def __new__(cls, shape: int | Sequence[int], dtype: DTypeLike = float, shm: str | SharedMemory = None,
                offset: int = 0, strides: tuple[int, int] = None, order: str = None) -> SharedArray:
        if isinstance(shm, str):
            shm = SharedMemory(shm)
        elif shm is None:
            shm = SharedMemory(create=True, size=np.prod(shape) * np.dtype(dtype).itemsize)
        new = super().__new__(cls, shape, dtype, shm.buf, offset, strides, order)
        new.shm = shm
        return new

    def __reduce__(self) -> tuple[Callable[[int | Sequence[int], DTypeLike, str], SharedArray],
                                  tuple[int | tuple[int, ...], np.dtype, str]]:
        return self.__class__, (self.shape, self.dtype, self.shm.name)

    def __enter__(self) -> SharedArray:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self, 'shm'):
            self.shm.close()
            self.shm.unlink()

    def __del__(self) -> None:
        if hasattr(self, 'shm'):
            self.shm.close()

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        if isinstance(obj, np.ndarray) and not isinstance(obj, SharedArray):
            raise TypeError('view casting to SharedArray is not implemented because right now we need to make a copy')

    @classmethod
    def from_array(cls, array: ArrayLike) -> SharedArray:
        """ copy existing array into a SharedArray """
        new = cls(array.shape, array.dtype)
        new[:] = array[:]
        return new
