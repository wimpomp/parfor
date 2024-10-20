from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Hashable, NoReturn, Optional

from .common import Bar, cpu_count


class Worker:
    nested = False

    def __init__(self, *args, **kwargs):
        pass


class PoolSingleton:
    def __init__(self, *args, **kwargs):
        pass

    def close(self):
        pass


class Task:
    def __init__(self, queue: queue.Queue, handle: Hashable, fun: Callable[[Any, ...], Any],  # noqa
                 args: tuple[Any, ...] = (), kwargs: dict[str, Any] = None) -> None:
        self.queue = queue
        self.handle = handle
        self.fun = fun
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.name = fun.__name__ if hasattr(fun, '__name__') else None
        self.started = False
        self.done = False
        self.result = None

    def __call__(self):
        if not self.done:
            self.result = self.fun(*self.args, **self.kwargs)
            try:
                self.queue.put(self.handle)
            except queue.ShutDown:
                pass

    def __repr__(self) -> str:
        if self.done:
            return f'Task {self.handle}, result: {self.result}'
        else:
            return f'Task {self.handle}'


class ParPool:
    """ Parallel processing with addition of iterations at any time and request of that result any time after that.
        The target function and its argument can be changed at any time.
    """
    def __init__(self, fun: Callable[[Any, ...], Any] = None,
                 args: tuple[Any] = None, kwargs: dict[str, Any] = None, n_processes: int = None, bar: Bar = None):
        self.queue = queue.Queue()
        self.handle = 0
        self.tasks = {}
        self.bar = bar
        self.bar_lengths = {}
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self.n_processes = n_processes or cpu_count
        self.threads = {}

    def __getstate__(self) -> NoReturn:
        raise RuntimeError(f'Cannot pickle {self.__class__.__name__} object.')

    def __enter__(self) -> ParPool:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.close()

    def close(self) -> None:
        self.queue.shutdown()  # noqa python3.13
        for thread in self.threads.values():
            thread.join()

    def __call__(self, n: Any, handle: Hashable = None, barlength: int = 1) -> None:
        self.add_task(args=(n, *(() if self.args is None else self.args)), handle=handle, barlength=barlength)

    def add_task(self, fun: Callable[[Any, ...], Any] = None, args: tuple[Any, ...] = None,
                 kwargs: dict[str, Any] = None, handle: Hashable = None, barlength: int = 1) -> Optional[int]:
        if handle is None:
            new_handle = self.handle
            self.handle += 1
        else:
            new_handle = handle
        if new_handle in self:
            raise ValueError(f'handle {new_handle} already present')
        task = Task(self.queue, new_handle, fun or self.fun, args or self.args, kwargs or self.kwargs)
        while len(self.threads) > self.n_processes:
            self.get_from_queue()
        thread = threading.Thread(target=task)
        thread.start()
        self.threads[new_handle] = thread
        self.tasks[new_handle] = task
        self.bar_lengths[new_handle] = barlength
        if handle is None:
            return new_handle

    def __setitem__(self, handle: Hashable, n: Any) -> None:
        """ Add new iteration. """
        self(n, handle=handle)

    def __getitem__(self, handle: Hashable) -> Any:
        """ Request result and delete its record. Wait if result not yet available. """
        if handle not in self:
            raise ValueError(f'No handle: {handle} in pool')
        while not self.tasks[handle].done:
            self.get_from_queue()
        task = self.tasks.pop(handle)
        return task.result

    def __contains__(self, handle: Hashable) -> bool:
        return handle in self.tasks

    def __delitem__(self, handle: Hashable) -> None:
        self.tasks.pop(handle)

    def get_from_queue(self) -> bool:
        """ Get an item from the queue and store it, return True if more messages are waiting. """
        try:
            handle = self.queue.get(True, 0.02)
            self.done(handle)
            return True
        except (queue.Empty, queue.ShutDown):
            return False

    def get_newest(self) -> Any:
        """ Request the newest key and result and delete its record. Wait if result not yet available. """
        while len(self.tasks):
            self.get_from_queue()
            for task in self.tasks.values():
                if task.done:
                    handle, result = task.handle, task.result
                    self.tasks.pop(handle)
                    return handle, result

    def process_queue(self) -> None:
        while self.get_from_queue():
            pass

    def done(self, handle: Hashable) -> None:
        thread = self.threads.pop(handle)
        thread.join()
        task = self.tasks[handle]
        task.done = True
        if hasattr(self.bar, 'update'):
            self.bar.update(self.bar_lengths.pop(handle))
