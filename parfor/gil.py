from __future__ import annotations

import asyncio
import multiprocessing
from collections import UserDict
from contextlib import redirect_stderr, redirect_stdout
from os import devnull, getpid
from time import time
from traceback import format_exc
from typing import Any, Callable, Hashable, NoReturn, Optional
from warnings import warn

from .common import Bar, cpu_count
from .pickler import dumps, loads


class SharedMemory(UserDict):
    def __init__(self, manager: multiprocessing.Manager) -> None:
        super().__init__()
        self.data = manager.dict()  # item_id: dilled representation of object
        self.references = manager.dict()  # item_id: counter
        self.references_lock = manager.Lock()
        self.cache = {}  # item_id: object
        self.trash_can = {}
        self.pool_ids = {}  # item_id: {(pool_id, task_handle), ...}

    def __getstate__(self) -> tuple[dict[int, bytes], dict[int, int], multiprocessing.Lock]:
        return self.data, self.references, self.references_lock

    def __setitem__(self, item_id: int, value: Any) -> None:
        if item_id not in self:  # values will not be changed
            try:
                self.data[item_id] = False, value
            except Exception:  # only use our pickler when necessary # noqa
                self.data[item_id] = True, dumps(value, recurse=True)
            with self.references_lock:
                try:
                    self.references[item_id] += 1
                except KeyError:
                    self.references[item_id] = 1
            self.cache[item_id] = value  # the id of the object will not be reused as long as the object exists

    def add_item(self, item: Any, pool_id: int, task_handle: Hashable) -> int:
        item_id = id(item)
        self[item_id] = item
        if item_id in self.pool_ids:
            self.pool_ids[item_id].add((pool_id, task_handle))
        else:
            self.pool_ids[item_id] = {(pool_id, task_handle)}
        return item_id

    def remove_pool(self, pool_id: int) -> None:
        """ remove objects used by a pool that won't be needed anymore """
        self.pool_ids = {key: v for key, value in self.pool_ids.items() if (v := {i for i in value if i[0] != pool_id})}
        for item_id in set(self.data.keys()) - set(self.pool_ids):
            del self[item_id]
        self.garbage_collect()

    def remove_task(self, pool_id: int, task: Task) -> None:
        """ remove objects used by a task that won't be needed anymore """
        self.pool_ids = {key: v for key, value in self.pool_ids.items() if (v := value - {(pool_id, task.handle)})}
        for item_id in {task.fun, *task.args, *task.kwargs} - set(self.pool_ids):
            del self[item_id]
        self.garbage_collect()

    # worker functions
    def __setstate__(self, state: dict) -> None:
        self.data, self.references, self.references_lock = state
        self.cache = {}
        self.trash_can = None

    def __getitem__(self, item_id: int) -> Any:
        if item_id not in self.cache:
            dilled, value = self.data[item_id]
            if dilled:
                value = loads(value)
            with self.references_lock:
                if item_id in self.references:
                    self.references[item_id] += 1
                else:
                    self.references[item_id] = 1
            self.cache[item_id] = value
        return self.cache[item_id]

    def garbage_collect(self) -> None:
        """ clean up the cache """
        for item_id in set(self.cache) - set(self.data.keys()):
            with self.references_lock:
                try:
                    self.references[item_id] -= 1
                except KeyError:
                    self.references[item_id] = 0
            if self.trash_can is not None and item_id not in self.trash_can:
                self.trash_can[item_id] = self.cache[item_id]
            del self.cache[item_id]

        if self.trash_can:
            for item_id in set(self.trash_can):
                if self.references[item_id] == 0:
                    # make sure every process removed the object before removing it in the parent
                    del self.references[item_id]
                    del self.trash_can[item_id]


class Task:
    def __init__(self, shared_memory: SharedMemory, pool_id: int, handle: Hashable, fun: Callable[[Any, ...], Any],
                 args: tuple[Any, ...] = (), kwargs: dict[str, Any] = None) -> None:
        self.pool_id = pool_id
        self.handle = handle
        self.fun = shared_memory.add_item(fun, pool_id, handle)
        self.args = [shared_memory.add_item(arg, pool_id, handle) for arg in args]
        self.kwargs = [] if kwargs is None else [shared_memory.add_item(item, pool_id, handle)
                                                 for item in kwargs.items()]
        self.name = fun.__name__ if hasattr(fun, '__name__') else None
        self.done = False
        self.result = None
        self.pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__
        if self.result is not None:
            state['result'] = dumps(self.result, recurse=True)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update({key: value for key, value in state.items() if key != 'result'})
        if state['result'] is None:
            self.result = None
        else:
            self.result = loads(state['result'])

    def __call__(self, shared_memory: SharedMemory) -> Task:
        if not self.done:
            fun = shared_memory[self.fun] or (lambda *args, **kwargs: None)  # noqa
            args = [shared_memory[arg] for arg in self.args]
            kwargs = dict([shared_memory[kwarg] for kwarg in self.kwargs])
            self.result = fun(*args, **kwargs)  # noqa
            self.done = True
        return self

    def __repr__(self) -> str:
        if self.done:
            return f'Task {self.handle}, result: {self.result}'
        else:
            return f'Task {self.handle}'


class Context(multiprocessing.context.SpawnContext):
    """ Provide a context where child processes never are daemonic. """
    class Process(multiprocessing.context.SpawnProcess):
        @property
        def daemon(self) -> bool:
            return False

        @daemon.setter
        def daemon(self, value: bool) -> None:
            pass


class ParPool:
    """ Parallel processing with addition of iterations at any time and request of that result any time after that.
        The target function and its argument can be changed at any time.
    """
    def __init__(self, fun: Callable[[Any, ...], Any] = None,
                 args: tuple[Any] = None, kwargs: dict[str, Any] = None, n_processes: int = None, bar: Bar = None):
        self.id = id(self)
        self.handle = 0
        self.tasks = {}
        self.bar = bar
        self.bar_lengths = {}
        self.spool = PoolSingleton(n_processes, self)
        self.manager = self.spool.manager
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self.is_started = False

    def __getstate__(self) -> NoReturn:
        raise RuntimeError(f'Cannot pickle {self.__class__.__name__} object.')

    def __enter__(self) -> ParPool:
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.close()

    def close(self) -> None:
        self.spool.remove_pool(self.id)

    def __call__(self, n: Any, handle: Hashable = None, barlength: int = 1) -> None:
        self.add_task(args=(n, *(() if self.args is None else self.args)), handle=handle, barlength=barlength)

    def add_task(self, fun: Callable[[Any, ...], Any] = None, args: tuple[Any, ...] = None,
                 kwargs: dict[str, Any] = None, handle: Hashable = None, barlength: int = 1) -> Optional[int]:
        if self.id not in self.spool.pools:
            raise ValueError(f'this pool is not registered (anymore) with the pool singleton')
        if handle is None:
            new_handle = self.handle
            self.handle += 1
        else:
            new_handle = handle
        if new_handle in self:
            raise ValueError(f'handle {new_handle} already present')
        task = Task(self.spool.shared_memory, self.id, new_handle,
                    fun or self.fun, args or self.args, kwargs or self.kwargs)
        self.tasks[new_handle] = task
        self.spool.add_task(task)
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
            if not self.spool.get_from_queue() and not self.tasks[handle].done and self.is_started \
                    and not self.working:
                for _ in range(10):  # wait some time while processing possible new messages
                    self.spool.get_from_queue()
                if not self.spool.get_from_queue() and not self.tasks[handle].done and self.is_started \
                        and not self.working:
                    # retry a task if the process was killed while working on a task
                    self.spool.add_task(self.tasks[handle])
                    warn(f'Task {handle} was restarted because the process working on it was probably killed.')
        result = self.tasks[handle].result
        self.tasks.pop(handle)
        return result

    def __contains__(self, handle: Hashable) -> bool:
        return handle in self.tasks

    def __delitem__(self, handle: Hashable) -> None:
        self.tasks.pop(handle)

    def get_newest(self) -> Any:
        return self.spool.get_newest_for_pool(self)

    def process_queue(self) -> None:
        self.spool.process_queue()

    def task_error(self, handle: Hashable, error: Exception) -> None:
        if handle in self:
            task = self.tasks[handle]
            print(f'Error from process working on iteration {handle}:\n')
            print(error)
            print('Retrying in main process...')
            task(self.spool.shared_memory)
            self.spool.shared_memory.remove_task(self.id, task)
            raise Exception(f'Function \'{task.name}\' cannot be executed by parfor, amend or execute in serial.')

    def done(self, task: Task) -> None:
        if task.handle in self:  # if not, the task was restarted erroneously
            self.tasks[task.handle] = task
            if hasattr(self.bar, 'update'):
                self.bar.update(self.bar_lengths.pop(task.handle))
        self.spool.shared_memory.remove_task(self.id, task)

    def started(self, handle: Hashable, pid: int) -> None:
        self.is_started = True
        if handle in self:  # if not, the task was restarted erroneously
            self.tasks[handle].pid = pid

    @property
    def working(self) -> bool:
        return not all([task.pid is None for task in self.tasks.values()])


class PoolSingleton:
    """ There can be only one pool at a time, but the pool can be restarted by calling close() and then constructing a
        new pool. The pool will close itself after 10 minutes of idle time. """

    instance = None

    def __new__(cls, n_processes: int = None, *args: Any, **kwargs: Any) -> PoolSingleton:
        # restart if any workers have shut down or if we want to have a different number of processes
        if cls.instance is not None:
            if (cls.instance.n_workers.value < cls.instance.n_processes or
                    cls.instance.n_processes != (n_processes or cpu_count)):
                cls.instance.close()
        if cls.instance is None or not cls.instance.is_alive:
            new = super().__new__(cls)
            new.n_processes = n_processes or cpu_count
            new.instance = new
            new.is_started = False
            ctx = Context()
            new.n_workers = ctx.Value('i', new.n_processes)
            new.event = ctx.Event()
            new.queue_in = ctx.Queue(3 * new.n_processes)
            new.queue_out = ctx.Queue(new.n_processes)
            new.manager = ctx.Manager()
            new.shared_memory = SharedMemory(new.manager)
            new.pool = ctx.Pool(new.n_processes,
                                Worker(new.shared_memory, new.queue_in, new.queue_out, new.n_workers, new.event))
            new.is_alive = True
            new.handle = 0
            new.pools = {}
            new.time_out = None
            cls.instance = new
        return cls.instance

    def __init__(self, n_processes: int = None, parpool: Parpool = None) -> None:  # noqa
        if parpool is not None:
            self.pools[parpool.id] = parpool
            if self.time_out is not None:
                self.time_out.cancel()
            self.time_out = None

    def __getstate__(self) -> NoReturn:
        raise RuntimeError(f'Cannot pickle {self.__class__.__name__} object.')

    def remove_pool(self, pool_id: int) -> None:
        self.shared_memory.remove_pool(pool_id)
        if pool_id in self.pools:
            self.pools.pop(pool_id)
        if len(self.pools) == 0:
            self.time_out = asyncio.get_event_loop().call_later(600, self.close)  # noqa

    def error(self, error: Exception) -> NoReturn:
        self.close()
        raise Exception(f'Error occurred in worker: {error}')

    def process_queue(self) -> None:
        while self.get_from_queue():
            pass

    def get_from_queue(self) -> bool:
        """ Get an item from the queue and store it, return True if more messages are waiting. """
        try:
            code, pool_id, *args = self.queue_out.get(True, 0.02)
            if pool_id is None:
                getattr(self, code)(*args)
            elif pool_id in self.pools:
                getattr(self.pools[pool_id], code)(*args)
            return True
        except multiprocessing.queues.Empty:  # noqa
            for pool in self.pools.values():
                for handle, task in pool.tasks.items():  # retry a task if the process doing it was killed
                    if task.pid is not None \
                            and task.pid not in [child.pid for child in multiprocessing.active_children()]:
                        self.queue_in.put(task)
                        warn(f'Task {task.handle} was restarted because process {task.pid} was probably killed.')
            return False

    def add_task(self, task: Task) -> None:
        """ Add new iteration, using optional manually defined handle."""
        if self.is_alive and not self.event.is_set():
            while self.queue_in.full():
                self.get_from_queue()
            self.queue_in.put(task)
        self.shared_memory.garbage_collect()

    def get_newest_for_pool(self, pool: ParPool) -> tuple[Hashable, Any]:
        """ Request the newest key and result and delete its record. Wait if result not yet available. """
        while len(pool.tasks):
            self.get_from_queue()
            for task in pool.tasks.values():
                if task.done:
                    handle, result = task.handle, task.result
                    pool.tasks.pop(handle)
                    return handle, result

    @classmethod
    def close(cls) -> None:
        if cls.instance is not None:
            instance = cls.instance
            cls.instance = None
            if instance.time_out is not None:
                instance.time_out.cancel()

            def empty_queue(queue):
                try:
                    if not queue._closed:  # noqa
                        while not queue.empty():
                            try:
                                queue.get(True, 0.02)
                            except multiprocessing.queues.Empty:  # noqa
                                pass
                except OSError:
                    pass

            def close_queue(queue: multiprocessing.queues.Queue) -> None:
                empty_queue(queue)  # noqa
                if not queue._closed:  # noqa
                    queue.close()
                queue.join_thread()

            if instance.is_alive:
                instance.is_alive = False
                instance.event.set()
                instance.pool.close()
                t = time()
                while instance.n_workers.value:
                    empty_queue(instance.queue_in)
                    empty_queue(instance.queue_out)
                    if time() - t > 10:
                        warn(f'Parfor: Closing pool timed out, {instance.n_workers.value} processes still alive.')
                        instance.pool.terminate()
                        break
                empty_queue(instance.queue_in)
                empty_queue(instance.queue_out)
                instance.pool.join()
                close_queue(instance.queue_in)
                close_queue(instance.queue_out)
                instance.manager.shutdown()
                instance.handle = 0


class Worker:
    """ Manages executing the target function which will be executed in different processes. """
    nested = False

    def __init__(self, shared_memory: SharedMemory, queue_in: multiprocessing.queues.Queue,
                 queue_out: multiprocessing.queues.Queue, n_workers: multiprocessing.Value,
                 event: multiprocessing.Event) -> None:
        self.shared_memory = shared_memory
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.n_workers = n_workers
        self.event = event

    def add_to_queue(self, *args: Any) -> None:
        while not self.event.is_set():
            try:
                self.queue_out.put(args, timeout=0.1)
                break
            except multiprocessing.queues.Full:  # noqa
                continue

    def __call__(self) -> None:
        Worker.nested = True
        pid = getpid()
        last_active_time = time()
        while not self.event.is_set() and time() - last_active_time < 600:
            try:
                task = self.queue_in.get(True, 0.02)
                try:
                    self.add_to_queue('started', task.pool_id, task.handle, pid)
                    with redirect_stdout(open(devnull, 'w')), redirect_stderr(open(devnull, 'w')):
                        self.add_to_queue('done', task.pool_id, task(self.shared_memory))
                except Exception:  # noqa
                    self.add_to_queue('task_error', task.pool_id, task.handle, format_exc())
                    self.event.set()
                self.shared_memory.garbage_collect()
                last_active_time = time()
            except (multiprocessing.queues.Empty, KeyboardInterrupt):  # noqa
                pass
            except Exception:  # noqa
                self.add_to_queue('error', None, format_exc())
                self.event.set()
                self.shared_memory.garbage_collect()
        for child in multiprocessing.active_children():
            child.kill()
        with self.n_workers:
            self.n_workers.value -= 1
