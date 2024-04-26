from __future__ import annotations

import multiprocessing
from collections import UserDict
from contextlib import ExitStack
from functools import wraps
from os import getpid
from time import time
from traceback import format_exc
from typing import Any, Callable, Hashable, Iterable, Iterator, NoReturn, Optional, Protocol, Sized, TypeVar
from warnings import warn

from tqdm.auto import tqdm

from .pickler import dumps, loads

cpu_count = int(multiprocessing.cpu_count())


Result = TypeVar('Result')
Iteration = TypeVar('Iteration')
Arg = TypeVar('Arg')
Return = TypeVar('Return')


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
        self.length = max(1, min(length, number))
        self.lengths = [((i + 1) * self.number_of_items // self.length) - (i * self.number_of_items // self.length)
                        for i in range(self.length)]

    def __iter__(self) -> Iterator[Any]:
        for i in range(self.length):
            p, q = (i * self.number_of_items // self.length), ((i + 1) * self.number_of_items // self.length)
            if len(self.iterators) == 1:
                yield [next(self.iterators[0]) for _ in range(q - p)]
            else:
                yield [[next(iterator) for _ in range(q-p)] for iterator in self.iterators]

    def __len__(self) -> int:
        return self.length


class Bar(Protocol):
    def update(self, n: int = 1) -> None: ...


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
                 args: tuple[Any] = None, kwargs: dict[str, Any] = None, bar: Bar = None):
        self.id = id(self)
        self.handle = 0
        self.tasks = {}
        self.bar = bar
        self.bar_lengths = {}
        self.spool = PoolSingleton(self)
        self.manager = self.spool.manager
        self.fun = fun
        self.args = args
        self.kwargs = kwargs
        self.is_started = False
        self.last_task = None

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
        self.last_task = task
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
            print('Retrying in main thread...')
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

    def __new__(cls, *args: Any, **kwargs: Any) -> PoolSingleton:
        if cls.instance is not None:  # restart if any workers have shut down
            if cls.instance.n_workers.value < cls.instance.n_processes:
                cls.instance.close()
        if cls.instance is None or not cls.instance.is_alive:
            new = super().__new__(cls)
            new.n_processes = cpu_count
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
            cls.instance = new
        return cls.instance

    def __init__(self, parpool: Parpool = None) -> None:  # noqa
        if parpool is not None:
            self.pools[parpool.id] = parpool

    def __getstate__(self) -> NoReturn:
        raise RuntimeError(f'Cannot pickle {self.__class__.__name__} object.')

    # def __del__(self):
    #     self.close()

    def remove_pool(self, pool_id: int) -> None:
        self.shared_memory.remove_pool(pool_id)
        if pool_id in self.pools:
            self.pools.pop(pool_id)

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
        if hasattr(cls, 'instance') and cls.instance is not None:
            instance = cls.instance
            cls.instance = None

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
                    self.add_to_queue('done', task.pool_id, task(self.shared_memory))
                except Exception:  # noqa
                    self.add_to_queue('task_error', task.pool_id, task.handle, format_exc())
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


def pmap(fun: Callable[[Iteration, Any, ...], Result], iterable: Iterable[Iteration] = None,
         args: tuple[Any, ...] = None, kwargs: dict[str, Any] = None, total: int = None, desc: str = None,
         bar: Bar | bool = True, terminator: Callable[[], None] = None, serial: bool = None, length: int = None,
         **bar_kwargs: Any) -> list[Result]:
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
            **bar_kwargs: keywords arguments for tqdm.tqdm

        output:
            list with results from applying the function \'fun\' to each iteration of the iterable / iterator

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

    args = args or ()
    kwargs = kwargs or {}

    if 'total' not in bar_kwargs:
        bar_kwargs['total'] = sum(iterable.lengths)
    if 'desc' not in bar_kwargs:
        bar_kwargs['desc'] = desc
    if 'disable' not in bar_kwargs:
        bar_kwargs['disable'] = not bar
    if serial is True or (serial is None and len(iterable) < min(cpu_count, 4)) or Worker.nested:  # serial case
        if callable(bar):
            return sum([chunk_fun(c, *args, **kwargs) for c in ExternalBar(iterable, bar)], [])
        else:
            return sum([chunk_fun(c, *args, **kwargs) for c in tqdm(iterable, **bar_kwargs)], [])
    else:   # parallel case
        with ExitStack() as stack:
            if callable(bar):
                bar = stack.enter_context(ExternalBar(callback=bar))
            else:
                bar = stack.enter_context(tqdm(**bar_kwargs))
            with ParPool(chunk_fun, args, kwargs, bar) as p:
                for i, (j, l) in enumerate(zip(iterable, iterable.lengths)):  # add work to the queue
                    p(j, handle=i, barlength=iterable.lengths[i])
                    if bar.total is None or bar.total < i+1:
                        bar.total = i+1
                if is_chunked:
                    return [p[i] for i in range(len(iterable))]
                else:
                    return sum([p[i] for i in range(len(iterable))], [])  # collect the results


@wraps(pmap)
def parfor(*args: Any, **kwargs: Any) -> Callable[[Callable[[Iteration, Any, ...], Result]], list[Result]]:
    def decfun(fun: Callable[[Iteration, Any, ...], Result]) -> list[Result]:
        return pmap(fun, *args, **kwargs)
    return decfun
