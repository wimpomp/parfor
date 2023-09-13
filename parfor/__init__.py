import multiprocessing
from collections import OrderedDict
from contextlib import ExitStack
from copy import copy
from functools import wraps
from os import getpid
from traceback import format_exc
from warnings import warn

from tqdm.auto import tqdm

from .pickler import Pickler, dumps, loads

cpu_count = int(multiprocessing.cpu_count())


class Chunks:
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

    def __init__(self, *iterators, size=None, number=None, ratio=None, length=None, s=None, n=None, r=None):
        # s, r and n are deprecated
        if s is not None:
            warn('parfor: use of \'s\' is deprecated, use \'size\' instead', DeprecationWarning, stacklevel=2)
            warn('parfor: use of \'s\' is deprecated, use \'size\' instead', DeprecationWarning, stacklevel=3)
            size = s
        if n is not None:
            warn('parfor: use of \'n\' is deprecated, use \'number\' instead', DeprecationWarning, stacklevel=2)
            warn('parfor: use of \'n\' is deprecated, use \'number\' instead', DeprecationWarning, stacklevel=3)
            number = n
        if r is not None:
            warn('parfor: use of \'r\' is deprecated, use \'ratio\' instead', DeprecationWarning, stacklevel=2)
            warn('parfor: use of \'r\' is deprecated, use \'ratio\' instead', DeprecationWarning, stacklevel=3)
            ratio = r
        if length is None:
            try:
                length = min(*[len(iterator) for iterator in iterators]) if len(iterators) > 1 else len(iterators[0])
            except TypeError:
                raise TypeError('Cannot determine the length of the iterator(s), so the length must be provided as an'
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
        self.iterators = [iter(arg) for arg in iterators]
        self.number_of_items = length
        self.length = max(1, min(length, number))
        self.lengths = [((i + 1) * self.number_of_items // self.length) - (i * self.number_of_items // self.length)
                        for i in range(self.length)]

    def __iter__(self):
        for i in range(self.length):
            p, q = (i * self.number_of_items // self.length), ((i + 1) * self.number_of_items // self.length)
            if len(self.iterators) == 1:
                yield [next(self.iterators[0]) for _ in range(q - p)]
            else:
                yield [[next(iterator) for _ in range(q-p)] for iterator in self.iterators]

    def __len__(self):
        return self.length


class ExternalBar:
    def __init__(self, iterable=None, callback=None, total=0):
        self.iterable = iterable
        self.callback = callback
        self.total = total
        self._n = 0

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return

    def __iter__(self):
        for n, item in enumerate(self.iterable):
            yield item
            self.n = n + 1

    def update(self, n=1):
        self.n += n

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        if n != self._n:
            self._n = n
            if self.callback is not None:
                self.callback(n)


class Hasher:
    def __init__(self, obj, hsh=None):
        if hsh is not None:
            self.obj, self.str, self.hash = None, obj, hsh
        elif isinstance(obj, Hasher):
            self.obj, self.str, self.hash = obj.obj, obj.str, obj.hash
        else:
            self.obj = obj
            self.str = dumps(self.obj, recurse=True)
            self.hash = hash(self.str)

    def __reduce__(self):
        return self.__class__, (self.str, self.hash)

    def set_from_cache(self, cache=None):
        if cache is None:
            self.obj = loads(self.str)
        elif self.hash in cache:
            self.obj = cache[self.hash]
        else:
            self.obj = cache[self.hash] = loads(self.str)


class HashDescriptor:
    def __set_name__(self, owner, name):
        self.owner, self.name = owner, '_' + name

    def __set__(self, instance, value):
        if isinstance(value, Hasher):
            setattr(instance, self.name, value)
        else:
            setattr(instance, self.name, Hasher(value))

    def __get__(self, instance, owner):
        return getattr(instance, self.name).obj


class DequeDict(OrderedDict):
    def __init__(self, maxlen=None, *args, **kwargs):
        self.maxlen = maxlen
        super().__init__(*args, **kwargs)

    def __truncate__(self):
        while len(self) > self.maxlen:
            self.popitem(False)

    def __setitem__(self, *args, **kwargs):
        super().__setitem__(*args, **kwargs)
        self.__truncate__()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.__truncate__()


class Task:
    fun = HashDescriptor()
    args = HashDescriptor()
    kwargs = HashDescriptor()

    def __init__(self, pool_id, fun=None, args=None, kwargs=None, handle=None, n=None, done=False, result=None):
        self.pool_id = pool_id
        self.fun = fun or (lambda *args, **kwargs: None)
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.handle = handle
        self.n = n
        self.done = done
        self.result = loads(result) if self.done else None
        self.pid = None

    def __reduce__(self):
        if self.done:
            return self.__class__, (self.pool_id, None, None, None, self.handle, None, self.done,
                                    dumps(self.result, recurse=True))
        else:
            return self.__class__, (self.pool_id, self._fun, self._args, self._kwargs, self.handle,
                                    dumps(self.n, recurse=True), self.done)

    def set_from_cache(self, cache=None):
        self.n = loads(self.n)
        self._fun.set_from_cache(cache)
        self._args.set_from_cache(cache)
        self._kwargs.set_from_cache(cache)

    def __call__(self):
        if not self.done:
            self.result = self.fun(self.n, *self.args, **self.kwargs)
            self.fun, self.args, self.kwargs, self.done = None, None, None, True  # Remove potentially big things
        return self

    def __repr__(self):
        if self.done:
            return f'Task {self.handle}, result: {self.result}'
        else:
            return f'Task {self.handle}'


class Context(multiprocessing.context.SpawnContext):
    """ Provide a context where child processes never are daemonic. """
    class Process(multiprocessing.context.SpawnProcess):
        @property
        def daemon(self):
            return False

        @daemon.setter
        def daemon(self, value):
            pass


class ParPool:
    """ Parallel processing with addition of iterations at any time and request of that result any time after that.
        The target function and its argument can be changed at any time.
    """
    def __init__(self, fun=None, args=None, kwargs=None, bar=None):
        self.id = id(self)
        self.handle = 0
        self.tasks = {}
        self.last_task = Task(self.id, fun, args, kwargs)
        self.bar = bar
        self.bar_lengths = {}
        self.spool = PoolSingleton(self)
        self.manager = self.spool.manager
        self.is_started = False

    def __getstate__(self):
        raise RuntimeError(f'Cannot pickle {self.__class__.__name__} object.')

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        if self.id in self.spool.pools:
            self.spool.pools.pop(self.id)

    def __call__(self, n, fun=None, args=None, kwargs=None, handle=None, barlength=1):
        if self.id not in self.spool.pools:
            raise ValueError(f'this pool is not registered (anymore) with the pool singleton')
        if handle is None:
            new_handle = self.handle
            self.handle += 1
        else:
            new_handle = handle
        if new_handle in self:
            raise ValueError(f'handle {new_handle} already present')
        new_task = copy(self.last_task)
        if fun is not None:
            new_task.fun = fun
        if args is not None:
            new_task.args = args
        if kwargs is not None:
            new_task.kwargs = kwargs
        new_task.handle = new_handle
        new_task.n = n
        self.tasks[new_handle] = new_task
        self.last_task = new_task
        self.spool.add_task(new_task)
        self.bar_lengths[new_handle] = barlength
        if handle is None:
            return new_handle

    def __setitem__(self, handle, n):
        """ Add new iteration. """
        self(n, handle=handle)

    def __getitem__(self, handle):
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
                    # retry a task if the process was killed while retrieving the task
                    self.spool.add_task(self.tasks[handle])
                    warn(f'Task {handle} was restarted because the process retrieving it was probably killed.')
        result = self.tasks[handle].result
        self.tasks.pop(handle)
        return result

    def __contains__(self, handle):
        return handle in self.tasks

    def __delitem__(self, handle):
        self.tasks.pop(handle)

    def get_newest(self):
        return self.spool.get_newest_for_pool(self)

    def process_queue(self):
        self.spool.process_queue()

    def task_error(self, handle, error):
        if handle in self:
            task = self.tasks[handle]
            print(f'Error from process working on iteration {handle}:\n')
            print(error)
            self.close()
            print('Retrying in main thread...')
            fun = task.fun.__name__
            task()
            raise Exception('Function \'{}\' cannot be executed by parfor, amend or execute in serial.'.format(fun))

    def done(self, task):
        if task.handle in self:  # if not, the task was restarted erroneously
            self.tasks[task.handle] = task
            if hasattr(self.bar, 'update'):
                self.bar.update(self.bar_lengths.pop(task.handle))

    def started(self, handle, pid):
        self.is_started = True
        if handle in self:  # if not, the task was restarted erroneously
            self.tasks[handle].pid = pid

    @property
    def working(self):
        return not all([task.pid is None for task in self.tasks.values()])


class PoolSingleton:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance') or cls.instance is None or not cls.instance.is_alive:
            new = super().__new__(cls)
            new.n_processes = cpu_count
            new.instance = new
            new.is_started = False
            ctx = Context()
            new.n_workers = ctx.Value('i', new.n_processes)
            new.event = ctx.Event()
            new.queue_in = ctx.Queue(3 * new.n_processes)
            new.queue_out = ctx.Queue(new.n_processes)
            new.pool = ctx.Pool(new.n_processes, Worker(new.queue_in, new.queue_out, new.n_workers, new.event))
            new.is_alive = True
            new.handle = 0
            new.pools = {}
            new.manager = ctx.Manager()
            cls.instance = new
        return cls.instance

    def __init__(self, parpool=None):
        if parpool is not None:
            self.pools[parpool.id] = parpool

    def __getstate__(self):
        raise RuntimeError(f'Cannot pickle {self.__class__.__name__} object.')

    def error(self, error):
        self.close()
        raise Exception(f'Error occurred in worker: {error}')

    def process_queue(self):
        while self.get_from_queue():
            pass

    def get_from_queue(self):
        """ Get an item from the queue and store it, return True if more messages are waiting. """
        try:
            code, pool_id, *args = self.queue_out.get(True, 0.02)
            if pool_id is None:
                getattr(self, code)(*args)
            elif pool_id in self.pools:
                getattr(self.pools[pool_id], code)(*args)
            return True
        except multiprocessing.queues.Empty:
            for pool in self.pools.values():
                for handle, task in pool.tasks.items():  # retry a task if the process doing it was killed
                    if task.pid is not None \
                            and task.pid not in [child.pid for child in multiprocessing.active_children()]:
                        self.queue_in.put(task)
                        warn(f'Task {task.handle} was restarted because process {task.pid} was probably killed.')
            return False

    def add_task(self, task):
        """ Add new iteration, using optional manually defined handle."""
        if self.is_alive and not self.event.is_set():
            while self.queue_in.full():
                self.get_from_queue()
            self.queue_in.put(task)

    def get_newest_for_pool(self, pool):
        """ Request the newest key and result and delete its record. Wait if result not yet available. """
        while len(pool.tasks):
            self.get_from_queue()
            for task in pool.tasks.values():
                if task.done:
                    handle, result = task.handle, task.result
                    pool.tasks.pop(handle)
                    return handle, result

    def close(self):
        self.__class__.instance = None

        def empty_queue(queue):
            if not queue._closed:
                while not queue.empty():
                    try:
                        queue.get(True, 0.02)
                    except multiprocessing.queues.Empty:
                        pass

        def close_queue(queue):
            empty_queue(queue)
            if not queue._closed:
                queue.close()
            queue.join_thread()

        if self.is_alive:
            self.is_alive = False
            self.event.set()
            self.pool.close()
            while self.n_workers.value:
                empty_queue(self.queue_in)
                empty_queue(self.queue_out)
            empty_queue(self.queue_in)
            empty_queue(self.queue_out)
            self.pool.join()
            close_queue(self.queue_in)
            close_queue(self.queue_out)
            self.handle = 0
            self.tasks = {}


class Worker:
    """ Manages executing the target function which will be executed in different processes. """
    def __init__(self, queue_in, queue_out, n_workers, event, cachesize=48):
        self.cache = DequeDict(cachesize)
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.n_workers = n_workers
        self.event = event

    def add_to_queue(self, *args):
        while not self.event.is_set():
            try:
                self.queue_out.put(args, timeout=0.1)
                break
            except multiprocessing.queues.Full:
                continue

    def __call__(self):
        pid = getpid()
        while not self.event.is_set():
            try:
                task = self.queue_in.get(True, 0.02)
                try:
                    self.add_to_queue('started', task.pool_id, task.handle, pid)
                    task.set_from_cache(self.cache)
                    self.add_to_queue('done', task.pool_id, task())
                except Exception:
                    self.add_to_queue('task_error', task.pool_id, task.handle, format_exc())
            except multiprocessing.queues.Empty:
                continue
            except Exception:
                self.add_to_queue('error', None, format_exc())
                self.event.set()
        for child in multiprocessing.active_children():
            child.kill()
        with self.n_workers.get_lock():
            self.n_workers.value -= 1


def pmap(fun, iterable=None, args=None, kwargs=None, total=None, desc=None, bar=True, terminator=None,
         serial=None, length=None, **bar_kwargs):
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
        def chunk_fun(iterator, *args, **kwargs):
            return [fun(i, *args, **kwargs) for i in iterator]

    args = args or ()
    kwargs = kwargs or {}

    if 'total' not in bar_kwargs:
        bar_kwargs['total'] = sum(iterable.lengths)
    if 'desc' not in bar_kwargs:
        bar_kwargs['desc'] = desc
    if 'disable' not in bar_kwargs:
        bar_kwargs['disable'] = not bar
    if serial is True or (serial is None and len(iterable) < min(cpu_count, 4)):  # serial case
        if callable(bar):
            return sum([chunk_fun(c, *args, **kwargs) for c in ExternalBar(iterable, bar)], [])
        else:
            return sum([chunk_fun(c, *args, **kwargs) for c in tqdm(iterable, **bar_kwargs)], [])
    else:   # parallel case
        with ExitStack() as stack:
            if callable(bar):
                bar = stack.enter_context(ExternalBar(callback=bar))
            elif bar is True:
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
def parfor(*args, **kwargs):
    def decfun(fun):
        return pmap(fun, *args, **kwargs)
    return decfun


def deprecated(cls, name):
    """ This is a decorator which can be used to mark functions and classes as deprecated. It will result in a warning
        being emitted when the function or class is used."""
    @wraps(cls)
    def wrapper(*args, **kwargs):
        warn(f'parfor: use of \'{name}\' is deprecated, use \'{cls.__name__}\' instead',
             category=DeprecationWarning, stacklevel=2)
        warn(f'parfor: use of \'{name}\' is deprecated, use \'{cls.__name__}\' instead',
             category=DeprecationWarning, stacklevel=3)
        return cls(*args, **kwargs)
    return wrapper


# backwards compatibility
parpool = deprecated(ParPool, 'parpool')
Parpool = deprecated(ParPool, 'Parpool')
chunks = deprecated(Chunks, 'chunks')
