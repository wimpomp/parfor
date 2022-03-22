import multiprocessing
import dill
from os import getpid
from tqdm.auto import tqdm
from traceback import format_exc
from pickle import PicklingError, dispatch_table
from psutil import Process
from collections import OrderedDict
from io import BytesIO
from warnings import warn

try:
    from javabridge import kill_vm
except ImportError:
    kill_vm = lambda: None

failed_rv = (lambda *args, **kwargs: None, ())
cpu_count = int(multiprocessing.cpu_count())


class Pickler(dill.Pickler):
    """ Overload dill to ignore unpickleble parts of objects.
        You probably didn't want to use these parts anyhow.
        However, if you did, you'll have to find some way to make them pickleble.
    """
    def save(self, obj, save_persistent_id=True):
        """ Copied from pickle and amended. """
        self.framer.commit_frame()

        # Check for persistent id (defined by a subclass)
        pid = self.persistent_id(obj)
        if pid is not None and save_persistent_id:
            self.save_pers(pid)
            return

        # Check the memo
        x = self.memo.get(id(obj))
        if x is not None:
            self.write(self.get(x[0]))
            return

        rv = NotImplemented
        reduce = getattr(self, "reducer_override", None)
        if reduce is not None:
            rv = reduce(obj)

        if rv is NotImplemented:
            # Check the type dispatch table
            t = type(obj)
            f = self.dispatch.get(t)
            if f is not None:
                f(self, obj)  # Call unbound method with explicit self
                return

            # Check private dispatch table if any, or else
            # copyreg.dispatch_table
            reduce = getattr(self, 'dispatch_table', dispatch_table).get(t)
            if reduce is not None:
                rv = reduce(obj)
            else:
                # Check for a class with a custom metaclass; treat as regular
                # class
                if issubclass(t, type):
                    self.save_global(obj)
                    return

                # Check for a __reduce_ex__ method, fall back to __reduce__
                reduce = getattr(obj, "__reduce_ex__", None)
                try:
                    if reduce is not None:
                        rv = reduce(self.proto)
                    else:
                        reduce = getattr(obj, "__reduce__", None)
                        if reduce is not None:
                            rv = reduce()
                        else:
                            raise PicklingError("Can't pickle %r object: %r" %
                                                (t.__name__, obj))
                except Exception:
                    rv = failed_rv

        # Check for string returned by reduce(), meaning "save as global"
        if isinstance(rv, str):
            try:
                self.save_global(obj, rv)
            except Exception:
                self.save_global(obj, failed_rv)
            return

        # Assert that reduce() returned a tuple
        if not isinstance(rv, tuple):
            raise PicklingError("%s must return string or tuple" % reduce)

        # Assert that it returned an appropriately sized tuple
        l = len(rv)
        if not (2 <= l <= 6):
            raise PicklingError("Tuple returned by %s must have "
                                "two to six elements" % reduce)

        # Save the reduce() output and finally memoize the object
        try:
            self.save_reduce(obj=obj, *rv)
        except Exception:
            self.save_reduce(obj=obj, *failed_rv)


def dumps(obj, protocol=None, byref=None, fmode=None, recurse=True, **kwds):
    """pickle an object to a string"""
    protocol = dill.settings['protocol'] if protocol is None else int(protocol)
    _kwds = kwds.copy()
    _kwds.update(dict(byref=byref, fmode=fmode, recurse=recurse))
    file = BytesIO()
    Pickler(file, protocol, **_kwds).dump(obj)
    return file.getvalue()


class chunks(object):
    """ Yield successive chunks from lists.
        Usage: chunks(s, list0, list1, ...)
               chunks(list0, list1, ..., s=s)
               chunks(list0, list1, ..., n=n)
               chunks(list0, list1, ..., r=r)
        s: size of chunks, might change to optimize devision between chunks
        n: number of chunks, coerced to 1 <= n <= len(list0)
        r: number of chunks / number of cpus, coerced to 1 <= n <= len(list0)
        both s and n or r are given: use n or r, unless the chunk size would be bigger than s
        both r and n are given: use n
    """

    def __init__(self, *args, **kwargs):
        if 's' in kwargs and ('n' in kwargs or 'r' in kwargs):
            number_of_items = min(*[len(a) for a in args]) if len(args) > 1 else len(args[0])
            n = kwargs['n'] if 'n' in kwargs else int(cpu_count * kwargs['r'])
            n = n if number_of_items < kwargs['s'] * n else round(number_of_items / kwargs['s'])
        elif 's' in kwargs:  # size of chunks
            number_of_items = min(*[len(a) for a in args]) if len(args) > 1 else len(args[0])
            n = round(number_of_items / kwargs['s'])
        elif 'n' in kwargs or 'r' in kwargs:  # number of chunks
            number_of_items = min(*[len(a) for a in args]) if len(args) > 1 else len(args[0])
            n = kwargs['n'] if 'n' in kwargs else int(cpu_count * kwargs['r'])
        else:  # size of chunks in 1st argument
            s, *args = args
            number_of_items = min(*[len(a) for a in args]) if len(args) > 1 else len(args[0])
            n = round(number_of_items / s)
        self.args = args
        self.number_of_arguments = len(args) == 1
        self.number_of_items = number_of_items
        self.len = max(1, min(number_of_items, n))
        self.lengths = [((i + 1) * self.number_of_items // self.len) - (i * self.number_of_items // self.len)
                        for i in range(self.len)]

    def __iter__(self):
        for i in range(self.len):
            p, q = (i * self.number_of_items // self.len), ((i + 1) * self.number_of_items // self.len)
            yield self.args[0][p:q] if self.number_of_arguments else [a[p:q] for a in self.args]

    def __len__(self):
        return self.len


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


External_bar = ExternalBar


class tqdmm(tqdm):
    """ Overload tqdm to make a special version of tqdm functioning as a meter. """

    def __init__(self, *args, **kwargs):
        self._n = 0
        self.disable = False
        if 'bar_format' not in kwargs and len(args) < 16:
            kwargs['bar_format'] = '{n}/{total}'
        super(tqdmm, self).__init__(*args, **kwargs)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if not value == self.n:
            self._n = int(value)
            self.refresh()

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        if not self.leave:
            self.n = self.total
        super(tqdmm, self).__exit__(exc_type, exc_value, traceback)


def parfor(*args, **kwargs):
    """ @parfor(iterator=None, args=(), kwargs={}, length=None, desc=None, bar=True, qbar=True, rP=1/3, serial=4):
        decorator to parallize for-loops

        required arguments:
            fun:    function taking arguments: iteration from  iterable, other arguments defined in args & kwargs
            iterable: iterable from which an item is given to fun as a first argument

        optional arguments:
            args:   tuple with other unnamed arguments to fun
            kwargs: dict with other named arguments to fun
            length: give the length of the iterator in cases where len(iterator) results in an error
            desc:   string with description of the progress bar
            bar:    bool enable progress bar, or a function taking the number of passed iterations as an argument
            pbar:   bool enable buffer indicator bar, or a function taking the queue size as an argument
            rP:     ratio workers to cpu cores, default: 1
            nP:     number of workers, default: None, overrides rP if not None
                number of workers will always be at least 2
            serial: switch to serial if number of tasks less than serial, default: 4

        output:       list with results from applying the decorated function to each iteration of the iterator
                      specified as the first argument to the function

        examples:
            << from time import sleep

            <<
            @parfor(range(10), (3,))
            def fun(i, a):
                sleep(1)
                return a*i**2
            fun

            >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]

            <<
            @parfor(range(10), (3,), bar=False)
            def fun(i, a):
                sleep(1)
                return a*i**2
            fun

            >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]

            <<
            def fun(i, a):
                sleep(1)
                return a*i**2
            pmap(fun, range(10), (3,))

            >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]

            equivalent to using the deco module:
            <<
            @concurrent
            def fun(i, a):
                time.sleep(1)
                return a*i**2

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
                fun.append(a*i**2)
            fun

            >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]
        """
    def decfun(fun):
        return pmap(fun, *args, **kwargs)
    return decfun


class Hasher(object):
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
            self.obj = dill.loads(self.str)
        elif self.hash in cache:
            self.obj = cache[self.hash]
        else:
            self.obj = cache[self.hash] = dill.loads(self.str)


class HashDescriptor(object):
    def __set_name__(self, owner, name):
        self.owner, self.name = owner, '_' + name

    def __set__(self, instance, value):
        if isinstance(value, Hasher):
            setattr(instance, self.name, value)
        else:
            setattr(instance, self.name, Hasher(value))

    def __get__(self, instance, owner):
        return getattr(instance, self.name).obj


class deque_dict(OrderedDict):
    def __init__(self, maxlen=None, *args, **kwargs):
        self.maxlen = maxlen
        super(deque_dict, self).__init__(*args, **kwargs)

    def __truncate__(self):
        while len(self) > self.maxlen:
            self.popitem(False)

    def __setitem__(self, *args, **kwargs):
        super(deque_dict, self).__setitem__(*args, **kwargs)
        self.__truncate__()

    def update(self, *args, **kwargs):
        super(deque_dict, self).update(*args, **kwargs)
        self.__truncate__()


class Task(object):
    fun = HashDescriptor()
    args = HashDescriptor()
    kwargs = HashDescriptor()

    def __init__(self, fun=None, args=None, kwargs=None, handle=None, n=None, done=False, result=None):
        self.fun = fun or (lambda *args, **kwargs: None)
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.handle = handle
        self.n = n
        self.done = done
        self.result = dill.loads(result) if self.done else None
        self.pid = None

    def __reduce__(self):
        if self.done:
            return self.__class__, (None, None, None, self.handle, None, self.done, dumps(self.result, recurse=True))
        else:
            return self.__class__, (self._fun, self._args, self._kwargs, self.handle, dumps(self.n, recurse=True),
                                    self.done)

    def set_from_cache(self, cache=None):
        self.n = dill.loads(self.n)
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
            return 'Task {}, result: {}'.format(self.handle, self.result)
        else:
            return 'Task {}'.format(self.handle)


class parpool(object):
    """ Parallel processing with addition of iterations at any time and request of that result any time after that.
        The target function and its argument can be changed at any time.
    """
    def __init__(self, fun=None, args=None, kwargs=None, rP=None, nP=None, bar=None, qbar=None, terminator=None,
                 qsize=None):
        """ fun, args, kwargs: target function and its arguments and keyword arguments
            rP: ratio workers to cpu cores, default: 1
            nP: number of workers, default, None, overrides rP if not None
            bar, qbar: instances of tqdm and tqdmm to use for monitoring buffer and progress """
        if rP is None and nP is None:
            self.nP = cpu_count
        elif nP is None:
            self.nP = int(round(rP * cpu_count))
        else:
            self.nP = int(nP)
        self.nP = max(self.nP, 2)
        self.task = Task(fun, args, kwargs)
        if hasattr(multiprocessing, 'get_context'):
            ctx = multiprocessing.get_context('spawn')
        else:
            ctx = multiprocessing
        self.is_started = False
        self.n_tasks = ctx.Value('i', self.nP)
        self.event = ctx.Event()
        self.queue_in = ctx.Queue(qsize or 3 * self.nP)
        self.queue_out = ctx.Queue(qsize or 12 * self.nP)
        self.pool = ctx.Pool(self.nP, self._worker(self.queue_in, self.queue_out, self.n_tasks, self.event, terminator))
        self.is_alive = True
        self.handle = 0
        self.tasks = {}
        self.bar = bar
        self.bar_lengths = {}
        self.qbar = qbar
        if self.qbar is not None:
            self.qbar.total = 3 * self.nP

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def _get_from_queue(self):
        """ Get an item from the queue and store it, return True if more messages are waiting. """
        try:
            code, *args = self.queue_out.get(True, 0.02)
            getattr(self, code)(*args)
            return True
        except multiprocessing.queues.Empty:
            for handle, task in self.tasks.items():  # retry a task if the process doing it was killed
                if task.pid is not None and task.pid not in [child.pid for child in Process().children()]:
                    self.queue_in.put(task)
                    warn('Task {} was restarted because process {} was probably killed.'.format(task.handle, task.pid))
            return False

    def error(self, error):
        self.close()
        raise Exception('Error occured in worker: {}'.format(error))

    def task_error(self, handle, error):
        if handle in self:
            task = self.tasks[handle]
            print('Error from process working on iteration {}:\n'.format(handle))
            print(error)
            self.close()
            print('Retrying in main thread...')
            fun = task.fun.__name__
            task()
            raise Exception('Function \'{}\' cannot be executed by parfor, amend or execute in serial.'.format(fun))

    def done(self, task):
        if task.handle in self:  # if not, the task was restarted erroneously
            self.tasks[task.handle] = task
            if self.bar is not None:
                self.bar.update(self.bar_lengths.pop(task.handle))
            self._qbar_update()

    def started(self, handle, pid):
        self.is_started = True
        if handle in self:  # if not, the task was restarted erroneously
            self.tasks[handle].pid = pid

    def __call__(self, n, fun=None, args=None, kwargs=None, handle=None, barlength=1):
        """ Add new iteration, using optional manually defined handle."""
        if self.is_alive and not self.event.is_set():
            self.task = Task(fun or self.task.fun, args or self.task.args, kwargs or self.task.kwargs, handle, n)
            while self.queue_in.full():
                self._get_from_queue()
            if handle is None:
                handle = self.handle
                self.handle += 1
                self.tasks[handle] = self.task
                self.queue_in.put(self.task)
                self.bar_lengths[handle] = barlength
                self._qbar_update()
                return handle
            elif handle not in self:
                self.tasks[handle] = self.task
                self.queue_in.put(self.task)
                self.bar_lengths[handle] = barlength
            self._qbar_update()

    def _qbar_update(self):
        if self.qbar is not None:
            try:
                self.qbar.n = self.queue_in.qsize()
            except Exception:
                pass

    def __setitem__(self, handle, n):
        """ Add new iteration. """
        self(n, handle=handle)

    def __getitem__(self, handle):
        """ Request result and delete its record. Wait if result not yet available. """
        if handle not in self:
            raise ValueError('No handle: {}'.format(handle))
        while not self.tasks[handle].done:
            if not self._get_from_queue() and not self.tasks[handle].done and self.is_started and not self.working:
                for _ in range(10):  # wait some time while processing possible new messages
                    self._get_from_queue()
                if not self._get_from_queue() and not self.tasks[handle].done and self.is_started and not self.working:
                    # retry a task if the process was killed while retrieving the task
                    self.queue_in.put(self.tasks[handle])
                    warn('Task {} was restarted because the process retrieving it was probably killed.'.format(handle))
        result = self.tasks[handle].result
        self.tasks.pop(handle)
        return result

    @property
    def working(self):
        return not all([task.pid is None for task in self.tasks.values()])

    def get_newest(self):
        """ Request the newest key and result and delete its record. Wait if result not yet available. """
        while len(self.tasks):
            self._get_from_queue()
            for task in self.tasks:
                if task.done:
                    return task.handle, task.result

    def __delitem__(self, handle):
        self.tasks.pop(handle)

    def __contains__(self, handle):
        return handle in self.tasks

    def __repr__(self):
        if self.is_alive:
            return '{} with {} workers.'.format(self.__class__, self.nP)
        else:
            return 'Closed {}'.format(self.__class__)

    def close(self):
        if self.is_alive:
            self.is_alive = False
            self.event.set()
            self.pool.close()
            while self.n_tasks.value:
                self._empty_queue(self.queue_in)
                self._empty_queue(self.queue_out)
            self._empty_queue(self.queue_in)
            self._empty_queue(self.queue_out)
            self.pool.join()
            self._close_queue(self.queue_in)
            self._close_queue(self.queue_out)
            self.handle = 0
            self.tasks = {}

    @staticmethod
    def _empty_queue(queue):
        if not queue._closed:
            while not queue.empty():
                try:
                    queue.get(True, 0.02)
                except multiprocessing.queues.Empty:
                    pass

    @staticmethod
    def _close_queue(queue):
        if not queue._closed:
            while not queue.empty():
                try:
                    queue.get(True, 0.02)
                except multiprocessing.queues.Empty:
                    pass
            queue.close()
        queue.join_thread()

    class _worker(object):
        """ Manages executing the target function which will be executed in different processes. """
        def __init__(self, queue_in, queue_out, n_tasks, event, terminator, cachesize=48):
            self.cache = deque_dict(cachesize)
            self.queue_in = queue_in
            self.queue_out = queue_out
            self.n_tasks = n_tasks
            self.event = event
            self.terminator = dumps(terminator, recurse=True)

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
                        self.add_to_queue('started', task.handle, pid)
                        task.set_from_cache(self.cache)
                        self.add_to_queue('done', task())
                    except Exception:
                        self.add_to_queue('task_error', task.handle, format_exc())
                        self.event.set()
                except multiprocessing.queues.Empty:
                    continue
                except Exception:
                    self.add_to_queue('error', format_exc())
                    self.event.set()
            terminator = dill.loads(self.terminator)
            kill_vm()
            if terminator is not None:
                terminator()
            with self.n_tasks.get_lock():
                self.n_tasks.value -= 1


def pmap(fun, iterable=None, args=None, kwargs=None, length=None, desc=None, bar=True, qbar=False, terminator=None,
         rP=1, nP=None, serial=4, qsize=None):
    """ map a function fun to each iteration in iterable
            best use: iterable is a generator and length is given to this function

        fun:    function taking arguments: iteration from  iterable, other arguments defined in args & kwargs
        iterable: iterable from which an item is given to fun as a first argument
        args:   tuple with other unnamed arguments to fun
        kwargs: dict with other named arguments to fun
        length: give the length of the iterator in cases where len(iterator) results in an error
        desc:   string with description of the progress bar
        bar:    bool enable progress bar, or a callback function taking the number of passed iterations as an argument
        pbar:   bool enable buffer indicator bar, or a callback function taking the queue size as an argument
        terminator: function which is executed in each worker after all the work is done
        rP:     ratio workers to cpu cores, default: 1
        nP:     number of workers, default, None, overrides rP if not None
        serial: switch to serial if number of tasks less than serial, default: 4
    """
    args = args or ()
    kwargs = kwargs or {}
    try:
        length = len(iterable)
    except Exception:
        pass
    if length and length < serial:  # serial case
        if callable(bar):
            return [fun(c, *args, **kwargs) for c in ExternalBar(iterable, bar)]
        else:
            return [fun(c, *args, **kwargs) for c in tqdm(iterable, total=length, desc=desc, disable=not bar)]
    else:                           # parallel case
        chunk = isinstance(iterable, chunks)
        if chunk:
            length = iterable.number_of_items
        with ExternalBar(callback=qbar) if callable(qbar) \
                else tqdmm(total=0, desc='Task buffer', disable=not qbar, leave=False) as qbar, \
             ExternalBar(callback=bar) if callable(bar) else tqdm(total=length, desc=desc, disable=not bar) as bar:
            with parpool(fun, args, kwargs, rP, nP, bar, qbar, terminator, qsize) as p:
                length = 0
                for i, j in enumerate(iterable):  # add work to the queue
                    if chunk:
                        p(j, handle=i, barlength=iterable.lengths[i])
                    else:
                        p[i] = j
                    if bar.total is None or bar.total < i+1:
                        bar.total = i+1
                    length += 1
                return [p[i] for i in range(length)]  # collect the results
