import dill
from pickle import PicklingError, dispatch_table
from io import BytesIO


failed_rv = (lambda *args, **kwargs: None, ())
loads = dill.loads


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
