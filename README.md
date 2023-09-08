[![pytest](https://github.com/wimpomp/parfor/actions/workflows/pytest.yml/badge.svg)](https://github.com/wimpomp/parfor/actions/workflows/pytest.yml)

# Parfor
Used to parallelize for-loops using parfor in Matlab? This package allows you to do the same in python.
Take any normal serial but parallelizable for-loop and execute it in parallel using easy syntax.
Don't worry about the technical details of using the multiprocessing module, race conditions, queues,
parfor handles all that. 

Tested on linux, Windows and OSX with python 3.10.

## Why is parfor better than just using multiprocessing?
- Easy to use
- Using dill instead of pickle: a lot more objects can be used when parallelizing
- Progress bars are built-in

## Installation
`pip install parfor`

## Usage
Parfor decorates a functions and returns the result of that function evaluated in parallel for each iteration of
an iterator.

## Requires
tqdm, dill

## Limitations
Objects passed to the pool need to be dillable (dill needs to serialize them). Generators and SwigPyObjects are examples
of objects that cannot be used. They can be used however, for the iterator argument when using parfor, but its
iterations need to be dillable. You might be able to make objects dillable anyhow using `dill.register` or with
`__reduce__`, `__getstate__`, etc.

## Arguments
### Required:
    fun:      function taking arguments: iteration from  iterable, other arguments defined in args & kwargs
    iterable: iterable or iterator from which an item is given to fun as a first argument

### Optional:
    args:   tuple with other unnamed arguments to fun
    kwargs: dict with other named arguments to fun
    total:  give the length of the iterator in cases where len(iterator) results in an error
    desc:   string with description of the progress bar
    bar:    bool enable progress bar,
                or a callback function taking the number of passed iterations as an argument
    pbar:   bool enable buffer indicator bar, or a callback function taking the queue size as an argument
    rP:     ratio workers to cpu cores, default: 1
    nP:     number of workers, default, None, overrides rP if not None
    serial: execute in series instead of parallel if True, None (default): let pmap decide
    qsize:  maximum size of the task queue
    length: deprecated alias for total
    **bar_kwargs: keywords arguments for tqdm.tqdm

### Return
    list with results from applying the function 'fun' to each iteration of the iterable / iterator

## Examples
### Normal serial for loop
    <<
    from time import sleep

    a = 3
    fun = []
    for i in range(10):
        sleep(1)
        fun.append(a*i**2)
    print(fun)

    >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]
    
### Using parfor to parallelize
    <<
    from time import sleep
    from parfor import parfor
    @parfor(range(10), (3,))
    def fun(i, a):
        sleep(1)
        return a*i**2
    print(fun)

    >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]

    <<
    @parfor(range(10), (3,), bar=False)
    def fun(i, a):
        sleep(1)
        return a*i**2
    print(fun)

    >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]

### Using parfor in a script/module/.py-file
Parfor should never be executed during the import phase of a .py-file. To prevent that from happening
use the `if __name__ == '__main__':` structure:

    <<
    from time import sleep
    from parfor import parfor
    
    if __name__ == '__main__':
        @parfor(range(10), (3,))
        def fun(i, a):
            sleep(1)
            return a*i**2
        print(fun)

    >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]    
or:

    <<
    from time import sleep
    from parfor import parfor
    
    def my_fun(*args, **kwargs):
        @parfor(range(10), (3,))
        def fun(i, a):
            sleep(1)
            return a*i**2
        return fun
    
    if __name__ == '__main__':
        print(my_fun())

    >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]

### If you hate decorators not returning a function
pmap maps an iterator to a function like map does, but in parallel

    <<
    from parfor import pmap
    from time import sleep
    def fun(i, a):
        sleep(1)
        return a*i**2
    print(pmap(fun, range(10), (3,)))

    >> [0, 3, 12, 27, 48, 75, 108, 147, 192, 243]     
    
### Using generators
If iterators like lists and tuples are too big for the memory, use generators instead.
Since generators don't have a predefined length, give parfor the length (total) as an argument (optional). 
    
    <<
    import numpy as np
    c = (im for im in imagereader)
    @parfor(c, total=len(imagereader))
    def fun(im):
        return np.mean(im)
        
    >> [list with means of the images]
    
# Extra's
## `pmap`
The function parfor decorates, use it like `map`.

## `Chunks`
Split a long iterator in bite-sized chunks to parallelize

## `ParPool`
More low-level accessibility to parallel execution. Submit tasks and request the result at any time,
(although necessarily submit first, then request a specific task), use different functions and function
arguments for different tasks.
