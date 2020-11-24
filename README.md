# Parfor
Used to parallelize for-loops using parfor in Matlab? This package allows you to do the same in python.
Take any normal serial but parallelizable for-loop and execute it in parallel using easy syntax.
Don't worry about the technical details of using the multiprocessing module, race conditions, queues,
parfor handles all that. 

Tested on linux on python 2.7 and 3.8 and on Windows and OSX on python 3.8.

## Why is parfor better than just using multiprocessing?
- Easy to use
- Using dill instead of pickle: a lot more objects can be used when parallelizing
- Progress bars are built-in

## Installation
pip install parfor

## Usage
Parfor decorates a functions and returns the result of that function evaluated in parallel for each iteration of
an iterator.

## Requires
tqdm, dill

## Limitations
Objects passed to the pool need to be dillable (dill needs to serialize them). Generators and SwigPyObjects are examples
of objects that cannot be used. They can be used however, for the iterator argument when using parfor, but its
iterations need to be dillable. You might be able to make objects dillable anyhow using dill.register.

The function evaluated in parallel needs to terminate. If parfor hangs after seeming to complete the task, it probably
is because the individual processes cannot terminate. Importing javabridge (used in python-bioformats) and starting the
java virtual machine can cause it to hang since the processes only terminate after the java vm has quit. In this case,
pass terminator=javabridge.kill_vm to parfor.

On OSX the buffer bar does not work due to limitations of the OS.

## Arguments
### Required:
    fun:      function taking arguments: iteration from  iterable, other arguments defined in args & kwargs
    iterable: iterable from which an item is given to fun as a first argument

### Optional:
    args:   tuple with other unnamed arguments to fun
    kwargs: dict with other named arguments to fun
    length: give the length of the iterator in cases where len(iterator) results in an error
    desc:   string with description of the progress bar
    bar:    bool enable progress bar
    pbar:   bool enable buffer indicator bar
    nP:     number of workers, default: number of cpu's/3
    serial: switch to serial if number of tasks less than serial, default: 4
    debug:  if an error occurs in an iteration, return the erorr instead of retrying in the main process

### Return
    list with results from applying the decorated function to each iteration of the iterator
    specified as the first argument to the function

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
    
### Using parfor on the same loop
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
Since generators don't have a predefined length, give parfor the length as an argument (optional). 
    
    <<
    import numpy as np
    c = (im for im in imagereader)
    @parfor(c, length=len(imagereader))
    def fun(im):
        return np.mean(im)
        
    >> [list with means of the images]
    
# Extra's
## Pmap
The function parfor decorates, use it like map.

## Chunks
Split a long iterator in bite-sized chunks to parallelize

## Parpool
More low-level accessibility to parallel execution. Submit tasks and request the result at any time,
(although necessarily submit first, then request a specific task), use different functions and function
arguments for different tasks.

## Tqdmm
Meter bar, inherited from tqdm, used for displaying buffers.
