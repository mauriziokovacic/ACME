import time


def profile(fun, *args, n=1000, **kwargs):
    t = time.time()
    for i in range(n):
        eval(fun, *args, **kwargs)
    return (time.time()-t)/n
