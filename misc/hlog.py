from contextlib import contextmanager
import logging
import threading

state = threading.local()
state.path = []

@contextmanager
def task(name):
    state.path.append(name)
    yield
    state.path.pop()

def log(value):
    print('%s %s' % ('/'.join(state.path), value))

def value(name, value):
    with task(name):
        log(value)

def loop(template, coll):
    for i, item in enumerate(coll):
        with task(template % i):
            yield item

def fn(name):
    def wrap(underlying):
        def wrapped(*args, **kwargs):
            with task(name):
                underlying(*args, **kwargs)
        return wrapped
    return wrap
