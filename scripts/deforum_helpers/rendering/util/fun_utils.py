import collections.abc
from functools import reduce
from itertools import chain


def flat_map(func, iterable):
    """Applies a function to each element in an iterable and flattens the results."""
    mapped_iterable = map(func, iterable)
    if any(isinstance(item, collections.abc.Iterable) for item in mapped_iterable):
        return chain.from_iterable(mapped_iterable)
    else:
        return mapped_iterable


def pipe(*funcs):
    """Pipes a value through a sequence of functions"""
    return lambda value: reduce(lambda x, f: f(x), funcs, value)
