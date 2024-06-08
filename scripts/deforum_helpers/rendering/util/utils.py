from contextlib import contextmanager


def put_all(dictionaries, key, callable_or_value):
    for dictionary in dictionaries:
        dictionary[key] = callable_or_value() if callable(callable_or_value) else callable_or_value


def put_if_present(dictionary, key, value):
    if value is not None:
        dictionary[key] = value


@contextmanager
def context(cls_or_instance):
    yield cls_or_instance
