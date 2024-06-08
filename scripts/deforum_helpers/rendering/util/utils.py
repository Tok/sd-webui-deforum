from contextlib import contextmanager
from PIL import Image


def put_all(dictionaries, key, value):
    return list(map(lambda d: {**d, key: value}, dictionaries))


def put_if_present(dictionary, key, value):
    if value is not None:
        dictionary[key] = value


def _call_or_use():
    return callable_or_value() if callable(callable_or_value) else callable_or_value


def call_or_use_on_cond(condition, callable_or_value):
    return _call_or_use() if condition else None


@contextmanager
def context(cls_or_instance):
    yield cls_or_instance


def create_img(dimensions):
    return Image.new('1', dimensions, 1)
