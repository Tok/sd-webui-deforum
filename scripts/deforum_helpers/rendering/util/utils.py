from contextlib import contextmanager
from PIL import Image


@contextmanager
def context(resource):
    """
    Context manager for aliasing and managing objects or class references within a `with` statement block.
    The function can help keeping code compact, less repetitive or more structured by providing a dedicated scope
    for resource access, at the tradeoff of introducing an additional level of nesting.
    It can be particularly useful when extracting logic to a separate function is undesirable.

    **Example Usage:**
    ```python
    with context(annoyingly_long_reference.to_some_expression.or_to.MyClass().or_something) as res:
        res.do_something()
        do_something_with_the_thing(res)
        res.do_something_else()
        # more lines using res
    ```

    **Arguments:**
    - `resource (object or class reference)`: The object or class reference to be managed within the context.

    **Yields:**
    - The provided `resource` argument.

    **Note:**
    The context manager is responsible for any necessary resource cleanup upon exiting the `with` block (if applicable).
    """
    yield resource


@contextmanager
def combo_context(*resources):
    yield resources


def put_all(dictionaries, key, value):
    return list(map(lambda d: {**d, key: value}, dictionaries))


def put_if_present(dictionary, key, value):
    if value is not None:
        dictionary[key] = value


def _call_or_use(callable_or_value):
    return callable_or_value() if callable(callable_or_value) else callable_or_value


def call_or_use_on_cond(condition, callable_or_value):
    return _call_or_use(callable_or_value) if condition else None


def create_img(dimensions):
    return Image.new('1', dimensions, 1)
