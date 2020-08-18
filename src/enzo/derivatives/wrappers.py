"""Wrappers for functions with derivatives."""

from ..docstrings import append_to_see_also

DERIVATIVE_MODULE = "enzo.derivatives"


def with_derivative(derivative):
    """Decorator for functions with derivatives

    Parameters
    ----------
    derivative : callable
        The derivative of the decorated function.
    """

    def wrapper(func):
        append_to_see_also(derivative, f":func:`{func.__module__}.{func.__name__}`")
        append_to_see_also(func, f":func:`{DERIVATIVE_MODULE}.{derivative.__name__}`")

        func.derivative = derivative
        return func

    return wrapper
