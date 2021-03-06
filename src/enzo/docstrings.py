"""Docstring-related functions, such as tools for sphinx crosslinking."""


def append_to_see_also(func, see_also_section):
    """Append `see_also_section` to the end of `func`'s docstring.

    If a See Also section doesn't exist, create one at the bottom.

    Warning
    -------
    This :func:`append_to_see_also` blindly appends to the end of `func`'s docstring if
    the docstring already contains a See Also section. This means the See Also
    section *must* be the last section in the docstring.

    Parameters
    ----------
    func : function
    see_also_section : str
    """
    full_see_also_section = f"\n\n    See Also\n    --------\n    {see_also_section}"
    try:
        see_also_present = "See Also" in func.__doc__
    except TypeError:
        func.__doc__ = full_see_also_section
    else:
        if see_also_present:
            func.__doc__ += see_also_section
        else:
            func.__doc__ += full_see_also_section
