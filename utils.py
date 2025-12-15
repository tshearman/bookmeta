from functools import reduce


def andthen(*funcs):
    """Return a function that applies the provided callables in reading order."""
    if not funcs:
        raise ValueError("andthen requires at least one function")

    def _inner_(v):
        for f in funcs:
            v = f(v)
        return v

    return _inner_
