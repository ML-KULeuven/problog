from __future__ import print_function

from .logic import *

def py2pl(d):
    """Translate a given Python datastructure into a Prolog datastructure."""

    if type(d) == list or type(d) == tuple:
        tail = Term('.', py2pl(d[-1]), Term('[]'))
        for el in reversed(d[:-1]):
            tail = Term('.', py2pl(el), tail)
        return tail

    if type(d) == str:
        return Constant('"{}"'.format(d))

    if type(d) == int or type(d) == float:
        return Constant(d)

    raise Exception("Cannot convert from Python to Prolog: {} ({}).".format(d, type(d)))
    return None


def pl2py(d):
    """Translate a given Prolog datastructure into a Python datastructure."""

    if isinstance(d, Constant):
        if type(d.value) == str:
            return d.value.replace('"','')
        return d.value

    if isinstance(d, Term):
        if d.functor == "." and d.arity == 2:
            # list
            elements = []
            tail = d
            while isinstance(tail,Term) and tail.arity == 2 and tail.functor == '.' :
                elements.append(pl2py(tail.args[0]))
                tail = tail.args[1]
            if str(tail) != '[]':
                elements.append(pl2py(tail))
            return elements

    raise Exception("Cannot convert from Prolog to Python: {} ({}).".format(d, type(d)))
    return None

