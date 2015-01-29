from __future__ import print_function

from logic import *

def py2pl(d):

    if type(d) == list or type(d) == tuple:
        if len(d) == 1:
            return Term('.', py2pl(d[0]), Term('[]'))
        tail = Term('.', py2pl(d[-2]), py2pl(d[-1]))
        for el in reversed(d[:-2]):
            tail = Term('.', py2pl(el), tail)
        return tail

    if type(d) == str or type(d) == int or type(d) == float:
        return Constant(d)

    raise Exception("Cannot convert from python to prolog: {} ({}).".format(d, type(d)))
    return None


def pl2py(d):

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

    raise Exception("Cannot convert from prolog to python: {} ({}).".format(d, type(d)))
    return None

