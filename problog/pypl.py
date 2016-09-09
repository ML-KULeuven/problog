"""
Part of the ProbLog distribution.

Copyright 2015 KU Leuven, DTAI Research Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function

from .logic import Constant, Term


def py2pl(d):
    """Translate a given Python datastructure into a Prolog datastructure."""

    if type(d) == list or type(d) == tuple:
        if type(d) == tuple:
            f = ','
            tail = py2pl(d[-1])
        else:
            f = '.'
            if not d:
                return Term('[]')
            tail = Term(f, py2pl(d[-1]), Term('[]'))
        for el in reversed(d[:-1]):
            tail = Term(f, py2pl(el), tail)
        return tail

    if type(d) == str:
        return Constant('"{}"'.format(d))

    if type(d) == int or type(d) == float:
        return Constant(d)

    if isinstance(d, Term):
        return d

    raise ValueError("Cannot convert from Python to Prolog: {} ({}).".format(d, type(d)))


def pl2py(d):
    """Translate a given Prolog datastructure into a Python datastructure."""

    if isinstance(d, Constant):
        if type(d.value) == str:
            return d.value.replace('"', '').replace("'", '')
        return d.value

    if isinstance(d, Term):
        if d.functor == "." and d.arity == 2:
            # list
            elements = []
            tail = d
            while isinstance(tail, Term) and tail.arity == 2 and tail.functor == '.':
                elements.append(pl2py(tail.args[0]))
                tail = tail.args[1]
            if str(tail) != '[]':
                elements.append(pl2py(tail))
            return elements
        elif d.functor == "," and d.arity == 2:
            # list
            elements = []
            tail = d
            while isinstance(tail, Term) and tail.arity == 2 and tail.functor == ',':
                elements.append(pl2py(tail.args[0]))
                tail = tail.args[1]
            if str(tail) != '[]':
                elements.append(pl2py(tail))
            return elements
        else:
            return d

    raise ValueError("Cannot convert from Prolog to Python: {} ({}).".format(d, type(d)))

