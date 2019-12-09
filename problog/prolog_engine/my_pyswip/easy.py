# -*- coding: utf-8 -*-


# pyswip.easy -- PySwip helper functions
# Copyright (c) 2007-2018 Yüce Tekol
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from my_pyswip.core import *


class InvalidTypeError(TypeError):
    def __init__(self, *args):
        type_ = args and args[0] or "Unknown"
        msg = "Term is expected to be of type: '%s'" % type_
        Exception.__init__(self, msg, *args)


class ArgumentTypeError(Exception):
    """
    Thrown when an argument has the wrong type.
    """
    def __init__(self, expected, got):
        msg = "Expected an argument of type '%s' but got '%s'" % (expected, got)
        Exception.__init__(self, msg)


class Atom(object):
    __slots__ = "handle", "swipl", "chars"

    def __init__(self, handleOrChars, swipl):
        """Create an atom.
        ``handleOrChars``: handle or string of the atom.
        """

        self.swipl = swipl

        if isinstance(handleOrChars, str):
            self.handle = self.swipl.PL_new_atom(handleOrChars)
            self.chars = handleOrChars
        else:
            self.handle = handleOrChars
            self.swipl.PL_register_atom(self.handle)
            #self.chars = c_char_p(PL_atom_chars(self.handle)).value
            self.chars = self.swipl.PL_atom_chars(self.handle)

    def fromTerm(cls, term, swipl):
        """Create an atom from a Term or term handle."""

        if isinstance(term, Term):
            term = term.handle
        elif not isinstance(term, (c_void_p, int)):
            raise ArgumentTypeError((str(Term), str(c_void_p)), str(type(term)))

        a = atom_t()
        if swipl.PL_get_atom(term, byref(a)):
            return cls(a.value, swipl)
    fromTerm = classmethod(fromTerm)

    def __del__(self):
        if not cleaned:
            self.swipl.PL_unregister_atom(self.handle)

    def get_value(self):
        ret = self.chars
        if not isinstance(ret, str):
            ret = ret.decode()
        return ret

    value = property(get_value)

    def __str__(self):
        if self.chars is not None:
            return self.value
        else:
            return self.__repr__()

    def __repr__(self):
        return str(self.handle).join(["Atom('", "')"])

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.handle == other.handle

    def __hash__(self):
        return self.handle


class Term(object):
    __slots__ = "handle", "swipl",  "chars", "__value", "a0"

    def __init__(self, swipl, handle=None, a0=None):
        self.swipl = swipl
        if handle:
            #self.handle = PL_copy_term_ref(handle)
            self.handle = handle
        else:
            self.handle = self.swipl.PL_new_term_ref()
        self.chars = None
        self.a0 = a0

    def __invert__(self):
        return Functor("not", self.swipl, 1)

    def get_value(self):
        pass

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.swipl.PL_compare(self.handle, other.handle) == 0

    def __hash__(self):
        return self.handle


class Variable(object):
    __slots__ = "handle", "swipl", "chars"

    def __init__(self, handle=None, swipl=None, name=None):
        self.swipl = swipl
        self.chars = None
        if name:
            self.chars = name
        if handle:
            self.handle = handle
            s = create_string_buffer(b"\00"*64)  # FIXME:
            ptr = cast(s, c_char_p)
            if self.swipl.PL_get_chars(handle, byref(ptr), CVT_VARIABLE|BUF_RING):
                self.chars = ptr.value
        else:
            self.handle = self.swipl.PL_new_term_ref()
            #PL_put_variable(self.handle)
        if (self.chars is not None) and not isinstance(self.chars, str):
            self.chars = self.chars.decode()

    def unify(self, value):
        t = self.get_term_ref()
        self._unifier(t, value)
        self.handle = t

    def get_term_ref(self):
        if self.handle is None:
            return self.swipl.PL_new_term_ref(self.handle)
        return self.swipl.PL_copy_term_ref(self.handle)

    def get_value(self):
        return getTerm(self.handle, self.swipl)

    value = property(get_value, unify)

    def unified(self):
        return self.swipl.PL_term_type(self.handle) == PL_VARIABLE

    def _unifier(self, t, value):
        if type(value) == str:
            self.swipl.PL_unify_atom_chars(t, value)
        elif type(value) == int:
            self.swipl.PL_unify_integer(t, value)
        elif type(value) == bool:
            self.swipl.PL_unify_bool(t, value)
        elif type(value) == float:
            self.swipl.PL_unify_float(t, value)
        elif type(value) == list:
            self._unify_list(t, value)
        elif type(value) == Atom:
            self.swipl.PL_unify_atom(t, value.handle)
        elif type(value) == Functor:
            self._unify_functor(t, value)
        elif type(value) == Term:
            self.swipl.PL_unify(t, value.handle)
        else:
            raise

    def _unify_functor(self, t, value):
        self.swipl.PL_unify_functor(t, value.handle)
        for i in range(value.arity):
            temp = self.swipl.PL_new_term_ref()
            self.swipl.PL_unify_arg(i+1, t, temp)
            if len(value.args) > i:
                self._unifier(temp, value.args[i])

    def _unify_list(self, t, value):
        for arg in value:
            a = self.swipl.PL_new_term_ref()
            self.swipl.PL_unify_list(t, a, t)
            self._unifier(a, arg)
        self.swipl.PL_unify_nil(t)

    def __str__(self):
        if self.chars is not None:
            return self.chars
        else:
            return self.__repr__()

    def __repr__(self):
        return "Variable(%s)" % self.handle

    def put(self, term):
        #PL_put_variable(term)
        self.swipl.PL_put_term(term, self.handle)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.swipl.PL_compare(self.handle, other.handle) == 0

    def __hash__(self):
        return self.handle


class Functor(object):
    __slots__ = "handle", "swipl", "name", "sname", "arity", "args", "__value", "a0"

    def __init__(self, handleOrName, swipl, arity=1, args=None, a0=None):
        """Create a functor.
        ``handleOrName``: functor handle, a string or an atom.
        """

        if "unify" not in swipl.func_names:
            swipl.func_names.add("unify")
            swipl.unify = Functor("=", swipl, arity=2)
            swipl.func[swipl.unify.handle] = swipl.unifier

        self.swipl = swipl
        self.args = args or []
        self.arity = arity

        if isinstance(handleOrName, str):
            self.name = Atom(handleOrName, self.swipl)
            self.handle = self.swipl.PL_new_functor(self.name.handle, arity)
            self.__value = "Functor%d" % self.handle
        elif isinstance(handleOrName, Atom):
            self.name = handleOrName
            self.handle = self.swipl.PL_new_functor(self.name.handle, arity)
            self.__value = "Functor%d" % self.handle
        else:
            self.handle = handleOrName
            self.name = Atom(self.swipl.PL_functor_name(self.handle), self.swipl)
            self.arity = self.swipl.PL_functor_arity(self.handle)
            try:
                self.__value = self.swipl.func[self.handle](self.arity, *self.args)
            except KeyError:
                self.__value = str(self)
        self.a0 = a0

    def fromTerm(cls, term, swipl):
        """Create a functor from a Term or term handle."""

        if isinstance(term, Term):
            term = term.handle
        elif not isinstance(term, (c_void_p, int)):
            raise ArgumentTypeError((str(Term), str(int)), str(type(term)))

        f = functor_t()
        if swipl.PL_get_functor(term, byref(f)):
            # get args
            args = []
            arity = swipl.PL_functor_arity(f.value)
            # let's have all args be consecutive
            a0 = swipl.PL_new_term_refs(arity)
            for i, a in enumerate(range(1, arity + 1)):
                if swipl.PL_get_arg(a, term, a0 + i):
                    args.append(getTerm(a0 + i, swipl))

            return cls(f.value, swipl, args=args, a0=a0)
    fromTerm = classmethod(fromTerm)

    value = property(lambda s: s.__value)

    def __call__(self, *args):
        assert self.arity == len(args)   # FIXME: Put a decent error message
        a = self.swipl.PL_new_term_refs(len(args))
        for i, arg in enumerate(args):
            term_arg = arg
            if type(arg) is Functor:
                term_arg = arg(*arg.args)
            putTerm(a + i, term_arg, self.swipl)

        t = self.swipl.PL_new_term_ref()
        self.swipl.PL_cons_functor_v(t, self.handle, a)

        return Term(self.swipl, t)

    def __str__(self):
        if self.name is not None and self.arity is not None:
            return "%s(%s)" % (self.name,
                               ', '.join([str(arg) for arg in self.args]))
        else:
            return self.__repr__()

    def __repr__(self):
        return "".join(["Functor(", ",".join(str(x) for x
            in [self.handle, self.arity]+self.args), ")"])

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        else:
            return self.swipl.PL_compare(self.handle, other.handle) == 0

    def __hash__(self):
        return self.handle


def putTerm(term, value, swipl):
    if isinstance(value, Term):
        swipl.PL_put_term(term, value.handle)
    elif isinstance(value, str):
        swipl.PL_put_atom_chars(term, value)
    elif isinstance(value, int):
        swipl.PL_put_integer(term, value)
    elif isinstance(value, Constant):
        value.putTerm(term, swipl)
    elif isinstance(value, PList):
        value.putTerm(term, swipl)
    elif isinstance(value, Variable):
        value.put(term)
    elif isinstance(value, list):
        putList(term, value, swipl)
    elif isinstance(value, Atom):
        print("ATOM")
    elif isinstance(value, Functor):
        swipl.PL_put_functor(term, value.handle)
    else:
        raise Exception("Not implemented")


def putList(l, ls, swipl):
    swipl.PL_put_nil(l)
    for item in reversed(ls):
        a = swipl.PL_new_term_ref()  #PL_new_term_refs(len(ls))
        putTerm(a, item, swipl)
        swipl.PL_cons_list(l, a, l)


# deprecated
def getAtomChars(t, swipl):
    """If t is an atom, return it as a string, otherwise raise InvalidTypeError.
    """
    s = c_char_p()
    if swipl.PL_get_atom_chars(t, byref(s)):
        return s.value
    else:
        raise InvalidTypeError("atom")


def getAtom(t, swipl):
    """If t is an atom, return it , otherwise raise InvalidTypeError.
    """
    return Atom.fromTerm(t, swipl)


def getBool(t, swipl):
    """If t is of type bool, return it, otherwise raise InvalidTypeError.
    """
    b = c_int()
    if swipl.PL_get_long(t, byref(b)):
        return bool(b.value)
    else:
        raise InvalidTypeError("bool")


def getLong(t, swipl):
    """If t is of type long, return it, otherwise raise InvalidTypeError.
    """
    i = c_long()
    if swipl.PL_get_long(t, byref(i)):
        return i.value
    else:
        raise InvalidTypeError("long")


getInteger = getLong  # just an alias for getLong


def getFloat(t, swipl):
    """If t is of type float, return it, otherwise raise InvalidTypeError.
    """
    d = c_double()
    if swipl.PL_get_float(t, byref(d)):
        return d.value
    else:
        raise InvalidTypeError("float")


def getString(t, swipl):
    """If t is of type string, return it, otherwise raise InvalidTypeError.
    """
    slen = c_int()
    s = c_char_p()
    if swipl.PL_get_string_chars(t, byref(s), byref(slen)):
        return s.value
    else:
        raise InvalidTypeError("string")


mappedTerms = {}
def getTerm(t, swipl):
    if t is None:
        return None
    global mappedTerms
    #print 'mappedTerms', mappedTerms

    #if t in mappedTerms:
    #    return mappedTerms[t]
    p = swipl.PL_term_type(t)
    if p < PL_TERM:
        res = _getterm_router[p](t, swipl)
    elif swipl.PL_is_list(t):
        res = getList(t, swipl)
    else:
        res = getFunctor(t, swipl)
    mappedTerms[t] = res
    return res


def getList(x, swipl):
    """
    Return t as a list.
    """

    t = swipl.PL_copy_term_ref(x)
    head = swipl.PL_new_term_ref()
    result = []
    while swipl.PL_get_list(t, head, t):
        result.append(getTerm(head, swipl))
        head = swipl.PL_new_term_ref()

    return result


def getFunctor(t, swipl):
    """Return t as a functor
    """
    return Functor.fromTerm(t, swipl)


def getVariable(t, swipl):
    return Variable(t, swipl)


_getterm_router = {
                   PL_VARIABLE: getVariable,
                   PL_ATOM: getAtom,
                   PL_STRING: getString,
                   PL_INTEGER: getInteger,
                   PL_FLOAT: getFloat,
                   PL_TERM: getTerm,
                  }

arities = {}


def _callbackWrapper(arity=1, nondeterministic=False):
    global arities

    res = arities.get((arity, nondeterministic))
    if res is None:
        if nondeterministic:
            res = CFUNCTYPE(*([foreign_t] + [term_t] * arity + [control_t]))
        else:
            res = CFUNCTYPE(*([foreign_t] + [term_t] * arity))
        arities[(arity, nondeterministic)] = res
    return res


funwraps = {}


def _foreignWrapper(fun, swipl, nondeterministic=False):
    global funwraps

    res = funwraps.get(fun)
    if res is None:
        def wrapper(*args):
            if nondeterministic:
                args = [getTerm(arg, swipl) for arg in args[:-1]] + [args[-1]]
            else:
                args = [getTerm(arg, swipl) for arg in args]
            r = fun(*args)
            return (r is None) and True or r

        res = wrapper
        funwraps[fun] = res
    return res


cwraps = []


def registerForeign(func, swipl, name=None, arity=None, flags=0):
    """Register a Python predicate
    ``func``: Function to be registered. The function should return a value in
    ``foreign_t``, ``True`` or ``False``.
    ``name`` : Name of the function. If this value is not used, ``func.func_name``
    should exist.
    ``arity``: Arity (number of arguments) of the function. If this value is not
    used, ``func.arity`` should exist.
    """
    global cwraps

    if arity is None:
        arity = func.arity

    if name is None:
        name = func.__name__

    nondeterministic = bool(flags & PL_FA_NONDETERMINISTIC)

    cwrap = _callbackWrapper(arity, nondeterministic)
    fwrap = _foreignWrapper(func, swipl, nondeterministic)
    fwrap2 = cwrap(fwrap)
    cwraps.append(fwrap2)
    return swipl.PL_register_foreign(name, arity, fwrap2, flags)
    # return PL_register_foreign(name, arity,
    #            _callbackWrapper(arity)(_foreignWrapper(func)), flags)


# newTermRef = PL_new_term_ref


def newTermRefs(count, swipl):
    a = swipl.PL_new_term_refs(count)
    return list(range(a, a + count))


def call(swipl, *terms, **kwargs):
    """Call term in module.
    ``term``: a Term or term handle
    """
    for kwarg in kwargs:
        if kwarg not in ["module"]:
            raise KeyError

    module = kwargs.get("module", None)

    t = terms[0]
    for tx in terms[1:]:
        t = Functor(",", swipl, 2)(t, tx)

    return swipl.PL_call(t.handle, module)


def newModule(name, swipl):
    """Create a new module.
    ``name``: An Atom or a string
    """
    if isinstance(name, str):
        name = Atom(name, swipl)

    return swipl.PL_new_module(name.handle)


class Query(object):
    qid = None
    fid = None

    def __init__(self, swipl, *terms, **kwargs):
        self.swipl = swipl
        for key in kwargs:
            if key not in ["flags", "module"]:
                raise Exception("Invalid kwarg: %s" % key, key)

        flags = kwargs.get("flags", PL_Q_NODEBUG|PL_Q_CATCH_EXCEPTION)
        module = kwargs.get("module", None)

        t = terms[0]
        for tx in terms[1:]:
            t = Functor(",", swipl, 2)(t, tx)

        f = Functor.fromTerm(t, swipl)
        p = swipl.PL_pred(f.handle, module)
        Query.qid = swipl.PL_open_query(module, flags, p, f.a0)

#    def __del__(self):
#        self.closeQuery()

    def nextSolution(swipl):
        return swipl.PL_next_solution(Query.qid)
    nextSolution = staticmethod(nextSolution)

    def cutQuery(swipl):
        swipl.PL_cut_query(Query.qid)
    cutQuery = staticmethod(cutQuery)

    def closeQuery(swipl):
        if Query.qid is not None:
            swipl.PL_close_query(Query.qid)
            Query.qid = None
    closeQuery = staticmethod(closeQuery)
