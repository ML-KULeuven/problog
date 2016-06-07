"""
problog.engine_builtin - Grounding engine builtins
--------------------------------------------------

Implementation of Prolog / ProbLog builtins.

..
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

from .logic import term2str, Term, Clause, Constant, term2list, list2term, is_ground, is_variable
from .program import PrologFile
from .errors import GroundingError, UserError
from .engine_unify import unify_value, UnifyError, substitute_simple
from .engine import UnknownClauseInternal, UnknownClause

import os
import imp  # For load_external
import inspect  # For load_external


def add_standard_builtins(engine, b=None, s=None, sp=None):
    """Adds standard builtins to the given engine.

    :param engine: engine to add builtins to
    :type engine: ClauseDBEngine
    :param b: wrapper for boolean builtins (returning True/False)
    :param s: wrapper for simple builtins (return deterministic results)
    :param sp: wrapper for probabilistic builtins (return probabilistic results)
    """

    engine.add_builtin('true', 0, b(_builtin_true))  # -1
    engine.add_builtin('fail', 0, b(_builtin_fail))  # -2
    engine.add_builtin('false', 0, b(_builtin_fail))  # -3

    engine.add_builtin('=', 2, s(_builtin_eq))  # -4
    engine.add_builtin('\=', 2, b(_builtin_neq))  # -5

    engine.add_builtin('findall', 3, sp(_builtin_findall))  # -6
    engine.add_builtin('all', 3, sp(_builtin_all))  # -7
    engine.add_builtin('all_or_none', 3, sp(_builtin_all_or_none))  # -8

    engine.add_builtin('==', 2, b(_builtin_same))
    engine.add_builtin('\==', 2, b(_builtin_notsame))

    engine.add_builtin('is', 2, s(_builtin_is))

    engine.add_builtin('>', 2, b(_builtin_gt))
    engine.add_builtin('<', 2, b(_builtin_lt))
    engine.add_builtin('=<', 2, b(_builtin_le))
    engine.add_builtin('>=', 2, b(_builtin_ge))
    engine.add_builtin('=\=', 2, b(_builtin_val_neq))
    engine.add_builtin('=:=', 2, b(_builtin_val_eq))

    engine.add_builtin('var', 1, b(_builtin_var))
    engine.add_builtin('atom', 1, b(_builtin_atom))
    engine.add_builtin('atomic', 1, b(_builtin_atomic))
    engine.add_builtin('compound', 1, b(_builtin_compound))
    engine.add_builtin('float', 1, b(_builtin_float))
    engine.add_builtin('rational', 1, b(_builtin_rational))
    engine.add_builtin('integer', 1, b(_builtin_integer))
    engine.add_builtin('nonvar', 1, b(_builtin_nonvar))
    engine.add_builtin('number', 1, b(_builtin_number))
    engine.add_builtin('simple', 1, b(_builtin_simple))
    engine.add_builtin('callable', 1, b(_builtin_callable))
    engine.add_builtin('dbreference', 1, b(_builtin_dbreference))
    engine.add_builtin('primitive', 1, b(_builtin_primitive))
    engine.add_builtin('ground', 1, b(_builtin_ground))
    engine.add_builtin('is_list', 1, b(_builtin_is_list))

    engine.add_builtin('=..', 2, s(_builtin_split_call))
    engine.add_builtin('arg', 3, s(_builtin_arg))
    engine.add_builtin('functor', 3, s(_builtin_functor))

    engine.add_builtin('@>', 2, b(_builtin_struct_gt))
    engine.add_builtin('@<', 2, b(_builtin_struct_lt))
    engine.add_builtin('@>=', 2, b(_builtin_struct_ge))
    engine.add_builtin('@=<', 2, b(_builtin_struct_le))
    engine.add_builtin('compare', 3, s(_builtin_compare))

    engine.add_builtin('length', 2, s(_builtin_length))
    # engine.add_builtin('call_external', 2, s(_builtin_call_external))

    engine.add_builtin('sort', 2, s(_builtin_sort))
    engine.add_builtin('between', 3, s(_builtin_between))
    engine.add_builtin('succ', 2, s(_builtin_succ))
    engine.add_builtin('plus', 3, s(_builtin_plus))

    engine.add_builtin('consult', 1, b(_builtin_consult))
    engine.add_builtin('.', 2, b(_builtin_consult_as_list))
    # engine.add_builtin('load_external', 1, b(_builtin_load_external))
    engine.add_builtin('unknown', 1, b(_builtin_unknown))

    engine.add_builtin('use_module', 1, b(_builtin_use_module))

    engine.add_builtin('call', 1, _builtin_call)
    for i in range(2, 10):
        engine.add_builtin('call', i, _builtin_calln)

    engine.add_builtin('subquery', 2, s(_builtin_subquery))
    engine.add_builtin('subquery', 3, s(_builtin_subquery))

    engine.add_builtin('sample_uniform1', 3, sp(_builtin_sample_uniform))

    for i in range(1, 10):
        engine.add_builtin('debugprint', i, b(_builtin_debugprint))

    for i in range(1, 10):
        engine.add_builtin('write', i, b(_builtin_write))

    for i in range(1, 10):
        engine.add_builtin('writenl', i, b(_builtin_writenl))

    for i in range(1, 10):
        engine.add_builtin('error', i, b(_builtin_error))

    engine.add_builtin('nl', 0, b(_builtin_nl))
    engine.add_builtin('cmd_args', 1, s(_builtin_cmdargs))
    engine.add_builtin('atom_number', 2, s(_builtin_atom_number))
    engine.add_builtin('nocache', 2, b(_builtin_nocache))


def _builtin_nocache(functor, arity, database=None, **kwd):
    check_mode((functor, arity), ['ai'], **kwd)
    database.dont_cache.add((str(functor), int(arity)))
    return True


def _builtin_cmdargs(lst, engine=None, **kwd):
    m = check_mode((lst,), ['v', 'L'], **kwd)
    args = engine.args
    if args is None:
        args = []
    args = list2term(list(map(Term, args)))
    if m == 0:
        return [(args,)]
    else:
        try:
            value = unify_value(args, lst, {})
            return [(value,)]
        except UnifyError:
            return []


def _builtin_atom_number(atom, number, **kwd):
    mode = check_mode((atom, number), ['vf', 'vi', 'av', 'af', 'ai'], **kwd)
    if mode in (0, 1):
        return [(Term(str(number)), number)]
    elif mode == 2:
        try:
            v = float(atom.functor)
        except ValueError:
            return []   # fail silently
            # raise GroundingError('Atom does not represent a number: \'%s\'' % atom)

        if round(v) == v:
            v = Constant(int(v))
        else:
            v = Constant(v)
        return [(atom, v)]
    else:
        if atom == str(number):
            return [(atom, number)]
        else:
            return []


# noinspection PyUnusedLocal
def _builtin_debugprint(*args, **kwd):
    print(' '.join(map(term2str, args)))
    return True


def term2str_noquote(term):
    res = term2str(term)
    if res[0] == res[-1] == "'":
        res = res[1:-1]
    return res

def _builtin_write(*args, **kwd):
    print(' '.join(map(term2str_noquote, args)), end='')
    return True


def _builtin_error(*args, **kwd):
    location = kwd.get('call_origin', (None, None))[1]
    database = kwd['database']
    location = database.lineno(location)
    message = ''.join(map(term2str_noquote, args))
    raise UserError(message, location=location)


def _builtin_writenl(*args, **kwd):
    print(' '.join(map(term2str_noquote, args)))
    return True

def _builtin_nl(**kwd):
    print()
    return True



class CallModeError(GroundingError):
    """
    Represents an error in builtin argument types.
    """

    def __init__(self, functor, args, accepted=None, message=None, location=None):
        if accepted is None:
            accepted = []
        if functor:
            self.scope = '%s/%s' % (functor, len(args))
        else:
            self.scope = None
        self.received = ', '.join(map(self._show_arg, args))
        self.expected = [', '.join(map(self._show_mode, mode)) for mode in accepted]
        msg = 'Invalid argument types for call'
        if self.scope:
            msg += " to '%s'" % self.scope
        msg += ': arguments: (%s)' % self.received
        if accepted:
            msg += ', expected: (%s)' % ') or ('.join(self.expected)
        else:
            msg += ', expected: ' + message
        GroundingError.__init__(self, msg, location)

    def _show_arg(self, x):
        return term2str(x)

    def _show_mode(self, t):
        return mode_types[t][0]


class StructSort(object):
    """
    Comparator of terms based on structure.
    """

    # noinspection PyUnusedLocal
    def __init__(self, obj, *args):
        self.obj = obj

    def __lt__(self, other):
        return struct_cmp(self.obj, other.obj) < 0

    def __gt__(self, other):
        return struct_cmp(self.obj, other.obj) > 0

    def __eq__(self, other):
        return struct_cmp(self.obj, other.obj) == 0

    def __le__(self, other):
        return struct_cmp(self.obj, other.obj) <= 0

    def __ge__(self, other):
        return struct_cmp(self.obj, other.obj) >= 0

    def __ne__(self, other):
        return struct_cmp(self.obj, other.obj) != 0


def _is_var(term):
    return is_variable(term) or term.is_var()


def _is_nonvar(term):
    return not _is_var(term)


def _is_term(term):
    return not _is_var(term) and not _is_constant(term)


def _is_float_pos(term):
    return _is_constant(term) and term.is_float()


def _is_float_neg(term):
    return _is_term(term) and term.arity == 1 and term.functor == "'-'" and \
        _is_float_pos(term.args[0])


def _is_float(term):
    return _is_float_pos(term) or _is_float_neg(term)


def _is_integer_pos(term):
    return _is_constant(term) and term.is_integer()


def _is_integer_neg(term):
    return _is_term(term) and term.arity == 1 and term.functor == "'-'" and \
        _is_integer_pos(term.args[0])


def _is_integer(term):
    return _is_integer_pos(term) or _is_integer_neg(term)


def _is_string(term):
    return _is_constant(term) and term.is_string()


def _is_number(term):
    return _is_float(term) or _is_integer(term)


def _is_constant(term):
    return not _is_var(term) and term.is_constant()


def _is_atom(term):
    return _is_term(term) and term.arity == 0


def _is_atomic(term):
    return _is_nonvar(term) and not _is_compound(term)


# noinspection PyUnusedLocal
def _is_rational(term):
    return False


# noinspection PyUnusedLocal
def _is_dbref(term):
    return False


def _is_compound(term):
    return _is_term(term) and term.arity > 0


def _is_list_maybe(term):
    """
    Check whether the term looks like a list (i.e. of the form '.'(_,_)).
    :param term:
    :return:
    """
    return _is_compound(term) and term.functor == '.' and term.arity == 2


def _is_list_nonempty(term):
    if _is_list_maybe(term):
        tail = list_tail(term)
        return _is_list_empty(tail) or _is_var(tail)
    return False


def _is_fixed_list(term):
    return _is_list_empty(term) or _is_fixed_list_nonempty(term)


def _is_fixed_list_nonempty(term):
    if _is_list_maybe(term):
        tail = list_tail(term)
        return _is_list_empty(tail)
    return False


def _is_list_empty(term):
    return _is_atom(term) and term.functor == '[]'


def _is_list(term):
    return _is_list_empty(term) or _is_list_nonempty(term)


def _is_compare(term):
    return _is_atom(term) and term.functor in ("'<'", "'='", "'>'")


mode_types = {
    'i': ('integer', _is_integer),
    'I': ('positive_integer', _is_integer_pos),
    'f': ('float', _is_float),
    'v': ('var', _is_var),
    'n': ('nonvar', _is_nonvar),
    'l': ('list', _is_list),
    'L': ('fixed_list', _is_fixed_list),  # List of fixed length (i.e. tail is [])
    '*': ('any', lambda x: True),
    '<': ('compare', _is_compare),  # < = >
    'g': ('ground', is_ground),
    'a': ('atom', _is_atom),
    'c': ('callable', _is_term)
}


# noinspection PyUnusedLocal
def check_mode(args, accepted, functor=None, location=None, database=None, **kwdargs):
    """Checks the arguments against a list of accepted types.

    :param args: arguments to check
    :type args: tuple of Term
    :param accepted: list of accepted combination of types (see mode_types)
    :type accepted: list of str
    :param functor: functor of the call (used for error message)
    :param location: location of the call (used for error message)
    :param database: database (used for error message)
    :param kwdargs: additional arguments (not used)
    :return: the index of the first mode in accepted that matches the arguments
    :rtype: int
    """
    for i, mode in enumerate(accepted):
        correct = True
        for a, t in zip(args, mode):
            name, test = mode_types[t]
            if not test(a):
                correct = False
                break
        if correct:
            return i
    if database and location:
        location = database.lineno(location)
    else:
        location = None
    raise CallModeError(functor, args, accepted, location=location)


def list_elements(term):
    """Extract elements from a List term.
    Ignores the list tail.

    :param term: term representing a list
    :type term: Term
    :return: elements of the list
    :rtype: list of Term
    """
    elements = []
    tail = term
    while _is_list_maybe(tail):
        elements.append(tail.args[0])
        tail = tail.args[1]
    return elements, tail


def list_tail(term):
    """Extract the tail of the list.

    :param term: Term representing a list
    :type term: Term
    :return: tail of the list
    :rtype: Term
    """
    tail = term
    while _is_list_maybe(tail):
        tail = tail.args[1]
    return tail


def _builtin_split_call(term, parts, database=None, location=None, **kwdargs):
    """Implements the '=..' builtin operator.

    :param term:
    :param parts:
    :param database:
    :param location:
    :param kwdargs:
    :return:
    """
    functor = '=..'
    # modes:
    #   <v> =.. list  => list has to be fixed length and non-empty
    #                       IF its length > 1 then first element should be an atom
    #   <n> =.. <list or var>
    #
    mode = check_mode((term, parts), ['vL', 'nv', 'nl'], functor=functor, **kwdargs)
    if mode == 0:
        elements, tail = list_elements(parts)
        if len(elements) == 0:
            raise CallModeError(functor, (term, parts),
                                message='non-empty list for arg #2 if arg #1 is a variable',
                                location=database.lineno(location))
        elif len(elements) > 1 and not _is_atom(elements[0]):
            raise CallModeError(functor, (term, parts),
                                message='atom as first element in list if arg #1 is a variable',
                                location=database.lineno(location))
        elif len(elements) == 1:
            # Special case => term == parts[0]
            return [(elements[0], parts)]
        else:
            term_part = elements[0](*elements[1:])
            return [(term_part, parts)]
    else:
        part_list = (term.with_args(),) + term.args
        current = Term('[]')
        for t in reversed(part_list):
            current = Term('.', t, current)
        try:
            local_values = {}
            list_part = unify_value(current, parts, local_values)
            elements, tail = list_elements(list_part)
            term_new = elements[0](*elements[1:])
            term_part = unify_value(term, term_new, local_values)
            return [(term_part, list_part)]
        except UnifyError:
            return []


def _builtin_arg(index, term, arguments, **kwdargs):
    check_mode((index, term, arguments), ['In*'], functor='arg', **kwdargs)
    index_v = int(index) - 1
    if 0 <= index_v < len(term.args):
        try:
            arg = term.args[index_v]
            res = unify_value(arg, arguments, {})
            return [(index, term, res)]
        except UnifyError:
            pass
    return []


def _builtin_functor(term, functor, arity, **kwdargs):
    mode = check_mode((term, functor, arity), ['vaI', 'n**'], functor='functor', **kwdargs)

    if mode == 0:
        kwdargs.get('callback').newResult(Term(functor, *((None,) * int(arity))), functor, arity)
    else:
        try:
            values = {}
            func_out = unify_value(functor, Term(term.functor), values)
            arity_out = unify_value(arity, Constant(term.arity), values)
            return [(term, func_out, arity_out)]
        except UnifyError:
            pass
    return []


# noinspection PyUnusedLocal
def _builtin_true(**kwdargs):
    """``true``"""
    return True


# noinspection PyUnusedLocal
def _builtin_fail(**kwdargs):
    """``fail``"""
    return False


# noinspection PyUnusedLocal
def _builtin_eq(arg1, arg2, **kwdargs):
    """``A = B``
        A and B not both variables
    """
    try:
        result = unify_value(arg1, arg2, {})
        return [(result, result)]
    except UnifyError:
        return []
        # except VariableUnification:
        #     raise VariableUnification(location = database.lineno(location))


# noinspection PyUnusedLocal
def _builtin_neq(arg1, arg2, **kwdargs):
    """``A \= B``
        A and B not both variables
    """
    try:
        unify_value(arg1, arg2, {})
        return False
    except UnifyError:
        return True


# noinspection PyUnusedLocal
def _builtin_notsame(arg1, arg2, **kwdargs):
    """``A \== B``"""
    return not arg1 == arg2


# noinspection PyUnusedLocal
def _builtin_same(arg1, arg2, **kwdargs):
    """``A == B``"""
    return arg1 == arg2


def _builtin_gt(arg1, arg2, engine=None, **kwdargs):
    """``A > B``
        A and B are ground
    """
    check_mode((arg1, arg2), ['gg'], functor='>', **kwdargs)
    a_value = arg1.compute_value(engine.functions)
    b_value = arg2.compute_value(engine.functions)
    if a_value is None or b_value is None:
        return False
    else:
        return a_value > b_value


def _builtin_lt(arg1, arg2, engine=None, **kwdargs):
    """``A > B``
        A and B are ground
    """
    check_mode((arg1, arg2), ['gg'], functor='<', **kwdargs)
    a_value = arg1.compute_value(engine.functions)
    b_value = arg2.compute_value(engine.functions)
    if a_value is None or b_value is None:
        return False
    else:
        return a_value < b_value


def _builtin_le(arg1, arg2, engine=None, **k):
    """``A =< B``
        A and B are ground
    """
    check_mode((arg1, arg2), ['gg'], functor='=<', **k)
    a_value = arg1.compute_value(engine.functions)
    b_value = arg2.compute_value(engine.functions)
    if a_value is None or b_value is None:
        return False
    else:
        return a_value <= b_value


def _builtin_ge(arg1, arg2, engine=None, **k):
    """``A >= B``
        A and B are ground
    """
    check_mode((arg1, arg2), ['gg'], functor='>=', **k)
    a_value = arg1.compute_value(engine.functions)
    b_value = arg2.compute_value(engine.functions)
    if a_value is None or b_value is None:
        return False
    else:
        return a_value >= b_value


def _builtin_val_neq(a, b, engine=None, **k):
    """``A =\= B``
        A and B are ground
    """
    check_mode((a, b), ['gg'], functor='=\=', **k)
    a_value = a.compute_value(engine.functions)
    b_value = b.compute_value(engine.functions)
    if a_value is None or b_value is None:
        return False
    else:
        return a_value != b_value


def _builtin_val_eq(a, b, engine=None, **k):
    """``A =:= B``
        A and B are ground
    """
    check_mode((a, b), ['gg'], functor='=:=', **k)
    a_value = a.compute_value(engine.functions)
    b_value = b.compute_value(engine.functions)
    if a_value is None or b_value is None:
        return False
    else:
        return a_value == b_value


def _builtin_is(a, b, engine=None, **k):
    """``A is B``
        B is ground

        @param a:
        @param b:
        @param engine:
        @param k:
    """
    check_mode((a, b), ['*g'], functor='is', **k)
    try:
        b_value = b.compute_value(engine.functions)
        if b_value is None:
            return []
        else:
            r = Constant(b_value)
            unify_value(a, r, {})
            return [(r, b)]
    except UnifyError:
        return []


# noinspection PyUnusedLocal
def _builtin_var(term, **k):
    return _is_var(term)


# noinspection PyUnusedLocal
def _builtin_atom(term, **k):
    return _is_atom(term)


# noinspection PyUnusedLocal
def _builtin_atomic(term, **k):
    return _is_atom(term) or _is_number(term)


# noinspection PyUnusedLocal
def _builtin_compound(term, **k):
    return _is_compound(term)


# noinspection PyUnusedLocal
def _builtin_float(term, **k):
    return _is_float(term)


# noinspection PyUnusedLocal
def _builtin_integer(term, **k):
    return _is_integer(term)


# noinspection PyUnusedLocal
def _builtin_nonvar(term, **k):
    return not _is_var(term)


# noinspection PyUnusedLocal
def _builtin_number(term, **k):
    return _is_number(term)


# noinspection PyUnusedLocal
def _builtin_simple(term, **k):
    return _is_var(term) or _is_atomic(term)


# noinspection PyUnusedLocal
def _builtin_callable(term, **k):
    return _is_term(term)


# noinspection PyUnusedLocal
def _builtin_rational(term, **k):
    return _is_rational(term)


# noinspection PyUnusedLocal
def _builtin_dbreference(term, **k):
    return _is_dbref(term)


# noinspection PyUnusedLocal
def _builtin_primitive(term, **k):
    return _is_atomic(term) or _is_dbref(term)


# noinspection PyUnusedLocal
def _builtin_ground(term, **k):
    return is_ground(term)


# noinspection PyUnusedLocal
def _builtin_is_list(term, **k):
    return _is_list(term)


def compare(a, b):
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


def struct_cmp(a, b):
    # Note: structural comparison
    # 1) Var < Num < Str < Atom < Compound
    # 2) Var by address
    # 3) Number by value, if == between int and float => float is smaller
    #   (iso prolog: Float always < Integer )
    # 4) String alphabetical
    # 5) Atoms alphabetical
    # 6) Compound: arity / functor / arguments

    # 1) Variables are smallest
    if _is_var(a):
        if _is_var(b):
            # 2) Variable by address
            return compare(a, b)
        else:
            return -1
    elif _is_var(b):
        return 1
    # assert( not is_var(A) and not is_var(B) )

    # 2) Numbers are second smallest
    if _is_number(a):
        if _is_number(b):
            # Just compare numbers on float value
            res = compare(float(a), float(b))
            if res == 0:
                # If the same, float is smaller.
                if _is_float(a) and _is_integer(b):
                    return -1
                elif _is_float(b) and _is_integer(a):
                    return 1
                else:
                    return 0
        else:
            return -1
    elif _is_number(b):
        return 1

    # 3) Strings are third
    if _is_string(a):
        if _is_string(b):
            return compare(str(a), str(b))
        else:
            return -1
    elif _is_string(b):
        return 1

    # 4) Atoms / terms come next
    # 4.1) By arity
    res = compare(a.arity, b.arity)
    if res != 0:
        return res

    # 4.2) By functor
    res = compare(a.functor, b.functor)
    if res != 0:
        return res

    # 4.3) By arguments (recursively)
    for a1, b1 in zip(a.args, b.args):
        res = struct_cmp(a1, b1)
        if res != 0:
            return res

    return 0


# noinspection PyUnusedLocal
def _builtin_struct_lt(a, b, **k):
    return struct_cmp(a, b) < 0


# noinspection PyUnusedLocal
def _builtin_struct_le(a, b, **k):
    return struct_cmp(a, b) <= 0


# noinspection PyUnusedLocal
def _builtin_struct_gt(a, b, **k):
    return struct_cmp(a, b) > 0


# noinspection PyUnusedLocal
def _builtin_struct_ge(a, b, **k):
    return struct_cmp(a, b) >= 0


def _builtin_compare(c, a, b, **k):
    mode = check_mode((c, a, b), ['<**', 'v**'], functor='compare', **k)
    compares = "'>'", "'='", "'<'"
    cp = struct_cmp(a, b)
    c_token = compares[1 - cp]

    if mode == 0:  # Given compare
        if c_token == c.functor:
            return [(c, a, b)]
    else:  # Unknown compare
        return [(Term(c_token), a, b)]


# numbervars(T,+N1,-Nn)    number the variables TBD?

def build_list(elements, tail):
    current = tail
    for el in reversed(elements):
        current = Term('.', el, current)
    return current


# class UnknownExternal(GroundingError):
#     """Undefined clause in call."""
#
#     def __init__(self, signature, location):
#         GroundingError.__init__(self, "Unknown external function '%s'" % signature, location)


# def _builtin_call_external(call, result, database=None, location=None, **k):
#     from . import pypl
#     check_mode((call, result), ['gv'], function='call_external', database=database,
#                location=location, **k)
#
#     func = k['engine'].get_external_call(call.functor)
#     if func is None:
#         raise UnknownExternal(call.functor, database.lineno(location))
#
#     values = [pypl.pl2py(arg) for arg in call.args]
#     computed_result = func(*values)
#
#     return [(call, pypl.py2pl(computed_result))]


def _builtin_length(l, n, **k):
    mode = check_mode((l, n), ['LI', 'Lv', 'lI', 'vI'], functor='length', **k)
    # Note that Prolog also accepts 'vv' and 'lv', but these are unbounded.
    # Note that lI is a subset of LI, but only first matching mode is returned.
    if mode == 0 or mode == 1:  # Given fixed list and maybe length
        elements, tail = list_elements(l)
        list_size = len(elements)
        try:
            n = unify_value(n, Constant(list_size), {})
            return [(l, n)]
        except UnifyError:
            return []
    else:  # Unbounded list or variable list and fixed length.
        if mode == 2:
            elements, tail = list_elements(l)
        else:
            elements, tail = [], l
        remain = int(n) - len(elements)
        if remain < 0:
            raise UnifyError()
        else:
            min_var = k.get('engine')._context_min_var(k.get('context'))
            extra = list(range(min_var, min_var - remain, -1))  # [None] * remain
        new_l = build_list(elements + extra, Term('[]'))
        return [(new_l, n)]


def _builtin_sort(l, s, **k):
    # TODO doesn't work properly with variables e.g. gives sort([X,Y,Y],[_])
    # should be sort([X,Y,Y],[X,Y])
    check_mode((l, s), ['L*'], functor='sort', **k)
    elements, tail = list_elements(l)
    # assert( is_list_empty(tail) )
    try:
        sorted_list = build_list(sorted(set(elements), key=StructSort), Term('[]'))
        s_out = unify_value(s, sorted_list, {})
        return [(l, s_out)]
    except UnifyError:
        return []


def _builtin_between(low, high, value, **k):
    """
    Implements the between/3 builtin.
   :param low:
   :param high:
   :param value:
   :param k:
   :return:
    """
    mode = check_mode((low, high, value), ['iii', 'iiv'], functor='between', **k)
    low_v = int(low)
    high_v = int(high)
    if mode == 0:  # Check
        value_v = int(value)
        if low_v <= value_v <= high_v:
            return [(low, high, value)]
    else:  # Enumerate
        results = []
        for value_v in range(low_v, high_v + 1):
            results.append((low, high, Constant(value_v)))
        return results


def _builtin_succ(a, b, **kwdargs):
    """
    Implements the succ/2 builtin.
   :param a: input argument
   :param b: output argument
   :param kwdargs: additional arguments
   :return:
    """
    mode = check_mode((a, b), ['vI', 'Iv', 'II'], functor='succ', **kwdargs)
    if mode == 0:
        b_v = int(b)
        return [(Constant(b_v - 1), b)]
    elif mode == 1:
        a_v = int(a)
        return [(a, Constant(a_v + 1))]
    else:
        a_v = int(a)
        b_v = int(b)
        if b_v == a_v + 1:
            return [(a, b)]
    return []


def _builtin_plus(a, b, c, **kwdargs):
    """
    Implements the plus/3 builtin.
   :param a: first argument
   :param b: second argument
   :param c: result argument
   :param kwdargs: additional arguments
   :return:
    """
    mode = check_mode((a, b, c), ['iii', 'iiv', 'ivi', 'vii'], functor='plus', **kwdargs)
    if mode == 0:
        a_v = int(a)
        b_v = int(b)
        c_v = int(c)
        if a_v + b_v == c_v:
            return [(a, b, c)]
    elif mode == 1:
        a_v = int(a)
        b_v = int(b)
        return [(a, b, Constant(a_v + b_v))]
    elif mode == 2:
        a_v = int(a)
        c_v = int(c)
        return [(a, Constant(c_v - a_v), c)]
    else:
        b_v = int(b)
        c_v = int(c)
        return [(Constant(c_v - b_v), b, c)]
    return []


def _atom_to_filename(atom):
    """Translate an atom to a filename.

   :param atom: filename as atom
   :type atom: Term
   :return: filename as string
   :rtype: str
    """
    atomstr = str(atom)
    if atomstr[0] == atomstr[-1] == "'":
        atomstr = atomstr[1:-1]
    return atomstr


class ConsultError(GroundingError):
    """Error during consult"""

    def __init__(self, message, location):
        GroundingError.__init__(self, message, location)


def _builtin_consult_as_list(op1, op2, **kwdargs):
    """Implementation of consult/1 using list notation.

   :param op1: first element in the list
   :param op2: tail of the list
   :param kwdargs: additional arugments
   :return: True
    """
    # TODO make non-recursive
    check_mode((op1, op2), ['*L'], functor='consult', **kwdargs)
    _builtin_consult(op1, **kwdargs)
    if _is_list_nonempty(op2):
        _builtin_consult_as_list(op2.args[0], op2.args[1], **kwdargs)
    return True


def _builtin_consult(filename, database=None, engine=None, **kwdargs):
    """
    Implementation of consult/1 builtin.
    A file will be loaded only once.
   :param filename: filename to load into the database
   :type filename: Term
   :param database: database containing the current logic program.
   :param kwdargs: additional arguments
   :return: True
    """

    root = database.source_root
    if filename.location:
        source_root = database.source_files[filename.location[0]]
        if source_root:
            root = os.path.dirname(source_root)
    check_mode((filename,), ['a'], functor='consult', **kwdargs)
    filename = os.path.join(root, _atom_to_filename(filename))
    if not os.path.exists(filename):
        filename += '.pl'
    if not os.path.exists(filename):
        raise ConsultError(message="Consult: file not found '%s'" % filename,
                           location=database.lineno(kwdargs.get('location')))

    # Prevent loading the same file twice
    if filename not in database.source_files:
        identifier = len(database.source_files)
        database.source_files.append(filename)
        database.source_parent.append(kwdargs.get('location'))
        pl = PrologFile(filename, identifier=identifier)
        database.line_info.append(pl.line_info[0])
        for clause in pl:
            database += clause
        # engine._process_directives(database)

    return True


# # noinspection PyUnusedLocal
# def _builtin_load_external(arg, engine=None, database=None, location=None, **kwdargs):
#     check_mode((arg,), ['a'], functor='load_external')
#     # Load external (python) files that are referenced in the model
#     externals = {}
#     root = database.source_root
#     if arg.location:
#         root = os.path.dirname(database.source_files[arg.location[0]])
#     filename = os.path.join(root, _atom_to_filename(arg))
#     if not os.path.exists(filename):
#         raise ConsultError(message="Load external: file not found '%s'" % filename,
#                            location=database.lineno(location))
#     try:
#         with open(filename, 'r') as extfile:
#             ext = imp.load_module('externals', extfile, filename, ('.py', 'U', 1))
#             for func_name, func in inspect.getmembers(ext, inspect.isfunction):
#                 externals[func_name] = func
#         engine.add_external_calls(externals)
#     except ImportError:
#         raise ConsultError(message="Error while loading external file '%s'" % filename,
#                            location=database.lineno(location))
#
#     return True


# noinspection PyUnusedLocal
def _builtin_unknown(arg, engine=None, **kwdargs):
    check_mode((arg,), ['a'], functor='unknown')
    if arg.functor == 'fail':
        engine.unknown = engine.UNKNOWN_FAIL
    else:
        engine.unknown = engine.UNKNOWN_ERROR
    return True


def _select_sublist(lst, target):
    """
    Enumerate all possible selection of elements from a list.
    This function is used to generate all solutions to findall/3.
    An element must be selected if it is TRUE in the target formula.

    :param lst: list to select elements from
    :type lst: list of tuple
    :param target: data structure containing truth value of nodes
    :type target: LogicFormula
    :return: generator of sublists
    """
    l = len(lst)

    # Generate an array that indicates the decision bit for each element in the list.
    # If an element is deterministically true, then no decision bit is needed.
    choice_bits = [None] * l
    x = 0
    for i in range(0, l):
        if lst[i][1] not in (target.TRUE, target.FALSE):
            choice_bits[i] = x
            x += 1

    # We have 2^x distinct lists. Each can be represented as a number between 0 and 2^x-1=n.
    n = (1 << x) - 1

    while n >= 0:
        # Generate the list of positive values and node identifiers
        # noinspection PyTypeChecker
        sublist = [lst[i] for i in range(0, l)
                   if (choice_bits[i] is None and lst[i][1] == target.TRUE) or (choice_bits[i] is not None and n & 1 << choice_bits[i])]
        # Generate the list of negative node identifiers
        # noinspection PyTypeChecker
        sublist_no = tuple([target.negate(lst[i][1]) for i in range(0, l)
                            if (choice_bits[i] is None and lst[i][1] == target.FALSE) or (
                            choice_bits[i] is not None and not n & 1 << choice_bits[i])])
        if sublist:
            terms, nodes = zip(*sublist)
        else:
            # Empty list.
            terms, nodes = (), ()
        yield terms, nodes + sublist_no + (0,)
        n -= 1


def _builtin_all_or_none(pattern, goal, result, **kwargs):
    return _builtin_all(pattern, goal, result, allow_none=True, **kwargs)


def _builtin_all(pattern, goal, result, allow_none=False, database=None, target=None,
                      engine=None, context=None, **kwdargs):
    """
    Implementation of all/3 builtin.
   :param pattern: pattern to extract
   :type pattern: Term
   :param goal: goal to evaluate
   :type goal: Term
   :param result: list to store results
   :type result: Term
   :param database: database holding logic program
   :type database: ClauseDB
   :param target: logic formula in which to store the result
   :type target: LogicFormula
   :param engine: engine that is used for evaluation
   :type engine: ClauseDBEngine
   :param kwdargs: additional arguments from engine
   :return: list results (tuple of lists and node identifiers)
    """
    # Check the modes.
    mode = check_mode((pattern, goal, result,), ['*cv', '*cl'], database=database, **kwdargs)

    findall_head = Term(engine.get_non_cache_functor(), pattern, *goal.variables())
    findall_clause = Clause(findall_head, goal)
    findall_db = database.extend()
    findall_db += findall_clause

    class _TranslateToNone(object):

        # noinspection PyUnusedLocal
        def __getitem__(self, item):
            return None

    findall_head = substitute_simple(findall_head, _TranslateToNone())

    results = engine.call(findall_head, subcall=True, database=findall_db, target=target, **kwdargs)
    results = [(res[0], n) for res, n in results]
    output = []

    for l, n in _select_sublist(results, target):
        if not l and not allow_none:
            continue
        node = target.add_and(n)
        if node is not None:
            res = build_list(l, Term('[]'))
            if mode == 0:  # var
                args = (pattern, goal, res)
                output.append((args, node))
            else:
                try:
                    res = unify_value(res, result, {})
                    args = (pattern, goal, res)
                    output.append((args, node))
                except UnifyError:
                    pass

    return output


def _builtin_findall_base(pattern, goal, result, database=None, target=None,
                          engine=None, context=None, **kwdargs):
    """
    Implementation of findall/3 builtin.
   :param pattern: pattern to extract
   :type pattern: Term
   :param goal: goal to evaluate
   :type goal: Term
   :param result: list to store results
   :type result: Term
   :param database: database holding logic program
   :type database: ClauseDB
   :param target: logic formula in which to store the result
   :type target: LogicFormula
   :param engine: engine that is used for evaluation
   :type engine: ClauseDBEngine
   :param kwdargs: additional arguments from engine
   :return: list results (tuple of lists and node identifiers)
    """
    # Check the modes.
    mode = check_mode((pattern, goal, result,), ['*cv', '*cl'], database=database, **kwdargs)

    findall_head = Term(engine.get_non_cache_functor(), pattern, *goal.variables())
    findall_clause = Clause(findall_head, goal)
    findall_db = database.extend()
    findall_db += findall_clause

    class _TranslateToNone(object):

        # noinspection PyUnusedLocal
        def __getitem__(self, item):
            return None

    findall_head = substitute_simple(findall_head, _TranslateToNone())

    findall_target = target.__class__(keep_order=True, keep_all=True, keep_duplicates=True)
    try:
        results = engine.call(findall_head, subcall=True, database=findall_db, target=findall_target, **kwdargs)
    except RuntimeError:
        raise IndirectCallCycleError(database.lineno(kwdargs.get('call_origin', (None, None))[1]))
    new_results = []
    keep_all_restore = target._keep_all
    target._keep_all = False
    for res, n in results:
        for mx, b in findall_target.enumerate_branches(n):
            b = list(b)
            b_renamed = [findall_target.copy_node(target, c) for c in b]
            proof_node = target.add_and(b_renamed)
            # TODO order detection mechanism is too fragile?
            if b:
                new_results.append((mx, res[0], proof_node))
            else:
                new_results.append((mx, res[0], proof_node))
    target._keep_all = keep_all_restore
    new_results = [(b, c) for a, b, c in sorted(new_results, key=lambda s: s[0])]

    output = []
    for l, n in _select_sublist(new_results, target):
        node = target.add_and(n)
        if node is not None:
            res = build_list(l, Term('[]'))
            if mode == 0:  # var
                args = (pattern, goal, res)
                output.append((args, node))
            else:
                try:
                    res = unify_value(res, result, {})
                    args = (pattern, goal, res)
                    output.append((args, node))
                except UnifyError:
                    pass
    return output


# noinspection PyUnusedLocal
def _builtin_sample_all(pattern, goal, result, database=None, target=None, **kwdargs):
    # Like findall.
    pass


def _builtin_sample_uniform(key, lst, result, database=None, target=None, **kwdargs):
    """Implements the sample_uniform(+Key,+List,-Result) builtin.
    This predicate succeeds once for each element in the list as result, and with probability \
    1/(length of the list).
    The first argument is used as an identifier such that calls with the same key enforce mutual \
    exclusivity on the results, that is, the probability of

        sample_uniform(K,L,R1), sample_uniform(K,L,R2), R1 \== R2

    is 0.



    :param key:
    :param lst:
    :param result:
    :param database:
    :type database: StackBasedEngine
    :param target:
    :type target: LogicFormula
    :param kwdargs:
    :return:
    """
    mode = check_mode((key, lst, result,), ['gLv', 'gLn'], database=database, **kwdargs)
    identifier = '_uniform_%s' % key
    elements, tail = list_elements(lst)
    if len(elements) == 0:
        return []
    else:
        prob = Constant(1 / float(len(elements)))
        results = []
        if mode == 0:
            for i, elem in enumerate(elements):
                elem_identifier = (identifier, i)
                # res = unify_value(result, elem)
                results.append(((key, lst, elem),
                                target.add_atom(identifier=elem_identifier, probability=prob,
                                                group=identifier)))
        else:
            res = None
            for el in elements:
                try:
                    res = unify_value(el, result, {})
                    break
                except UnifyError:
                    pass
            if res is not None:
                results.append(((key, lst, res),
                                target.add_atom(identifier=identifier, probability=prob)))
        return results


def _builtin_findall(pattern, goal, result, **kwdargs):
    return _builtin_findall_base(pattern, goal, result, **kwdargs)


# noinspection PyPep8Naming
class problog_export(object):
    database = None

    @classmethod
    def add_function(cls, name, in_args, out_args, function):
        if problog_export.database is not None:
            problog_export.database.add_extern(name, in_args + out_args, function)

    # noinspection PyUnusedLocal
    def __init__(self, *args, **kwdargs):
        # TODO check if arguments are in order: input first, output last
        self.input_arguments = [a[1:] for a in args if a[0] == '+']
        self.output_arguments = [a[1:] for a in args if a[0] == '-']

    def _convert_input(self, a, t):
        if t == 'str':
            return str(a)
        elif t == 'int':
            return int(a)
        elif t == 'float':
            return float(a)
        elif t == 'list':
            return term2list(a)
        elif t == 'term':
            return a
        else:
            raise ValueError("Unknown type specifier '%s'!" % t)

    def _type_to_callmode(self, t):
        if t == 'str':
            return 'a'
        elif t == 'int':
            return 'i'
        elif t == 'float':
            return 'f'
        elif t == 'list':
            return 'L'
        elif t == 'term':
            return '*'
        else:
            raise ValueError("Unknown type specifier '%s'!" % t)

    def _extract_callmode(self):
        callmode_in = ''
        for t in self.input_arguments:
            callmode_in += self._type_to_callmode(t)

        # multiple call modes: index = binary encoding on whether the output is bound
        # 0 -> all unbound
        # 1 -> first output arg is bound
        # 2 -> second output arg is bound
        # 3 -> first and second are bound

        n = len(self.output_arguments)
        for i in range(0, 1 << n):
            callmode = callmode_in
            for j, t in enumerate(self.output_arguments):
                if i & (1 << (n - j - 1)):
                    callmode += self._type_to_callmode(t)
                else:
                    callmode += 'v'
            yield callmode

    def _convert_output(self, a, t):
        if t == 'str':
            return Term(a)
        elif t == 'int':
            return Constant(a)
        elif t == 'float':
            return Constant(a)
        elif t == 'list':
            return list2term(a)
        elif t == 'term':
            if not isinstance(a, Term):
                raise ValueError("Expected term output, got '%s' instead." % type(a))
            return a
        else:
            raise ValueError("Unknown type specifier '%s'!" % t)

    def _convert_inputs(self, args):
        return [self._convert_input(a, t) for a, t in zip(args, self.input_arguments)]

    def _convert_outputs(self, args):
        return [self._convert_output(a, t) for a, t in zip(args, self.output_arguments)]

    def __call__(self, function, funcname=None):
        if funcname is None:
            funcname = function.__name__

        def _wrapped_function(*args, **kwdargs):
            bound = check_mode(args, list(self._extract_callmode()), funcname, **kwdargs)
            converted_args = self._convert_inputs(args)
            result = function(*converted_args)
            if len(self.output_arguments) == 1:
                result = [result]

            try:
                transformed = []
                for i, r in enumerate(result):
                    r = self._convert_output(r, self.output_arguments[i])
                    if bound & (1 << (len(self.output_arguments) - i - 1)):
                        r = unify_value(r, args[len(self.input_arguments) + i], {})
                    transformed.append(r)
                result = args[:len(self.input_arguments)] + tuple(transformed)
                return [result]
            except UnifyError:
                return []

        problog_export.add_function(funcname, len(self.input_arguments),
                                    len(self.output_arguments), _wrapped_function)
        return function


# noinspection PyPep8Naming
class problog_export_raw(problog_export):

    # noinspection PyUnusedLocal
    def __init__(self, *args, **kwdargs):
        problog_export.__init__(self, *args, **kwdargs)
        self.input_arguments = [a[1:] for a in args]

    def _convert_input(self, a, t):
        if is_variable(a):
            return None
        else:
            return problog_export._convert_input(self, a, t)

    def _extract_callmode(self):
        callmode_in = ''

        # multiple call modes: index = binary encoding on whether the output is bound
        # 0 -> all unbound
        # 1 -> first output arg is bound
        # 2 -> second output arg is bound
        # 3 -> first and second are bound

        n = len(self.input_arguments)
        for i in range(0, 1 << n):
            callmode = callmode_in
            for j, t in enumerate(self.input_arguments):
                if i & (1 << (n - j - 1)):
                    callmode += self._type_to_callmode(t)
                else:
                    callmode += 'v'
            yield callmode

    def __call__(self, function, funcname=None):
        if funcname is None:
            funcname = function.__name__


        def _wrapped_function(*args, **kwdargs):
            bound = check_mode(args, list(self._extract_callmode()), funcname, **kwdargs)
            converted_args = self._convert_inputs(args)
            results = []
            for result in function(*converted_args):
                if len(result) == 2 and type(result[0]) == tuple:
                    # Probabilistic
                    result, p = result
                    raise Exception('We don\'t support probabilistic yet!')
                else:
                    p = None

                # result is always a list of tuples
                try:
                    transformed = []
                    for i, r in enumerate(result):
                        r = self._convert_output(r, self.input_arguments[i])
                        if bound & (1 << i):
                            r = unify_value(r, args[i], {})
                        transformed.append(r)
                    result = tuple(transformed)
                    results.append(result)
                except UnifyError:
                    pass
            return results

        problog_export.add_function(funcname, len(self.input_arguments),
                                    0, _wrapped_function)
        return function


# noinspection PyPep8Naming
class problog_export_nondet(problog_export):
    def __call__(self, function, funcname=None):
        if funcname is None:
            funcname = function.__name__

        def _wrapped_function(*args, **kwdargs):
            bound = check_mode(args, list(self._extract_callmode()), funcname, **kwdargs)
            converted_args = self._convert_inputs(args)
            results = []
            for result in function(*converted_args):
                if len(self.output_arguments) == 1:
                    result = [result]

                try:
                    transformed = []
                    for i, r in enumerate(result):
                        r = self._convert_output(r, self.output_arguments[i])
                        if bound & (1 << (len(self.output_arguments) - i - 1)):
                            r = unify_value(r, args[len(self.input_arguments) + i], {})
                        transformed.append(r)
                    result = args[:len(self.input_arguments)] + tuple(transformed)
                    results.append(result)
                except UnifyError:
                    pass
            return results

        problog_export.add_function(funcname, len(self.input_arguments),
                                    len(self.output_arguments), _wrapped_function)
        return function


def _builtin_use_module(filename, database=None, location=None, **kwdargs):
    if filename.functor == 'library' and filename.arity == 1:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'library',
                                _atom_to_filename(filename.args[0]))
        if not os.path.exists(filename + '.pl'):
            filename += '.py'
    else:
        root = database.source_root
        if filename.location:
            source_root = database.source_files[filename.location[0]]
            if source_root:
                root = os.path.dirname(source_root)

        filename = os.path.join(root, _atom_to_filename(filename))

    if filename[-3:] == '.py':
        try:
            load_external_module(database, filename)
        except IOError as err:
            raise ConsultError('Error while reading external library: %s' % str(err),
                               database.lineno(location))
        return True
    else:
        return _builtin_consult(Term(filename), database=database, location=location, **kwdargs)


def load_external_module(database, filename):
    import imp
    problog_export.database = database
    with open(filename, 'r') as extfile:
        imp.load_module('externals', extfile, filename, ('.py', 'U', 1))


def _builtin_call(term, args=(), engine=None, callback=None, transform=None, context=None, **kwdargs):
    check_mode((term,), ['c'], functor='call')
    # Find the define node for the given query term.
    term_call = term.with_args(*(term.args + args))

    try:
        if transform is None:
            from .engine_stack import Transformations
            transform = Transformations()

        def _trans(result):
            n = len(term.args)
            res1 = result[:n]
            res2 = result[n:]
            return [term.with_args(*res1)] + list(res2)
        transform.addFunction(_trans)

        actions = engine.call_intern(term_call, transform=transform, **kwdargs)
    except UnknownClauseInternal:
        raise UnknownClause(term_call.signature, kwdargs['database'].lineno(kwdargs['location']))
    return True, actions


def _builtin_subquery(term, prob, evidence=None, engine=None, database=None, **kwdargs):
    if evidence:
        check_mode((term, prob, evidence), ['cvL'], functor='subquery')
    else:
        check_mode((term, prob), ['cv'], functor='subquery')

    from .sdd_formula import SDD

    eng = engine.__class__()

    target = eng.ground(database, term, label='query')

    if evidence:
        for ev in term2list(evidence):
            target = eng.ground(database, ev, target=target, label=target.LABEL_EVIDENCE_POS)

    results = SDD.create_from(target).evaluate()
    if evidence:
        return [(t, Constant(p), evidence) for t, p in results.items()]
    else:
        return [(t, Constant(p)) for t, p in results.items()]


def _builtin_calln(term, *args, **kwdargs):
    return _builtin_call(term, args, **kwdargs)


class IndirectCallCycleError(GroundingError):
    """Cycle should not pass through indirect calls (e.g. call/1, findall/3)."""

    def __init__(self, location=None):
        GroundingError.__init__(self,
                                'Indirect cycle detected (passing through findall/3)',
                                location)
