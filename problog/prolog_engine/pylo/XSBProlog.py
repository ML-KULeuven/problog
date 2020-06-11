from pylo.Prolog import (
    Prolog
)
from pylo.language import Constant, Variable, Functor, Structure, List, Predicate, Literal, Negation, Clause, global_context
import sys
sys.path.append("../../build")
import os

from pylo import pyxsb
from typing import Union, Dict, Sequence
from functools import reduce


def _is_list(term: str):
    return term.startswith('[')


def _is_structure(term: str):
    first_bracket = term.find('(')

    if first_bracket == -1:
        return False
    else:
        tmp = term[:first_bracket]
        return all([x.isalnum() for x in tmp]) and tmp[0].isalpha()


def _pyxsb_string_to_const_or_var(term: str):
    if term[0].islower():
        return global_context.get_symbol(term)
    elif term.isnumeric():
        if '.' in term:
            return float(term)
        else:
            return int(term)
    else:
        return global_context.get_variable(term)


def _extract_arguments_from_compound(term: str):
    if _is_list(term):
        term = term[1:-1]  # remove '[' and ']'
    else:
        first_bracket = term.find('(')
        term = term[first_bracket+1:-1] # remove outer brackets

    args = []
    open_brackets = 0
    last_open_char = 0
    for i in range(len(term)):
        char = term[i]
        if term[i] in ['(', '[']:
            open_brackets += 1
        elif term[i] in [')', ']']:
            open_brackets -= 1
        elif term[i] == ',' and open_brackets == 0:
            args.append(term[last_open_char:i])
            last_open_char = i + 1
        elif i == len(term) - 1:
            args.append(term[last_open_char:])
        else:
            pass

    return args


def _pyxsb_string_to_structure(term: str):
    first_bracket = term.find('(')
    functor = term[:first_bracket]
    args = [_pyxsb_string_to_pylo(x) for x in _extract_arguments_from_compound(term)]
    functor = global_context.get_symbol(functor, arity=len(args))

    return Structure(functor, args)


def _pyxsb_string_to_list(term: str):
    args = [_pyxsb_string_to_pylo(x) for x in _extract_arguments_from_compound(term)]
    return List(args)


def _pyxsb_string_to_pylo(term: str):
    if _is_list(term):
        return _pyxsb_string_to_list(term)
    elif _is_structure(term):
        return _pyxsb_string_to_structure(term)
    else:
        return _pyxsb_string_to_const_or_var(term)


class XSBProlog(Prolog):

    def __init__(self, exec_path=None):
        if exec_path is None:
            exec_path = os.getenv('XSB_HOME', None)
            raise Exception(f"Cannot find XSB_HOME environment variable")
        pyxsb.pyxsb_init_string(exec_path)
        super().__init__()

    def __del__(self):
        pyxsb.pyxsb_close()

    def consult(self, filename: str):
        return pyxsb.pyxsb_command_string(f"consult('{filename}').")

    def use_module(self, module: str, **kwargs):
        assert 'predicates' in kwargs, "XSB Prolog: need to specify which predicates to import from module"
        predicates = kwargs['predicates']
        command = f"use_module({module},[{','.join([x.get_name() + '/' + str(x.get_arity()) for x in predicates])}])."
        return pyxsb.pyxsb_command_string(command)

    def asserta(self, clause: Union[Clause, Literal]):
        if isinstance(clause, Literal):
            return pyxsb.pyxsb_command_string(f"asserta({clause}).")
        else:
            return pyxsb.pyxsb_command_string(f"asserta(({clause})).")

    def assertz(self, clause: Union[Literal, Clause]):
        if isinstance(clause, Literal):
            return pyxsb.pyxsb_command_string(f"assertz({clause}).")
        else:
            return pyxsb.pyxsb_command_string(f"assertz(({clause})).")

    def retract(self, clause: Union[Literal]):
        return pyxsb.pyxsb_command_string(f"retract({clause}).")

    def has_solution(self, *query):
        string_repr = ','.join([str(x) for x in query])
        res = pyxsb.pyxsb_query_string(f"{string_repr}.")

        if res:
            pyxsb.pyxsb_close_query()

        return True if res else False

    def query(self, *query, **kwargs):
        if 'max_solutions' in kwargs:
            max_solutions = kwargs['max_solutions']
        else:
            max_solutions = -1

        vars_of_interest = [[y for y in x.get_arguments() if isinstance(y, Variable)] for x in query]
        vars_of_interest = reduce(lambda x, y: x + y, vars_of_interest, [])
        vars_of_interest = reduce(lambda x, y: x + [y] if y not in x else x, vars_of_interest, [])

        string_repr = ','.join([str(x) for x in query])
        res = pyxsb.pyxsb_query_string(f"{string_repr}.")

        all_solutions = []
        while res and max_solutions != 0:
            vals = [x for x in res.strip().split(";")]
            var_assignments = [_pyxsb_string_to_pylo(x) for x in vals]
            all_solutions.append(dict([(v, s) for v, s in zip(vars_of_interest, var_assignments)]))

            res = pyxsb.pyxsb_next_string()
            max_solutions -= 1

        return all_solutions


if __name__ == '__main__':
    pl = XSBProlog("/home/yann/xsb-git/XSB")

    p = global_context.get_predicate("p", 2)
    f = global_context.get_functor("t", 3)
    f1 = p("a", "b")

    pl.assertz(f1)

    X = global_context.get_variable("X")
    Y = global_context.get_variable("Y")

    query = p(X, Y)

    r = pl.has_solution(query)
    print("has solution", r)

    rv = pl.query(query)
    print("all solutions", rv)

    f2 = p("a", "c")
    pl.assertz(f2)

    rv = pl.query(query)
    print("all solutions after adding f2", rv)

    func1 = f(1, 2, 3)
    f3 = p(func1, "b")
    pl.assertz(f3)

    rv = pl.query(query)
    print("all solutions after adding structure", rv)

    l = List([1, 2, 3, 4, 5])

    member = global_context.get_predicate("member", 2)
    pl.use_module("lists", predicates=[member])

    query2 = member(X, l)

    rv = pl.query(query2)
    print("all solutions to list membership ", rv)

    r = global_context.get_predicate("r", 2)
    f4 = r("a", l)
    f5 = r("a", "b")

    pl.asserta(f4)
    pl.asserta(f5)

    query3 = r(X, Y)

    rv = pl.query(query3)
    print("all solutions after adding list ", rv)

