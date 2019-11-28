from problog.logic import Term, Constant, list2term, term2list, Var, is_list
from problog.parser import PrologParser
from problog.program import ExtendedPrologFactory
from pyswip import Prolog, Functor, Atom, registerForeign, PL_FA_NONDETERMINISTIC, Variable
from time import time


def pyswip_to_term(pyswip_obj):
    if type(pyswip_obj) is Functor:
        args = [pyswip_to_term(a) for a in pyswip_obj.args]
        return Term(pyswip_obj.name.get_value(), *args)
    elif type(pyswip_obj) is Atom:
        return Term(pyswip_obj.get_value())
    elif type(pyswip_obj) is int or type(pyswip_obj) is float:
        return Constant(pyswip_obj)
    elif type(pyswip_obj) is list:
        return list2term([pyswip_to_term(o) for o in pyswip_obj])
    elif type(pyswip_obj) is Variable:
        return Var(pyswip_obj.chars)
    else:
        raise Exception('Unhandled type {} from object {}'.format(type(pyswip_obj), pyswip_obj))


def term_to_pyswip(term):
    if type(term) is Term:
        args = [term_to_pyswip(arg) for arg in term.args]
        if not args:
            return Atom(term.functor)
        elif is_list(term):
            return args
        return Functor(term.functor, arity=term.arity, args=args)
    elif type(term) is Constant:
        return term.functor
    elif type(term) is Var:
        return Variable(name=term.name)
    else:
        raise Exception('Unhandled type {} from object {} -> Robin has to fix it'.format(type(term), term))


parser = PrologParser(ExtendedPrologFactory())


def parse(to_parse):
    if type(to_parse) is str:
        return parser.parseString(str(to_parse) + '.')[0]
    return pyswip_to_term(to_parse)
