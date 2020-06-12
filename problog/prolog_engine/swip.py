from problog.logic import Term, Constant, list2term, term2list, Var, is_list, make_safe, unquote
from problog.parser import PrologParser
from problog.program import ExtendedPrologFactory
from problog.prolog_engine.my_pyswip import Prolog, Functor, Atom, registerForeign, PL_FA_NONDETERMINISTIC, Variable, Term as SWITerm
from time import time


def pyswip_to_term(pyswip_obj, rtype=None):
    if type(pyswip_obj) is Functor:
        args = [pyswip_to_term(a) for a in pyswip_obj.args]
        return Term(pyswip_obj.name.get_value(), *args)
    elif type(pyswip_obj) is Atom:
        return Term(pyswip_obj.get_value())
    elif type(pyswip_obj) in (int, float):
        return Constant(pyswip_obj)
    elif type(pyswip_obj) is str:
        return Constant(make_safe(pyswip_obj))
    elif type(pyswip_obj) is list:
        if rtype is list:
            return [pyswip_to_term(o, rtype=rtype) for o in pyswip_obj]
        return list2term([pyswip_to_term(o) for o in pyswip_obj])
    elif type(pyswip_obj) is Variable:
        return Var("A" + pyswip_obj.chars[1:])
    else:
        raise Exception('Unhandled type {} from object {}'.format(type(pyswip_obj), pyswip_obj))


def pyswip_to_number(pyswip_obj, number_type):
    if type(pyswip_obj) is number_type:
        return pyswip_obj
    raise Exception('Unable to cast type {} of object {} into {}'.format(type(pyswip_obj), pyswip_obj, number_type))


def bind_functor(functor):
    a = functor.swipl.PL_new_term_refs(len(functor.args))
    for i, arg in enumerate(functor.args):
        functor.swipl.PL_put_term(a + i, arg.handle)

    t = functor.swipl.PL_new_term_ref()
    functor.swipl.PL_cons_functor_v(t, functor.handle, a)

    return SWITerm(functor.swipl, t)


def term_to_pyswip(term, swipl, rtype=None):
    if type(term) is list:
        return [term_to_pyswip(arg, swipl) for arg in term]
    if type(term) is Term:
        if is_list(term):
            arglist = [term_to_pyswip(term.args[0], swipl)]
            temp = term.args[1]
            while len(temp.args) == 2:
                arglist.append(term_to_pyswip(temp.args[0], swipl))
                temp = temp.args[1]
            return arglist
        elif not term.args:
            return Atom(term.functor, swipl)
        else:
            args = [term_to_pyswip(arg, swipl) for arg in term.args]
        return Functor(term.functor, swipl, arity=term.arity, args=args)
    elif type(term) is Constant:
        f = term.functor
        if type(f) is str:
            f = unquote(f)
            if rtype is Term:
                f = Atom(f, swipl)
        return f
    elif type(term) is Var:
        return Variable(name=term.name)
    else:
        raise Exception('Unhandled type {} from object {} -> Robin has to fix it'.format(type(term), term))


parser = PrologParser(ExtendedPrologFactory())


def parse(to_parse):
    if type(to_parse) is str:
        return parser.parseString(str(to_parse) + '.')[0]
    return pyswip_to_term(to_parse)


def parse_result(result):
    out = {}
    for k in result[0]:
        v = result[0][k]
        if type(v) is list:
            out[k] = [p for p in term2list(parse(result[0][k]))]
        else:
            out[k] = parse(v)
    return out
