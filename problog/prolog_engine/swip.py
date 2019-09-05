from pyswip import Prolog, Functor, Atom, registerForeign, PL_FA_NONDETERMINISTIC

from problog.logic import Term, Constant, list2term
from problog.parser import PrologParser
from problog.program import ExtendedPrologFactory


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
    else:
        raise Exception('Unhandled type {}'.format(type(pyswip_obj)))


def query(consult_file, query, asserts=None):
    prolog = Prolog()
    prolog.consult(str(consult_file))
    prolog.retractall('rule(X,Y)')
    prolog.retractall('fact(X,Y,Z)')
    if asserts is not None:
        for a in asserts:
            prolog.assertz(str(a))

    result = list(prolog.query(str(query)))
    proofs = []
    for r in result:
        proof = r['Proofs']
        proofs += [pyswip_to_term(p) for p in proof]
    return proofs

def register_foreign(f, **kwargs):
    registerForeign(f, flags=PL_FA_NONDETERMINISTIC, **kwargs)
parser = PrologParser(ExtendedPrologFactory())


def parse(to_parse):
    return parser.parseString(str(to_parse) + '.')[0]
