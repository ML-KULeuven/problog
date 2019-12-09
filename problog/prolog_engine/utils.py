from __future__ import print_function

from extern import swi_problog_export, swi_problog_export_nondet
from problog.logic import Constant, list2term, Term
from prolog_engine import PrologEngine
from problog.formula import LogicFormula
from problog.sdd_formula import SDD


@swi_problog_export("+term", "-term", "+term", "-term")
def export_test(*term, **kwargs):
    return Constant(term[0].functor + term[1].functor), Constant(0.8)


@swi_problog_export("+term", "-term")
def export_unify(term, **kwargs):
    return term,


@swi_problog_export("+term")
def export_print(term, **kwargs):
    print("PRINT ", term)
    return ()


@swi_problog_export_nondet("+int", "+int", "-int")
def export_between(lower_bound, upper_bound, **kwargs):
    return [(i, ) for i in range(lower_bound, upper_bound)]


@swi_problog_export("+term", "-list")
def findall(template, **kwargs):
    result = run_subquery(template, **kwargs)
    return list2term(list(result)),


@swi_problog_export_nondet("?term", "-float")
def subquery(template, **kwargs):
    result = run_subquery(template, **kwargs)
    return [[term, get_probability(term)] for term in result]


def run_subquery(template, **kwargs):
    database = kwargs["database"]
    engine = PrologEngine()
    sp = engine.prepare(database)
    formula = engine.ground(sp, str(template), target=LogicFormula(keep_all=False), label=LogicFormula.LABEL_QUERY)
    sdd = SDD.create_from(formula)
    result = sdd.evaluate()
    return result


def get_probability(term):
    probability = term.probability
    if type(probability) is float or type(probability) is int:
        return probability
    if probability is not None:
        return probability.value
    return 1.0

