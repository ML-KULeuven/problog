from __future__ import print_function

from problog.prolog_engine.extern import swi_problog_export, swi_problog_export_nondet
from problog.logic import Constant, list2term, Term
from problog.prolog_engine.prolog_engine import PrologEngine
from problog.formula import LogicFormula
from problog.sdd_formula import SDD
from copy import deepcopy


@swi_problog_export("+term", "+term", "-list")
def export_findall(template, goal, **kwargs):
    result = run_subquery(template, goal, **kwargs)
    return list(result)


@swi_problog_export_nondet("?term", "-float")
def subquery(template, **kwargs):
    result = run_subquery(template, **kwargs)
    return [[term, get_probability(term)] for term in result]


def run_subquery(template, goal=None, **kwargs):
    database = kwargs["database"]
    engine = PrologEngine()
    sp = engine.prepare(database)
    str_template = str(template)
    if goal is not None:
        str_template = "subquery_template__(" + str(template) + ")"
        sp.add_clause_string(str_template, str(goal))
    formula = engine.ground(sp, str_template,
                            target=LogicFormula(keep_all=False), label=LogicFormula.LABEL_QUERY)
    sdd = SDD.create_from(formula)
    result = sdd.evaluate()
    if goal is not None:
        temp = {}
        for k in result:
            for arg in k.args:
                if arg not in temp:
                    temp[arg] = result[k]
                else:
                    temp[arg] = max(temp[arg], result[k])
        result = temp
    return result


def get_probability(term):
    probability = term.probability
    if type(probability) is float or type(probability) is int:
        return probability
    if probability is not None:
        return probability.value
    return 1.0
