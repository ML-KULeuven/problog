"""
Module name
"""

from __future__ import print_function

testprogram = """
    a(1).
    a(2).
    b(1,2).
    b(2,3).
    c(2,6).
    c(3,4).
    c(3,5).
    d(5).

    q(X,Z) :- @f(a(X),b(X,Y)), c(Y,Z), @g(d(Z)).

    query(q(_,_)).
"""

from problog.engine import DefaultEngine
from problog.program import PrologString
from problog.logic import *
from problog.formula import LogicFormula
from problog.engine_stack import SimpleProbabilisticBuiltIn

import re
import sys
from collections import defaultdict

def builtin_annotate(annotation, *terms, **kwdargs):
    return builtin_annotate_help(annotation, terms, **kwdargs)

def builtin_annotate_help(annotation, terms, target=None, database=None, engine=None, **kwdargs):
    body = And.fromList(terms)
    body_vars = body.variables()

    clause_head = Term(engine.get_non_cache_functor(), *body_vars)
    clause = Clause(clause_head, body)
    subdb = database.extend()
    subdb += clause

    results = engine.call(clause_head, subcall=True, database=subdb, target=target, **kwdargs)
    results = [(res, n) for res, n in results]

    output = []
    for res, node in results:
        varvalues = {var: val for var, val in zip(body_vars, res)}
        output.append(([annotation] + [term.apply(varvalues) for term in terms], node))
        target.annotations[node].append(annotation)
    return output

def main(filename=None):

    # Step 1: remove syntactic sugar
    if filename is None:
        data = testprogram
    else:
        with open(filename) as f:
            data = f.read()

    data, count = re.subn('@([^(]+)[(]', r'annotate(\1, ', data)
    model = PrologString(data)


    engine = DefaultEngine(label_all=True)
    for i in range(2, 10):
        engine.add_builtin('annotate', i, SimpleProbabilisticBuiltIn(builtin_annotate))
    db = engine.prepare(model)

    gp = LogicFormula(keep_all=True)
    gp.annotations = defaultdict(list)

    gp = engine.ground_all(db, target=gp)
    print (gp)
    print (gp.annotations)
    pass


if __name__ == '__main__':
    main(*sys.argv[1:])
