#! /usr/bin/env python

"""
ProbLog command-line interface.

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

import sys

from problog.program import PrologFile
from problog.constraint import TrueConstraint
from problog.formula import LogicFormula, LogicDAG
from problog.cnf_formula import CNF
from problog.logic import Term
from problog.maxsat import get_solver, get_available_solvers


def main(argv):
    args = argparser().parse_args(argv)
    inputfile = args.inputfile

    pl = PrologFile(inputfile)

    gp = LogicFormula.createFrom(pl, label_all=True, avoid_name_clash=True)

    dag = LogicDAG.createFrom(pl, label_all=True, avoid_name_clash=True)

    cnf = CNF.createFrom(dag)

    for qn, qi in cnf.evidence():
        cnf.add_constraint(TrueConstraint(qi))

    for qn, qi in cnf.queries():
        cnf.add_constraint(TrueConstraint(qi))

    solver = get_solver(args.solver)

    result = solver.evaluate(cnf)

    if result is None:
        print ('%% The model is not satisfiable.')
    elif not args.output_all:
        # Old output
        facts = set(result)
        pfacts = cnf.get_weights()
        true_facts = set()
        for name, node in cnf.get_names():
            if node in pfacts:
                if node in facts:
                    facts.remove(node)
                    # print (name)
                    if name.functor.startswith('problog_'):
                        functor = name.functor.split('_', 1)[1].rsplit('_', 2)[0]
                        name = Term(functor, *name.args)
                    true_facts.add(name)
                elif -node in facts:
                    facts.remove(-node)
                    # print (-name)
                else:
                    pass
                    # print (-name)

        for n in true_facts:
            print (n)
    else:
        # TODO do this on original ground program before cycle-breaking
        truthvalues = reduce_formula(dag, result)

        true_facts = set()
        for i, n, t in dag:
            if n.name is not None and truthvalues[i - 1]:
                if not n.name.functor.startswith('problog_'):
                    true_facts.add(n.name)

        for n in true_facts:
            print (n)



def reduce_formula(formula, facts):
    # Assume formula is cycle free.

    values = [None] * len(formula)
    for f in facts:
        if f > 0:
            values[f - 1] = True
        else:
            values[-f - 1] = False

    from collections import deque
    nodes = deque(range(1, len(formula) + 1))
    while nodes:
        index = nodes.popleft()
        value = values[index - 1]
        if value is None:
            node = formula.get_node(index)
            nodetype = type(node).__name__
            if nodetype == 'atom':
                pass  # Really shouldn't happen
            else:
                children = [values[c] if c > 0 else not values[-c] for c in node.children]
                if nodetype == 'disj':
                    if True in children:
                        values[index - 1] = True
                    elif None in children:
                        # not ready yet, push it back on the queue
                        nodes.append(index)
                    else:
                        values[index - 1] = False
                else:
                    if False in children:
                        values[index - 1] = False
                    elif None in children:
                        nodes.append(index)
                    else:
                        values[index - 1] = True
    return values


def argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('--solver', choices=get_available_solvers(),
                        default=None, help="MaxSAT solver to use")
    parser.add_argument('--full', dest='output_all', action='store_true',
                        help='Show all true atoms in ground program '
                             '(instead of probabilistic facts only.')
    return parser


if __name__ == '__main__':
    main(sys.argv[1:])
