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
import traceback

from problog.program import PrologFile
from problog.constraint import TrueConstraint
from problog.formula import LogicFormula, LogicDAG
from problog.cnf_formula import CNF
from problog.maxsat import get_solver, get_available_solvers
from problog.errors import process_error
from problog import get_evaluatable
from problog.evaluator import Semiring, OperationNotSupported, SemiringProbability
from problog.logic import Term


def main(argv):
    args = argparser().parse_args(argv)

    if args.use_semiring:
        return mpe_semiring(args)
    else:
        return mpe_maxsat(args)


def mpe_semiring(args):
    inputfile = args.inputfile

    if args.web:
        result_handler = print_result_json
    else:
        result_handler = print_result

    if args.output is not None:
        outf = open(args.output, 'w')
    else:
        outf = sys.stdout

    try:
        pl = PrologFile(inputfile)

        semiring = SemiringMPEState()
        kc_class = get_evaluatable()        # TODO should pass semiring
                                            # -> no compilation required for MPE
                                            # BUT semiring does not support negation
                                            # so formula should be NNF (only negation of facts)
        lf = LogicFormula.create_from(pl, label_all=True)

        if lf.evidence():
            qn = lf.add_and([y for x, y in lf.evidence()])
            lf.clear_evidence()

            if lf.queries():
                print ('%% WARNING: ignoring queries in file', file=sys.stderr)
            lf.clear_queries()

            query_name = Term('query')
            lf.add_query(query_name, qn)
            # print (lf)
            kc = kc_class.create_from(lf)
            results = kc.evaluate(semiring=semiring)
            prob, facts = results[query_name]
        else:
            prob, facts = 1.0, []

        result_handler((True, (prob, facts)), outf)

    except Exception as err:
        trace = traceback.format_exc()
        err.trace = trace
        result_handler((False, err), outf)


def mpe_maxsat(args):
    inputfile = args.inputfile

    if args.web:
        result_handler = print_result_json
    else:
        result_handler = print_result

    if args.output is not None:
        outf = open(args.output, 'w')
    else:
        outf = sys.stdout

    try:
        pl = PrologFile(inputfile)

        dag = LogicDAG.createFrom(pl)

        if dag.queries():
            print('%% WARNING: ignoring queries in file', file=sys.stderr)
            dag.clear_queries()

        cnf = CNF.createFrom(dag)

        for qn, qi in cnf.evidence():
            cnf.add_constraint(TrueConstraint(qi))

        # for qn, qi in cnf.queries():
        #     cnf.add_constraint(TrueConstraint(qi))

        solver = get_solver(args.solver)

        result = frozenset(solver.evaluate(cnf))
        weights = cnf.extract_weights(SemiringProbability())
        output_facts = None
        prob = 1.0
        if result is not None:
            output_facts = []
            for i, n, t in dag:
                if t == 'atom':
                    if i in result:
                        output_facts.append(n.name)
                        prob *= weights[i][0]
                    elif -i in result:
                        output_facts.append(-n.name)
                        prob *= weights[i][1]

        result_handler((True, (prob, output_facts)), outf)
    except Exception as err:
        trace = traceback.format_exc()
        err.trace = trace
        result_handler((False, err), outf)

    if args.output is not None:
        outf.close()


def print_result(result, output=sys.stdout):
    success, result = result
    prob, facts = result
    if success:
        if facts is None:
            print ('%% The model is not satisfiable.', file=output)
        else:
            for atom in facts:
                print(atom, file=output)
            if prob is not None:
                print ('%% Probability: %.10g' % prob)
        return 0
    else:
        print(process_error(result), file=output)
        return 1


def print_result_json(d, output):
    """Pretty print result.

    :param d: result from run_problog
    :param output: output file
    :param precision:
    :return:
    """
    import json
    result = {}
    success, d = d
    if success:
        prob, facts = d
        result['SUCCESS'] = True
        if facts is not None:
            result['atoms'] = list(map(lambda n: (str(-n), False) if n.is_negated() else (str(n), True), facts))
        if prob is not None:
            result['prob'] = round(prob, 10)
    else:
        result['SUCCESS'] = False
        result['err'] = process_error(d)
        result['original'] = str(d)
    print (json.dumps(result), file=output)
    return 0


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


class SemiringMPEState(Semiring):

    def __init__(self):
        Semiring.__init__(self)

    def zero(self):
        return 0.0, set()

    def one(self):
        return 1.0, set()

    def plus(self, a, b):
        if a[0] > b[0]:
            return a
        elif a[0] < b[0]:
            return b
        else:
            return a[0], a[1] | b[1]

    def times(self, a, b):
        return a[0] * b[0], a[1] | b[1]

    def pos_value(self, a, key=None):
        return float(a), {key}

    def neg_value(self, a, key=None):
        return 1.0 - float(a), {-key}

    def is_nsp(self):
        return True

    def result(self, a, formula=None):
        return a

    def ad_complement(self, ws, key=None):
        s = sum([x[0] for x in ws])
        return 1.0 - s, {key}


def argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('--solver', choices=get_available_solvers(),
                        default=None, help="MaxSAT solver to use")
    parser.add_argument('--full', dest='output_all', action='store_true',
                        help='Also show false atoms.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Write output to given file (default: write to stdout)')
    parser.add_argument('--web', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--use-maxsat', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--use-semiring', action='store_true', help=argparse.SUPPRESS)
    return parser


if __name__ == '__main__':
    main(sys.argv[1:])
