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
from problog.engine import DefaultEngine
from problog.logic import Term, Constant
from problog.nnf_formula import NNF
from problog.constraint import TrueConstraint
from problog.formula import LogicFormula, LogicDAG
from problog.cnf_formula import CNF
from problog.maxsat import get_solver, get_available_solvers
from problog.errors import process_error
from problog.evaluator import SemiringProbability
from problog import get_evaluatables, get_evaluatable


class WeightSemiring(SemiringProbability):

    def value(self, a):
        return float(a)

    # def times(self, a, b):
    #     return a + b
    #
    # def plus(self, a, b):
    #     raise Exception()

    def pos_value(self, a):
        return self.value(a)

    def neg_value(self, a):
        return 0.0


def main(argv, result_handler=None):
    args = argparser().parse_args(argv)
    inputfile = args.inputfile

    if result_handler is None:
        if args.web:
            result_handler = print_result_json
        else:
            result_handler = print_result

    if args.output is not None:
        outf = open(args.output, 'w')
    else:
        outf = sys.stdout

    # try:
    pl = PrologFile(inputfile)

    eng = DefaultEngine()
    db = eng.prepare(pl)

    decisions = dict((d[0], None) for d in eng.query(db, Term('decision', None)))
    utilities = dict(eng.query(db, Term('utility', None, None)))

    for d in decisions:
        db += d.with_probability(Constant(0.5))

    gp = eng.ground_all(db, target=None, queries=utilities.keys(), evidence=decisions.items())

    knowledge = get_evaluatable(args.koption).create_from(gp)

    best_choice = None
    best_score = None

    decision_names = decisions.keys()
    for i in range(0, 1 << len(decisions)):
        choices = num2bits(i, len(decisions))

        evidence = dict(zip(decision_names, choices))
        result = knowledge.evaluate(evidence=evidence)

        score = 0.0
        for r in result:
            score += result[r] * float(utilities[r])
        # print (result, score)

        if best_score is None or score > best_score:
            best_score = score
            best_choice = dict(evidence)
    print (best_choice, best_score)

    # result_handler((True, best_choice), outf)
    # except Exception as err:
    #    result_handler((False, err), outf)

    if args.output is not None:
        outf.close()


def num2bits(n, nbits):
    bits = [False] * nbits
    for i in range(1, nbits + 1):
        bits[nbits - i] = bool(n % 2)
        n >>= 1
    return bits


def print_result(result, output=sys.stdout):
    success, result = result
    if success:
        if result is None:
            print ('%% The model is not satisfiable.', file=output)
        else:
            for atom in result.items():
                print('%s: %s' % atom, file=output)
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
        result['SUCCESS'] = True
        if d is not None:
            result['atoms'] = list(map(lambda n: (str(-n), False) if n.is_negated() else (str(n), True), d))
    else:
        result['SUCCESS'] = False
        result['err'] = process_error(d)
        result['original'] = str(d)
    print (json.dumps(result), file=output)
    return 0


def argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('--knowledge', '-k', dest='koption',
                        choices=get_evaluatables(),
                        default=None, help="Knowledge compilation tool.")

    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Write output to given file (default: write to stdout)')
    parser.add_argument('--web', action='store_true', help=argparse.SUPPRESS)
    return parser


if __name__ == '__main__':
    main(sys.argv[1:])
