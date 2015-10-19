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
from problog.maxsat import get_solver, get_available_solvers
from problog.errors import process_error


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

    try:
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

        output_facts = None
        if result is not None:
            # TODO do this on original ground program before cycle-breaking
            truthvalues = reduce_formula(dag, result)

            output_facts = set()
            true_facts = set()
            false_facts = set()
            for i, n, t in dag:
                if n.name is not None and truthvalues[i - 1]:
                    if not n.name.functor.startswith('_problog_'):
                        true_facts.add(n.name)
                if n.name is not None and not truthvalues[i - 1]:
                    if not n.name.functor.startswith('_problog_'):
                        false_facts.add(n.name)

            for n in true_facts:
                output_facts.add(n)

            if args.output_all:
                for n in false_facts:
                    output_facts.add(-n)

        result_handler((True, output_facts), outf)
    except Exception as err:
        result_handler((False, err), outf)

    if args.output is not None:
        outf.close()


def print_result(result, output=sys.stdout):
    success, result = result
    if success:
        if result is None:
            print ('%% The model is not satisfiable.', file=output)
        else:
            for atom in result:
                print(atom, file=output)
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
                        help='Also show false atoms.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Write output to given file (default: write to stdout)')
    parser.add_argument('--web', action='store_true', help=argparse.SUPPRESS)
    return parser


if __name__ == '__main__':
    main(sys.argv[1:])
