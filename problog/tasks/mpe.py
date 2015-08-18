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
from problog.formula import TrueConstraint
from problog.cnf_formula import CNF
from problog.maxsat import get_solver, get_available_solvers


def main(argv):
    args = argparser().parse_args(argv)
    inputfile = args.inputfile

    cnf = CNF.createFrom(PrologFile(inputfile), label_all=True)

    for qn, qi in cnf.evidence():
        cnf.add_constraint(TrueConstraint(qi))

    for qn, qi in cnf.queries():
        cnf.add_constraint(TrueConstraint(qi))

    solver = get_solver(args.solver)

    result = solver.evaluate(cnf)
    if result is None:
        print ('%% The model is not satisfiable.')
    else:
        facts = set(result)
        for name, node in cnf.get_names():
            if node in facts:
                facts.remove(node)
                print (name)
            elif -node in facts:
                facts.remove(-node)
                print (-name)
        for f in facts:
            if f < 0:
                print ('\+node_%s' % -f)
            else:
                print ('node_%s' % f)


def argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('--solver', choices=get_available_solvers(),
                        default=None, help="MaxSAT solver to use")
    return parser


if __name__ == '__main__':
    main(sys.argv[1:])
