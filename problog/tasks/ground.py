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

import os
import sys

from problog.formula import LogicDAG, LogicFormula
from problog.evaluator import SemiringLogProbability
from problog.parser import DefaultPrologParser
from problog.program import ExtendedPrologFactory, PrologFile
from problog.cnf_formula import CNF


def main(argv, result_handler=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='MODEL', type=str, help='input ProbLog model')
    parser.add_argument('--format', choices=('dot', 'pl', 'cnf', 'internal'), default=None,
                        help='output format')
    parser.add_argument('--break-cycles', action='store_true', help='perform cycle breaking')
    parser.add_argument('--keep-all', action='store_true', help='also output deterministic nodes')
    parser.add_argument('--propagate-evidence', action='store_true', help='propagate evidence')
    parser.add_argument('--propagate-weights', action='store_true', help='propagate evidence')
    parser.add_argument('--compact', action='store_true',
                        help='allow compact model (may remove some predicates)')
    parser.add_argument('-o', '--output', type=str, help='output file', default=None)
    args = parser.parse_args(argv)

    outformat = args.format
    outfile = sys.stdout
    if args.output:
        outfile = open(args.output, 'w')
        if outformat is None:
            outformat = os.path.splitext(args.output)[1][1:]

    if outformat == 'cnf' and not args.break_cycles:
        print('Warning: CNF output requires cycle-breaking; cycle breaking enabled.',
              file=sys.stderr)

    if args.break_cycles or outformat == 'cnf':
        target = LogicDAG
    else:
        target = LogicFormula

    if args.propagate_weights:
        semiring = SemiringLogProbability()
    else:
        semiring = None

    gp = target.createFrom(
        PrologFile(args.filename, parser=DefaultPrologParser(ExtendedPrologFactory())),
        label_all=True, avoid_name_clash=not args.compact, keep_order=True, keep_all=args.keep_all,
        propagate_evidence=args.propagate_evidence, propagate_weights=semiring)

    if outformat == 'pl':
        print(gp.to_prolog(), file=outfile)
    elif outformat == 'dot':
        print(gp.to_dot(), file=outfile)
    elif outformat == 'cnf':
        print(CNF.createFrom(gp).to_dimacs(), file=outfile)
    elif outformat == 'internal':
        print(str(gp), file=outfile)
    else:
        print(gp.to_prolog(), file=outfile)

    if args.output:
        outfile.close()
