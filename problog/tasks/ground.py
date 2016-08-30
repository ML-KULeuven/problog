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
from problog.errors import process_error
from problog.util import subprocess_check_output, mktempfile


def main(argv, result_handler=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='MODEL', type=str, help='input ProbLog model')
    parser.add_argument('--format', choices=('dot', 'pl', 'cnf', 'svg', 'internal'), default=None,
                        help='output format')
    parser.add_argument('--break-cycles', action='store_true', help='perform cycle breaking')
    parser.add_argument('--keep-all', action='store_true', help='also output deterministic nodes')
    parser.add_argument('--keep-duplicates', action='store_true', help='don\'t eliminate duplicate literals')
    parser.add_argument('--any-order', action='store_true', help='allow reordering nodes')
    parser.add_argument('--hide-builtins', action='store_true', help='hide deterministic part based on builtins')
    parser.add_argument('--propagate-evidence', action='store_true', help='propagate evidence')
    parser.add_argument('--propagate-weights', action='store_true', help='propagate evidence')
    parser.add_argument('--compact', action='store_true',
                        help='allow compact model (may remove some predicates)')
    parser.add_argument('--noninterpretable', action='store_true')
    parser.add_argument('--web', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-o', '--output', type=str, help='output file', default=None)
    parser.add_argument('-a', '--arg', dest='args', action='append',
                        help='Pass additional arguments to the cmd_args builtin.')

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

    if args.web:
        print_result = print_result_json
    else:
        print_result = print_result_standard

    try:
        gp = target.createFrom(
            PrologFile(args.filename, parser=DefaultPrologParser(ExtendedPrologFactory())),
            label_all=not args.noninterpretable, avoid_name_clash=not args.compact,
            keep_order=not args.any_order,
            keep_all=args.keep_all, keep_duplicates=args.keep_duplicates,
            hide_builtins=args.hide_builtins,
            propagate_evidence=args.propagate_evidence, propagate_weights=semiring, args=args.args)

        if outformat == 'pl':
            rc = print_result((True, gp.to_prolog()), output=outfile)
        elif outformat == 'dot':
            rc = print_result((True, gp.to_dot()), output=outfile)
        elif outformat == 'svg':
            dot = gp.to_dot()
            tmpfile = mktempfile('.dot')
            with open(tmpfile, 'w') as f:
                print(dot, file=f)
            svg = subprocess_check_output(['dot', tmpfile, '-Tsvg'])
            rc = print_result((True, svg), output=outfile)
        elif outformat == 'cnf':
            rc = print_result((True, CNF.createFrom(gp).to_dimacs()), output=outfile)
        elif outformat == 'internal':
            rc = print_result((True, str(gp)), output=outfile)
        else:
            rc = print_result((True, gp.to_prolog()), output=outfile)
    except Exception as err:
        import traceback
        err.trace = traceback.format_exc()
        rc = print_result((False, err))

    if args.output:
        outfile.close()

    if rc:
        sys.exit(rc)


def print_result_standard(result, output=sys.stdout):
    success, result = result
    if success:
        print (result, file=output)
        return 0
    else:
        print (process_error(result), file=output)
        return 1


def print_result_json(result, output=sys.stdout):
    success, result = result
    import json
    out = {'SUCCESS': success}
    if success:
        out['result'] = str(result)
    else:
        out['err'] = vars(result)
    print (json.dumps(out), file=output)
    return 0
