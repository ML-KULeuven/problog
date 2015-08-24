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

import stat
import sys
import os

from ..program import PrologFile, ExtendedPrologFactory
from ..evaluator import SemiringLogProbability, SemiringProbability
from ..nnf_formula import NNF
from ..sdd_formula import SDD
from ..bdd_formula import BDD
from ..forward import ForwardBDD, ForwardSDD
from ..kbest import KBestFormula
from ..util import Timer, start_timer, stop_timer, init_logger, format_dictionary
from ..core import process_error
from ..parser import DefaultPrologParser
from ..debug import EngineTracer


def main(argv):
    parser = argparser()
    args = parser.parse_args(argv)

    logger = init_logger(args.verbose)

    if args.recursion_limit:
        sys.setrecursionlimit(args.recursion_limit)

    if args.output is None:
        output = sys.stdout
    else:
        output = open(args.output, 'w')

    parse_class = DefaultPrologParser

    if args.timeout:
        start_timer(args.timeout)

    if len(args.filenames) == 0:
        mode = os.fstat(0).st_mode
        if stat.S_ISFIFO(mode) or stat.S_ISREG(mode):
            # stdin is piped or redirected
            args.filenames = ['-']
        else:
            # stdin is terminal
            # No interactive input, exit
            print('ERROR: Expected a file or stream as input.\n', file=sys.stderr)
            parser.print_help()
            sys.exit(1)

    if args.koption in ('nnf', 'ddnnf'):
        knowledge = NNF
    elif args.koption == 'sdd':
        knowledge = SDD
    elif args.koption == 'fsdd':
        knowledge = ForwardSDD
    elif args.koption == 'bdd':
        knowledge = BDD
    elif args.koption == 'fbdd':
        knowledge = ForwardBDD
    elif args.koption == 'kbest':
        knowledge = KBestFormula
    elif args.koption is None:
        if SDD.is_available():
            logger.info('Using SDD path')
            knowledge = SDD
        else:
            logger.info('Using d-DNNF path')
            knowledge = NNF
    else:
        raise ValueError("Unknown option for --knowledge: '%s'" % args.knowledge)

    if args.logspace:
        semiring = SemiringLogProbability()
    else:
        semiring = SemiringProbability()

    if args.propagate_weights:
        args.propagate_weights = semiring

    for filename in args.filenames:
        if len(args.filenames) > 1:
            print ('Results for %s:' % filename)
        result = run_problog(filename, knowledge, semiring, parse_class, **vars(args))
        retcode = print_result(result, output)
        if len(args.filenames) == 1:
            sys.exit(retcode)

    if args.output is not None:
        output.close()

    if args.timeout:
        stop_timer()


def print_result(d, output, precision=8):
    """
    Pretty print result.
    :param d: result from run_problog
    :param output: output file
    :param precision:
    :return:
    """
    success, d = d
    if success:
        print(format_dictionary(d, precision), file=output)
        return 0
    else:
        print (d, file=output)
        return 1


def run_problog(filename, knowledge=NNF, semiring=None, parse_class=DefaultPrologParser,
                debug=False, engine_debug=False, **kwdargs):
    """Run ProbLog.
    :param filename: input file
    :param knowledge: knowledge compilation class
    :param semiring: semiring to use
    :param parse_class: prolog parser to use
    :param debug: enable advanced error output
    :param engine_debug: enable engine debugging output
    :param kwdargs: additional arguments
    :return: tuple where first value indicates success, and second value contains result details
    """
    if engine_debug:
        debugger = EngineTracer()
    else:
        debugger = None

    try:
        with Timer('Total time to process model'):
            parser = parse_class(ExtendedPrologFactory())
            formula = knowledge.createFrom(PrologFile(filename, parser=parser), debugger=debugger,
                                           **kwdargs)
        with Timer('Evaluation'):
            result = formula.evaluate(semiring=semiring, **kwdargs)
        return True, result
    except Exception as err:
        return False, process_error(err, debug=debug)


def argparser():
    """Create the default argument parser for ProbLog.
    :return: argument parser
    :rtype: argparse.ArgumentParser
    """
    import argparse

    class InputFile(str):
        """Stub class for file input arguments."""
        pass

    class OutputFile(str):
        """Stub class for file output arguments."""
        pass

    description = """ProbLog 2.1 command line interface

    The arguments listed below are for the default mode.
    ProbLog also supports the following alternative modes:

      - (default): inference
      - install: run the installer
      - ground: generate ground program (see ground --help)
      - sample: generate samples from the model (see sample --help)
      - unittest: run the testsuite

    Select a mode by adding one of these keywords as first argument (e.g. problog-cli.py install).
    """

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filenames', metavar='MODEL', nargs='*', type=InputFile)
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    parser.add_argument('--knowledge', '-k', dest='koption',
                        choices=('sdd', 'nnf', 'ddnnf', 'bdd', 'fsdd', 'fbdd', 'kbest'),
                        default=None, help="Knowledge compilation tool.")

    # Evaluation semiring
    ls_group = parser.add_mutually_exclusive_group()
    ls_group.add_argument('--logspace', action='store_true',
                          help="Use log space evaluation (default).", default=True)
    ls_group.add_argument('--nologspace', dest='logspace', action='store_false',
                          help="Use normal space evaluation.")

    parser.add_argument('--output', '-o', help="Output file (default stdout)", type=OutputFile)
    parser.add_argument('--recursion-limit',
                        help="Set Python recursion limit. (default: %d)" % sys.getrecursionlimit(),
                        default=sys.getrecursionlimit(), type=int)
    parser.add_argument('--timeout', '-t', type=int, default=0,
                        help="Set timeout (in seconds, default=off).")
    parser.add_argument('--compile-timeout', type=int, default=0,
                        help="Set timeout for compilation (in seconds, default=off).")
    parser.add_argument('--debug', '-d', action='store_true',
                        help="Enable debug mode (print full errors).")

    # Additional arguments (passed through)
    parser.add_argument('--engine-debug', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--propagate-evidence', action='store_true',
                        dest='propagate_evidence',
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--propagate-weights', action='store_true', default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--convergence', '-c', type=float, default=argparse.SUPPRESS,
                        help='stop anytime when bounds are within this range')

    # SDD garbage collection
    sdd_auto_gc_group = parser.add_mutually_exclusive_group()
    sdd_auto_gc_group.add_argument('--sdd-auto-gc', action='store_true', dest='sdd_auto_gc',
                                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sdd_auto_gc_group.add_argument('--sdd-no-auto-gc', action='store_false', dest='sdd_auto_gc',
                                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    sdd_fixvars_group = parser.add_mutually_exclusive_group()
    sdd_fixvars_group.add_argument('--sdd-preset-variables', action='store_true',
                                   dest='sdd_preset_variables',
                                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sdd_fixvars_group.add_argument('--sdd-no-preset-variables', action='store_false',
                                   dest='sdd_preset_variables',
                                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)

    return parser

if __name__ == '__main__':
    main(sys.argv[1:])
