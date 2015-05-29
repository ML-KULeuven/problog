#! /usr/bin/env python
"""
ProbLog command-line interface.
"""

from __future__ import print_function

import os
import sys
import logging
import stat

from problog.program import PrologFile, ExtendedPrologFactory
from problog.evaluator import SemiringSymbolic, SemiringLogProbability
from problog.nnf_formula import NNF
from problog.sdd_formula import SDD
from problog.formula import LogicFormula, LogicDAG
from problog.cnf_formula import CNF
from problog.util import Timer, start_timer, stop_timer
from problog.core import process_error, process_result
from problog.parser import DefaultPrologParser
from problog.debug import EngineTracer


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
        print(process_result(d, precision), file=output)
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
            formula = knowledge.createFrom(PrologFile(filename, parser=parser), debugger=debugger, **kwdargs)
        with Timer('Evaluation'):
            result = formula.evaluate(semiring=semiring)
        return True, result
    except Exception as err:
        return False, process_error(err, debug=debug)


def ground(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='MODEL', type=str, help='input ProbLog model')
    parser.add_argument('--format', choices=('dot', 'pl', 'cnf'), default=None, help='output format')
    parser.add_argument('--break-cycles', action='store_true', help='perform cycle breaking')
    parser.add_argument('-o', '--output', type=str, help='output file', default=None)
    args = parser.parse_args(argv)

    outformat = args.format
    outfile = sys.stdout
    if args.output:
        outfile = open(args.output, 'w')
        if outformat is None:
            outformat = os.path.splitext(args.output)[1][1:]

    if outformat == 'cnf' and not args.break_cycles:
        print ('Warning: CNF output requires cycle-breaking; cycle breaking enabled.', file=sys.stderr)

    if args.break_cycles or outformat == 'cnf':
        target = LogicDAG
    else:
        target = LogicFormula

    gp = target.createFrom(
        PrologFile(args.filename, parser=DefaultPrologParser(ExtendedPrologFactory())),
        label_all=True, avoid_name_clash=True, keep_order=True)

    if outformat == 'pl':
        print (gp.to_prolog(), file=outfile)
    elif outformat == 'dot':
        print (gp.to_dot(), file=outfile)
    elif outformat == 'cnf':
        print (CNF.createFrom(gp).to_dimacs(), file=outfile)
    else:
        print (gp.to_prolog(), file=outfile)

    if args.output:
        outfile.close()


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
    parser.add_argument('--knowledge', '-k', dest='koption', choices=('sdd', 'nnf', 'ddnnf'),
                        default=None, help="Knowledge compilation tool.")

    # Evaluation semiring
    ls_group = parser.add_mutually_exclusive_group()
    ls_group.add_argument('--symbolic', action='store_true',
                          help="Use symbolic evaluation.")
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
    parser.add_argument('--debug', '-d', action='store_true',
                        help="Enable debug mode (print full errors).")

    # Additional arguments (passed through)
    parser.add_argument('--engine-debug', action='store_true', help=argparse.SUPPRESS)

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


def main(argv):
    """Main function.
    :param argv: command line arguments
    """

    if len(argv) > 0:
        if argv[0] == 'install':
            from problog import setup
            setup.install()
            return
        elif argv[0] == 'info':
            from problog.core import list_transformations
            list_transformations()
            return
        elif argv[0] == 'unittest':
            import unittest
            test_results = unittest.TextTestResult(sys.stderr, False, 1)
            unittest.TestLoader().discover(os.path.dirname(__file__)).run(test_results)
            return
        elif argv[0] == 'sample':
            from problog.sample import main
            return main(argv[1:])
        elif argv[0] == 'ground':
            return ground(argv[1:])

    parser = argparser()
    args = parser.parse_args(argv)

    logger = logging.getLogger('problog')
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Output level: DEBUG')
    elif args.verbose is None:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
        logger.info('Output level: INFO')
    else:
        logger.setLevel(logging.DEBUG)
        logger.debug('Output level: DEBUG')

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
    elif args.koption is None:
        if SDD.is_available() and not args.symbolic:
            logger.info('Using SDD path')
            knowledge = SDD
        else:
            logger.info('Using d-DNNF path')
            knowledge = NNF
    else:
        raise ValueError("Unknown option for --knowledge: '%s'" % args.knowledge)

    if args.symbolic:
        semiring = SemiringSymbolic()
    elif args.logspace:
        semiring = SemiringLogProbability()
    else:
        semiring = None

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


if __name__ == '__main__':
    main(sys.argv[1:])