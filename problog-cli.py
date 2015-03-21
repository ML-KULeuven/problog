#! /usr/bin/env python

from __future__ import print_function

import os, sys, subprocess, traceback, logging, stat

from problog.program import PrologFile, ExtendedPrologFactory
from problog.evaluator import SemiringSymbolic, SemiringLogProbability, Evaluator
from problog.nnf_formula import NNF
from problog.sdd_formula import SDD
from problog.util import Timer, start_timer, stop_timer
from problog.core import process_error
from problog.parser import DefaultPrologParser


def print_result( d, output, precision=8 ) :    
    success, d = d
    if success :
        if not d : return 0 # no queries
        l = max( len(str(k)) for k in d )
        f_flt = '\t%' + str(l) + 's : %.' + str(precision) + 'g' 
        f_str = '\t%' + str(l) + 's : %s' 
        for it in sorted([ (str(k),v) for k,v in d.items()]) :
            if type(it[1]) == float :
                print(f_flt % it, file=output)
            else :
                print(f_str % it, file=output)
        return 0
    else :
        print (d, file=output)
        return 1

def main( filename, knowledge=NNF, semiring=None, parser_class=DefaultPrologParser, debug=False ) :
    logger = logging.getLogger('problog')

    try :
        with Timer('Total time to processing model'):
          parser = parse_class(ExtendedPrologFactory())
          formula = knowledge.createFrom(PrologFile(filename, parser=parser))
        with Timer('Evaluation'):
          result = formula.evaluate(semiring=semiring)
        return True, result
    except Exception as err :
        return False, process_error(err, debug=debug)

def argparser() :
    import argparse
    
    class inputfile(str) : pass
    class outputfile(str) : pass
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', metavar='MODEL', nargs='*', type=inputfile)
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    parser.add_argument('--knowledge', '-k', choices=('sdd','nnf'), default=None, help="Knowledge compilation tool.")
    parser.add_argument('--symbolic', action='store_true', help="Use symbolic evaluation.")
    parser.add_argument('--logspace', action='store_true', help="Use log space evaluation.")
    parser.add_argument('--output', '-o', help="Output file (default stdout)", type=outputfile)
    parser.add_argument('--recursion-limit', help="Set recursion limit. Increase this value if you get an unbounded program error. (default: %d)" % sys.getrecursionlimit(), default=sys.getrecursionlimit(), type=int)
    parser.add_argument('--timeout', '-t', type=int, default=0, help="Set timeout (in seconds, default=off).")
    parser.add_argument('--debug', '-d', action='store_true', help="Enable debug mode (print full errors).")
    return parser

if __name__ == '__main__' :
    parser = argparser()
    args = parser.parse_args()

    logger = logging.getLogger('problog')
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if args.debug :
        logger.setLevel(logging.DEBUG)
        logger.debug('Output level: DEBUG')
    elif args.verbose == None:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
        logger.info('Output level: INFO')
    else:
        logger.setLevel(logging.DEBUG)
        logger.debug('Output level: DEBUG')

    if args.recursion_limit :
        sys.setrecursionlimit(args.recursion_limit)

    if args.output == None :
        output = sys.stdout
    else :
        output = open(args.output, 'w')
    
    parse_class = DefaultPrologParser
    
    if args.timeout : start_timer(args.timeout)

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

    if args.filenames[0] == 'install' :
        from problog import setup
        setup.install()
    elif args.filenames[0] == 'info' :
        from problog.core import list_transformations
        list_transformations()
    elif args.filenames[0] == 'unittest' :
        import unittest
        test_results = unittest.TextTestResult(sys.stderr, False, 1)
        unittest.TestLoader().discover(os.path.dirname(__file__)).run(test_results)
    else :
        if args.knowledge == 'nnf' :
            knowledge = NNF
        elif args.knowledge == 'sdd' :
            knowledge = SDD
        elif args.knowledge == None :
            if SDD.is_available() and not args.symbolic :
                logger.info('Using SDD path')
                knowledge = SDD
            else :
                logger.info('Using d-DNNF path')
                knowledge = NNF
        else :
            raise ValueError("Unknown option for --knowledge: '%s'" % args.knowledge)
        
        if args.symbolic :
            semiring = SemiringSymbolic()
        elif args.logspace:
            semiring = SemiringLogProbability()
        else :
            semiring = None
    
        for filename in args.filenames :
            if len(args.filenames) > 1 : print ('Results for %s:' % filename)
            retcode = print_result( main(filename, knowledge, semiring, parse_class, args.debug), output )
            if len(args.filenames) == 1 : sys.exit(retcode)

    if args.output != None : output.close()
    
    if args.timeout : stop_timer()
    
