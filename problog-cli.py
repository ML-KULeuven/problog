#! /usr/bin/env python

from __future__ import print_function

import os, sys, subprocess, traceback, logging

from problog.program import PrologFile, ExtendedPrologFactory
from problog.evaluator import SemiringSymbolic, SemiringLogProbability, Evaluator
from problog.nnf_formula import NNF
from problog.sdd_formula import SDD
from problog.util import Timer, start_timer, stop_timer
from problog.core import process_error
from problog.parser import DefaultPrologParser, FastPrologParser


def print_result( d, output, precision=8 ) :    
    success, d = d
    if success :
        if not d : return 0 # no queries
        l = max( len(k) for k in d )
        f_flt = '\t%' + str(l) + 's : %.' + str(precision) + 'g' 
        f_str = '\t%' + str(l) + 's : %s' 
        for it in sorted(d.items()) :
            if type(it[1]) == float :
                print(f_flt % it, file=output)
            else :
                print(f_str % it, file=output)
        return 0
    else :
        print ('Error:', d, file=output)
        return 1

def main( filename, knowledge=NNF, semiring=None, parser_class=DefaultPrologParser ) :
    logger = logging.getLogger('problog')

    try :
        with Timer('Total time to processing model'):
          parser = parse_class(ExtendedPrologFactory())
          formula = knowledge.createFrom(PrologFile(filename, parser=parser))
        with Timer('Evaluation'):
          result = formula.evaluate(semiring=semiring)
        return True, result
    except Exception as err :
        return False, process_error(err)

def argparser() :
    import argparse
    
    class inputfile(str) : pass
    class outputfile(str) : pass
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', metavar='MODEL', nargs='+', type=inputfile)
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    parser.add_argument('--knowledge', '-k', choices=('sdd','nnf'), default=None, help="Knowledge compilation tool.")
    parser.add_argument('--symbolic', action='store_true', help="Use symbolic evaluation.")
    parser.add_argument('--logspace', action='store_true', help="Use log space evaluation.")
    parser.add_argument('--output', '-o', help="Output file (default stdout)", type=outputfile)
    parser.add_argument('--recursion-limit', help="Set recursion limit. Increase this value if you get an unbounded program error. (default: %d)" % sys.getrecursionlimit(), default=sys.getrecursionlimit(), type=int)
    parser.add_argument('--faster-parser', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--timeout', '-t', type=int, default=0, help="Set timeout (in seconds, default=off).")
    return parser

if __name__ == '__main__' :
    parser = argparser()
    args = parser.parse_args()

    logger = logging.getLogger('problog')
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if args.verbose == None:
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
    if args.faster_parser : parse_class = FastPrologParser
    
    if args.timeout : start_timer(args.timeout)
    
    if args.filenames[0] == 'install' :
        from problog import setup
        setup.install()
    elif args.filenames[0] == 'info' :
        from problog.core import list_transformations
        list_transformations()
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
            try :
                if len(args.filenames) > 1 : print ('Results for %s:' % filename)
                retcode = print_result( main(filename, knowledge, semiring, parse_class), output )
                if len(args.filenames) == 1 : sys.exit(retcode)
            except subprocess.CalledProcessError :
                print ('error')

    if args.output != None : output.close()
    
    if args.timeout : stop_timer()
    
