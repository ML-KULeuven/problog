#! /usr/bin/env python

from __future__ import print_function

import os, sys, subprocess, traceback, logging

from problog.program import PrologFile, ExtendedPrologFactory
from problog.evaluator import SemiringSymbolic, Evaluator
from problog.nnf_formula import NNF
from problog.sdd_formula import SDD
from problog.util import Timer
from problog.parser import DefaultPrologParser


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
            
def process_error( err ) :
    """Take the given error raise by ProbLog and produce a meaningful error message."""
    
    err_type = type(err).__name__
    if err_type == 'ParseException' :
        return 'Parsing error on %s:%s: %s.\n%s' % (err.lineno, err.col, err.msg, err.line )
    elif err_type == 'UnknownClause' :
        return 'Predicate undefined: \'%s\'.' % (err )
    elif err_type == 'PrologInstantiationError' :
        return 'Arithmetic operation on uninstantiated variable.' 
    elif err_type == 'UnboundProgramError' :
        return 'Unbounded program or program too large.'
    else :
        traceback.print_exc()
        return 'Unknown error: %s' % (err_type)


def main( filename, knowledge=NNF, semiring=None ) :
    logger = logging.getLogger('problog')

    try :
        with Timer('Total time to processing model'):
          parser = DefaultPrologParser(ExtendedPrologFactory())
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
    parser.add_argument('--knowledge', '-k', choices=('sdd','nnf'), default='nnf', help="Knowledge compilation tool.")
    parser.add_argument('--symbolic', action='store_true', help="Use symbolic evaluation.")
    parser.add_argument('--output', '-o', help="Output file (default stdout)", type=outputfile)
    parser.add_argument('--recursion-limit', help="Set recursion limit. Increase this value if you get an unbounded program error. (default: %d)" % sys.getrecursionlimit(), default=sys.getrecursionlimit(), type=int)
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
    
    if args.filenames[0] == 'install' :
        from problog import setup
        setup.install()
    else :
        if args.knowledge == 'nnf' :
            knowledge = NNF
        elif args.knowledge == 'sdd' :
            knowledge = SDD
        else :
            raise ValueError("Unknown option for --knowledge: '%s'" % args.path)
        
        if args.symbolic :
            semiring = SemiringSymbolic()
        else :
            semiring = None
    
        for filename in args.filenames :
            try :
                if len(args.filenames) > 1 : print ('Results for %s:' % filename)
                retcode = print_result( main(filename, knowledge, semiring), output )
                if len(args.filenames) == 1 : sys.exit(retcode)
            except subprocess.CalledProcessError :
                print ('error')

    if args.output != None : output.close()
    
