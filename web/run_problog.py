#! /usr/bin/env python

from __future__ import print_function

import os, sys, subprocess, traceback

sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../')))

from problog.parser import PrologParser
from problog.program import PrologFile, PrologFactory
from problog.evaluator import SemiringSymbolic, Evaluator
from problog.nnf_formula import NNF
# from problog.sdd_formula import SDD

def print_result( d, output, precision=8 ) :
    success, d = d
    if success :
        import json
        print (200, 'application/json', json.dumps(d), file=output)
    else :
        print (400, 'application/json', json.dumps(d), file=output)
    return 0 
    

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


def main( filename) :

    try :
        model = PrologFile(filename, parser=PrologParser(PrologFactory()))
        formula = NNF.createFrom( model )
        result = formula.evaluate()
        return True, result
    except Exception as err :
        return False, process_error(err)
        
if __name__ == '__main__' :
    import argparse
        
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='MODEL')
    parser.add_argument('output', metavar='OUTPUT')
    
    args = parser.parse_args()

    result = main(args.filename)
    with open(args.output, 'w') as output :        
        retcode = print_result( result , output )

    # Always exit with code 0
