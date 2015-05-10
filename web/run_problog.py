#! /usr/bin/env python

from __future__ import print_function

import os, sys, subprocess, traceback, json

sys.setrecursionlimit(10000)

sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../')))

from problog.parser import DefaultPrologParser
from problog.program import PrologFile, ExtendedPrologFactory
from problog.evaluator import SemiringLogProbability, Evaluator
from problog.nnf_formula import NNF
from problog.sdd_formula import SDD
from problog.core import process_error, GroundingError, ParseError

def print_result( d, output, precision=8 ) :
    success, d = d
    if success :
        d['SUCCESS'] = True
        d = { str(k) : round(v,12) for k,v in d.items() }
        print (200, 'application/json', json.dumps(d), file=output)
    else :
        #print (400, 'application/json', json.dumps(d), file=output)
        d['SUCCESS'] = False
        d = { str(k) : v for k,v in d.items() }
        print (200, 'application/json', json.dumps(d), file=output)
    return 0 
    
def process_error( err ) :
    """Take the given error raise by ProbLog and produce a meaningful error message."""
    err_type = type(err).__name__
    if err_type == 'MemoryError':
        return { 'message': 'ProbLog exceeded the available memory limit.' }
    elif err_type == 'ParseException' :
        return { 'message': 'Parsing error on %s:%s: %s.\n%s' % (err.lineno, err.col, err.msg, err.line ), 'lineno' : err.lineno, 'col': err.col }
    elif isinstance(err, ParseError) :
        return { 'message': 'Parsing error on %s:%s: %s.' % (err.lineno, err.col, err.msg ), 'lineno' : err.lineno, 'col' : err.col } 
    elif isinstance(err, GroundingError) :
        try :
            location = err.location
            if location :
                return { 'message': 'Error during grounding: %s' % err, 'lineno' : location[0], 'col' : location[1] } 
            else :
                return { 'message': 'Error during grounding: %s' % err }
        except AttributeError :
            return { 'message': 'Error during grounding: %s' % err }
    else :
        traceback.print_exc()
        return { 'message' : 'Unknown error: %s' % (err_type) }


def main( filename) :

    try :
        model = PrologFile(filename, parser=DefaultPrologParser(ExtendedPrologFactory()))
        formula = NNF.createFrom( model )
        result = formula.evaluate(semiring=SemiringLogProbability())
        return True, result
    except Exception as err :
        return False, {'err':process_error(err)}
        
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
