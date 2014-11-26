#! /usr/bin/env python

from __future__ import print_function

import os, sys, subprocess

# Make sure ProbLog is on the PYTHON_PATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..' ) )

from problog import PrologFile, Evaluator
from problog.evaluator import NNF, SemiringSymbolic
from problog.logic.sdd_formula import SDD



def print_result( d, precision=8 ) :
    l = max( len(k) for k in d )
    f_flt = '\t%' + str(l) + 's : %.' + str(precision) + 'g' 
    f_str = '\t%' + str(l) + 's : %s' 
    for it in sorted(d.items()) :
        if type(it[1]) == float :
            print(f_flt % it)
        else :
            print(f_str % it)

def main( filename, knowledge=NNF, semiring=None ) :
    
    print ('Results for %s:' % filename)
        
    formula = knowledge.createFrom( PrologFile(filename) )
    
    result = formula.evaluate(semiring=semiring)
    
    print_result(result)
        
if __name__ == '__main__' :
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+')
    parser.add_argument('--knowledge', '-k', choices=('sdd','nnf'), default='nnf')
    parser.add_argument('--symbolic', action='store_true')
    
    args = parser.parse_args()
    
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
            main(filename, knowledge, semiring)
        except subprocess.CalledProcessError :
            print ('error')