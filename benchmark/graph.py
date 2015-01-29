#! /usr/bin/env python

from __future__ import print_function

import os
import sys
import logging

problog_path =os.path.abspath(os.path.join(os.path.dirname(__file__),'../') )
sys.path.insert(0, problog_path)

import problog

from problog.util import Timer
from problog.logic import Term, Constant

from problog.engine_event import EventBasedEngine
from problog.engine_stack import StackBasedEngine

class GraphFile(problog.logic.LogicProgram) :
    
    def __init__(self, filename) :
        self.filename = filename
        
    def __iter__(self) :
        
        with open(self.filename) as f :
            extra_lines = []
            functor = None
            for line in f :
                line = line.strip()
                if line.startswith('== ') :
                    functor = line[2:-2].strip()
                elif line.startswith('======') :
                    functor = None
                elif functor :
                    parts = line.split()
                    if len(parts) == 3 :
                        yield Term(functor, Term("'" + parts[0] + "'"), Term("'" + parts[1] + "'"), p=Constant(float(parts[2])))
                    elif len(parts) == 2 :
                        print (line)
                        yield Term(functor, Term("'" + parts[0] + "'"), Term("'" + parts[1] + "'"))
                    else :
                        pass
                else :
                    extra_lines.append(line)
        pl = problog.program.PrologString('\n'.join(extra_lines))
        for line in pl :
            yield line
            
class DummyLF(problog.formula.LogicFormula) :
    
    def addAtom(self, *args, **kwdargs) :
        return 0
        
    def addOr(self, *args, **kwdargs) :
        return 0
        
    def addAnd(self, *args, **kwdargs) :
        return 0

def main(filename, engine='1', n=0, L=0) :
    
    logger = logging.getLogger('problog')
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    
    if filename.endswith('.pgraph') :
        pl = GraphFile(filename)
    else :
        pl = problog.program.PrologFile(filename)
    
    with Timer('Read file') :
        lines = list(pl)
    
    if engine == 'e1' :
        engine = EventBasedEngine()
        stats = None    # Engine doesn't support stats
    elif engine == 'e2' :
        engine = StackBasedEngineOld()
        stats = [0] * 5
    else :
        engine = StackBasedEngine()
        stats = [0] * 5
    
    with Timer('Compile file') :
        db = engine.prepare(lines)
    
    find_query = problog.logic.Term('query',None)
    
    gp = DummyLF()
    
    with Timer('Retrieving queries') :
        queries = [ r[0] for r in engine.query(db, find_query) ]
        
    if n > 0 : queries = queries[:n]
    
    logger.info('Number of queries: %s' % len(queries))
    
    non_false = 0
    try :
        with Timer('Evaluating queries') :
            for i, query in enumerate(queries) :
                if L > 0 : query = query.withArgs( *( query.args + (Constant(L),))  )
                with Timer('Evaluating query %s: %s' % (i+1, query)) :
                    result = engine.ground(db, query, gp=gp, label='query', stats=stats)
    except KeyboardInterrupt :
        logger.info('Interrupted by user.')
    
    failed = len( [ r for r,n in gp.queries() if n == None ] )
    
    logger.info('Ground program size: %s' % len(gp))
    logger.info('Number of failed queries: %s' % failed)
        
    if stats :
        logger.info('Engine message statistics: %s' % ', '.join( ('%s: %s' % (t,stats[i]) ) for i,t in enumerate('rceoC') ) )
    print (engine.stats)

if __name__ == '__main__' :
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('filename')
    p.add_argument('-e', '--engine', choices=('e1','e2','e3'), default='e3')
    p.add_argument('-n', type=int, default=0)
    p.add_argument('-L', type=int, default=0)
    
    
    args = p.parse_args()
    
    main(args.filename, args.engine, args.n, args.L)
        

