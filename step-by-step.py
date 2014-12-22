from __future__ import print_function

import sys
import time

from problog.program import PrologFile
from problog.logic import Term
from problog.evaluator import SemiringSymbolic, Evaluator
from problog.engine import DefaultEngine, EngineLogger, SimpleEngineLogger
from problog.nnf_formula import NNF
from problog.cnf_formula import CNF
from problog.sdd_formula import SDD

def ground(model, gp) :
    
    
    queries = engine.query(db, Term( 'query', None ))
    evidence = engine.query(db, Term( 'evidence', None, None ))
        
    for query in queries :
        gp = engine.ground(db, query[0], gp, label='query')

    for query in evidence :
        if str(query[1]) == 'true' :
            gp = engine.ground(db, query[0], gp, label='evidence')
        else :
            gp = engine.ground(db, query[0], gp, label='-evidence')

    return gp

class Timer(object) :
    
    def __init__(self, msg) :
        self.message = msg
        self.start_time = None
        
    def __enter__(self) :
        self.start_time = time.time()
        
    def __exit__(self, *args) :
        print ('%s: %.4fs' % (self.message, time.time()-self.start_time))


def main(filename) :
    
    model = PrologFile(filename)
    
    #EngineLogger.setClass(SimpleEngineLogger)
    
    engine = DefaultEngine()
    
    with Timer('parsing') :
        db = engine.prepare(model)
    
    with Timer('queries') :
        queries = engine.query(db, Term( 'query', None ))
    
    with Timer('evidence') :
        evidence = engine.query(db, Term( 'evidence', None, None ))
    
    print ('nr of queries:', len(queries))
    print ('nr of evidence:', len(evidence))
    
    with Timer('ground') :
        gp = None
        for i, query in enumerate(queries) :
            print ('ground query %d (%s):' % (i+1,query), '='*100)
            with Timer('ground query %d (%s):' % (i+1,query)) :
                gp = engine.ground(db, query[0], gp, label='query')
            
    
        for query in evidence :
            if str(query[1]) == 'true' :
                gp = engine.ground(db, query[0], gp, label='evidence')
            else :
                gp = engine.ground(db, query[0], gp, label='-evidence')
    
    print (gp)
    
    
    with Timer('acyclic') :
        gp = gp.makeAcyclic()
    
    print (gp)
                
    with Timer('convert') :
        cnf = CNF.createFrom(gp)
    #print (cnf.getNamesWithLabel())
                
    with Timer('compile') :
        nnf = NNF.createFrom(cnf)
    
    #print (nnf)
    
    with Timer('evaluate') :
        result = nnf.evaluate()
    
    for it in result.items() :
        print ('%s : %s' % (it))
    

if __name__ == '__main__' :
    main(sys.argv[1])