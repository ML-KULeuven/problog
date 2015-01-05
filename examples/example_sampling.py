#! /usr/bin/env python

from __future__ import print_function

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..' ) )

from problog.program import PrologFile
from problog.logic import Term
from problog.engine import EventBasedEngine, UnifyError, unify, instantiate
from problog.formula import LogicFormula
from problog.interface import *
import random


# TODO support evidence

class SamplingEngine(EventBasedEngine) :
    
    def __init__(self, **kwdargs) :
        EventBasedEngine.__init__(self, **kwdargs)
        
        self.reset()
     
    def reset(self) :
        self.facts = {}
        self.groups = {}
        self.probability = 1.0
        
    def sample(self, model, queries=None) :
        engine = self
        db = engine.prepare(model)
    
        if queries == None :
            queries = [ q[0] for q in engine.query(db, Term( 'query', None )) ]
    
        # if evidence == None :
        #     evidence = engine.query(db, Term( 'evidence', None, None ))
        
        target = LogicFormula()

        for query in queries :
            target = engine.ground(db, query, target, label=LABEL_QUERY)

        # for query in evidence :
        #     if str(query[1]) == 'true' :
        #         target = engine.ground(db, query[0], target, label=LABEL_EVIDENCE_POS)
        #     elif str(query[1]) == 'false' :
        #         target = engine.ground(db, query[0], target, label=LABEL_EVIDENCE_NEG)
        #     else :
        #         target = engine.ground(db, query[0], target, label=LABEL_EVIDENCE_MAYBE)
        
        for k, p in self.groups.items() :
            if p != None :
                self.probability *= p
        result = target.queries(), self.facts, self.probability

        self.reset()
        
        return result
        
    def _eval_fact( self, db, gp, node_id, node, call_args, parent ) :
        try :
            # Verify that fact arguments unify with call arguments.
            for a,b in zip(node.args, call_args) :
                unify(a, b)
            # Successful unification: notify parent callback.
            
            if not node_id in self.facts :
                if node.probability == None : 
                    value = True
                else :
                    p = float(node.probability)
                    value = ( random.random() <= p )
                    if value :
                        self.probability *= p
                    else :
                        self.probability *= (1-p)
                self.facts[node_id] = value
            if value :
                parent.newResult( node.args, 0 )
        except UnifyError :
            # Failed unification: don't send result.
            pass
        # Send complete message.
        parent.complete()    

    def _eval_choice( self, db, gp, node_id, node, call_args, parent ) :
        # TODO doesn't work if remaining choice (of false) is 0
        
        # This never fails.
        # Choice is ground so result is the same as call arguments.
        result = tuple(call_args)
        # Ground probability.
        probability = instantiate( node.probability, call_args )
        # Create a new atom in ground program.
        origin = (node.group, result)
        
        key = (node.group, result, node.choice)
        if not key in self.facts : 
            p = float(instantiate( node.probability, call_args ))
            if origin in self.groups :
                r = self.groups[origin] # remaining probability in the group
            else :
                r = 1.0                    
            
            if r == None or r < 1e-8 :   # r is too small
                value = False
                q = 0
            else :
                q = p/r
                value = ( random.random() <= q )
            if value :
                self.probability *= p
                self.groups[origin] = None   # Other choices in group are not allowed
            elif r != None :
                self.groups[origin] = r-p   # Adjust remaining probability
            self.facts[node_id] = value

        if value :
            parent.newResult( result, 0 )
        parent.complete()

def sample( filename, N=1, with_facts=False ) :
    pl = PrologFile(filename)
    
    engine = SamplingEngine(builtins=True)
    db = engine.prepare(pl)
    
    for i in range(0, N) :
        queries, facts, probability = engine.sample(db)
        print ('====================')
        for k, v in queries :
            if v == 0 :
                print ('%s.' % k)
        
        if with_facts :
            for f, v in facts.items() :
                if v :
                    node = db.getNode(f)
                    if node.functor != 'query' :
                        if node.args :
                            print ('%s(%s).' % (node.functor, ', '.join(map(str,node.args) ) ) )
                        else :
                            print ('%s.' % (node.functor ) )
        
        print ('%%Probability: %.4g' % probability)
    
if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('filename')
    parser.add_argument('-N', type=int, default=1, help="Number of samples.")
    parser.add_argument('--with-facts', action='store_true', help="Also output choice facts (default: just queries).")
    args = parser.parse_args()
    
    
    sample( args.filename, args.N, args.with_facts )