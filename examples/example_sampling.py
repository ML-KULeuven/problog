#! /usr/bin/env python

"""Sample possible worlds from a given ProbLog model.

    Note: currently, evidence is ignored.

"""


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
    """Sampling engine.
    
    This engine operates much in the same way as the EventBasedEngine, except for how it evaluates facts and choices.
     
    """
    
    def __init__(self, **kwdargs) :
        EventBasedEngine.__init__(self, **kwdargs)
        
        self.reset()
     
    def reset(self) :
        self.facts = {}
        self.groups = {}
        self.probability = 1.0
        
    def sample(self, model, queries=None) :
        """Sample one assignment to the queries of the given model.
        
        Returns a tuple containing:
            queries : list of pairs (name, value)
            facts: list of probabilistic facts with their sampled value (name, value)
            probability: overall probability of the assignment
        """
        
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
        
        # Take into account remaining probabilities of no-choice nodes for annotated disjunctions.
        for k, p in self.groups.items() :
            if p != None :
                self.probability *= p
                
        translate = lambda x : (x[0],x[1] == 0)
        
        print (self.facts, target.queries())
        
        facts = []
        for f, v in self.facts.items() :
            if v :
                node = db.getNode(f)
                if node.functor != 'query' :
                    if node.args :
                        facts.append('%s(%s)' % (node.functor, ', '.join(map(str,node.args) ) ) )
                    else :
                        facts.append('%s' % (node.functor ) )
        
        
        result = map(translate,target.queries()), facts, self.probability

        self.reset()
        
        return result
        
    def _eval_fact( self, db, gp, node_id, node, call_args, parent ) :
        try :
            # Verify that fact arguments unify with call arguments.
            for a,b in zip(node.args, call_args) :
                unify(a, b)
            # Successful unification: notify parent callback.
            
            # Retrieve fact value from the cache
            if not node_id in self.facts :
                # Fact was not found in cache. Fix its value.
                if node.probability == None : 
                    # Deterministic fact.
                    value = True
                else :
                    # Probabilistic fact.
                    p = float(node.probability)
                    value = ( random.random() <= p )
                    if value :
                        self.probability *= p
                    else :
                        self.probability *= (1-p)
                self.facts[node_id] = value
                
            # If fact is True
            if value :
                parent.newResult( node.args, 0 )
        except UnifyError :
            # Failed unification: don't send result.
            pass
        # Send complete message.
        parent.complete()    

    def _eval_choice( self, db, gp, node_id, node, call_args, parent ) :
        # This never fails.
        # Choice is ground so result is the same as call arguments.
        result = tuple(call_args)
        # Ground probability.
        probability = instantiate( node.probability, call_args )
        # Create a new atom in ground program.
        # Atoms with same origin are mutually exclusive.
        origin = (node.group, result)
        
        # How does it work?
        #   The choices are made sequentially. 
        #   If the first choice yields true, then all remaining choices are 0.0.
        #   If the first choice yields false, then the remaining probabilities have to be adjusted.
        #   This is achieved by keeping track of the remaining probability for each origin. (in self.groups)
        #   Initially, the remaining probability is 1.0. 
        #   When a choice atom is set to True, the remaining probability for that group becomes None (which indicates choice made, equivalent to 0.0).
        #   When a choice atom is set to False, the probability of that atom is subtracted from the current value (nothing is done when the remaining probability is already None).
        #   If no choice is made (implicit additional choice) the remaining probability should be taken into account when computing the sample's probability.
        # Example: 0.1::a; 0.2::b; 0.3::c <- ...
        #   When we encounter 'b' and set it to false (with probability 0.8), we update r to 0.8.
        #   The probabilities are adjusted to 0.125::a; 0.375::c <- ...
        #   When we now encounter 'a' and set it to false (with probability 0.875), we update r to 0.7 (using the original probability of a).
        #   We update the probability of 'c' to 0.42857... ( = 0.3/0.7 )
        #   When we now encounter 'c' and set it to false (with probability 0.57142), we update r to 0.4 (using the original probability of a).
        #   The remaining probability for not choosing a,b or c is indeed 0.4.

        key = (node.group, result, node.choice)
        if not key in self.facts : 
            p = float(instantiate( node.probability, call_args ))
            if origin in self.groups :
                r = self.groups[origin] # remaining probability in the group
            else :
                r = 1.0                    
            
            if r == None or r < 1e-8 :   # r is too small or another choice was made for this origin
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
            # Note: if value == False: no update of probability.    
            
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
            if v :
                print ('%s.' % k)
        
        if with_facts :
            print ('\n'.join(map((lambda s : '%s.' % s) ,facts)))
        
        print ('%%Probability: %.4g' % probability)
    
if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('filename')
    parser.add_argument('-N', type=int, default=1, help="Number of samples.")
    parser.add_argument('--with-facts', action='store_true', help="Also output choice facts (default: just queries).")
    args = parser.parse_args()
    
    
    sample( args.filename, args.N, args.with_facts )