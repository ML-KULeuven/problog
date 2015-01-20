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


from problog.engine_alt import EvalNode, StackBasedEngine

class EvalFact(EvalNode) :
    def __call__(self) :
        actions = []
        try :
            # Verify that fact arguments unify with call arguments.
            for a,b in zip(self.node.args, self.context) :
                unify(a, b)
            # Successful unification: notify parent callback.
            # Retrieve fact value from the cache
            if not self.node_id in self.engine.facts :
                # Fact was not found in cache. Fix its value.
                if self.node.probability == None : 
                    # Deterministic fact.
                    value = True
                else :
                    # Probabilistic fact.
                    p = float(self.node.probability)
                    value = ( random.random() <= p )
                    if value :
                        self.engine.probability *= p
                    else :
                        self.engine.probability *= (1-p)
                self.engine.facts[self.node_id] = value
            else :
                value = self.engine.facts[self.node_id]
                
            # If fact is True
            if value :
                actions += self.notifyResult( self.node.args, 0, True )
            else :
                actions += self.notifyComplete()
        except UnifyError :
            # Failed unification: don't send result.
            # Send complete message.
            actions += self.notifyComplete()
        return True, actions        # Clean up, actions


        
class EvalChoice(EvalNode) :
    def __call__(self) :
        # This never fails.
        # Choice is ground so result is the same as call arguments.
        result = tuple(self.context)
        # Ground probability.
        probability = instantiate( self.node.probability, result )
        # Create a new atom in ground program.
        # Atoms with same origin are mutually exclusive.
        origin = (self.node.group, result)
        
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

        key = (self.node.group, result, self.node.choice)
        if not key in self.engine.facts : 
            p = float(instantiate( self.node.probability, result ))
            if origin in self.engine.groups :
                r = self.engine.groups[origin] # remaining probability in the group
            else :
                r = 1.0                    
            
            if r == None or r < 1e-8 :   # r is too small or another choice was made for this origin
                value = False
                q = 0
            else :
                q = p/r
                value = ( random.random() <= q )
            if value :
                self.engine.probability *= p
                self.engine.groups[origin] = None   # Other choices in group are not allowed
            elif r != None :
                self.engine.groups[origin] = r-p   # Adjust remaining probability
            # Note: if value == False: no update of probability.    
            
            self.engine.facts[key] = value
        else :
            value = self.engine.facts[key]

        if value :
            return True, self.notifyResult(result, 0, True)
        else :
            return True, self.notifyComplete()

class SamplingEngine(StackBasedEngine) :
    """Sampling engine.
    
    This engine operates much in the same way as the EventBasedEngine, except for how it evaluates facts and choices.
     
    """
    
    def __init__(self, **kwdargs) :
        StackBasedEngine.__init__(self, **kwdargs)
        # Override standard fact and choice handlers
        self.node_types['fact'] = EvalFact
        self.node_types['choice'] = EvalChoice
        
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
        facts = []
        for f, v in self.facts.items() :
            if v :                
                if type(f) == tuple :
                    node = db.getNode(f[0])
                    args = f[1]
                else :
                    node = db.getNode(f)
                    args = node.args
                if node.functor != 'query' :
                    if args :
                        facts.append('%s(%s)' % (node.functor, ', '.join(map(str,args) ) ) )
                    else :
                        facts.append('%s' % (node.functor ) )
        
        
        result = map(translate,target.queries()), facts, self.probability
        
        self.reset()
        
        return result
        
def sample( filename, N=1, with_facts=False, oneline=False ) :
    pl = PrologFile(filename)
    
    engine = SamplingEngine(builtins=True)
    db = engine.prepare(pl)
    
    for i in range(0, N) :
        queries, facts, probability = engine.sample(db)
        lines = []
        for k, v in queries :
            if v :
                lines.append('%s.' % k)
        
        if with_facts :
            for f in facts :
                lines.append('%s.' % f)
        
        if oneline :
            print (' '.join(lines), '%% P=%.4g' % probability)
        else :
            print ('====================')
            print ('\n'.join(lines))
            print ('%%Probability: %.4g' % probability)
            
def estimate( filename, N=1 ) :
    from collections import defaultdict
    
    
    pl = PrologFile(filename)
    
    engine = SamplingEngine(builtins=True)
    db = engine.prepare(pl)
    
    estimates = defaultdict(float)
    counts = 0.0
    for i in range(0,N) :
        queries, facts, prob = engine.sample(db)
        for k, v in queries :
            if v :
                estimates[k] += 1.0
        counts += 1.0
    
    for k in estimates :
        estimates[k] = estimates[k] / counts
    return estimates

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

    
if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('filename')
    parser.add_argument('-N', type=int, default=1, help="Number of samples.")
    parser.add_argument('--with-facts', action='store_true', help="Also output choice facts (default: just queries).")
    parser.add_argument('--oneline', action='store_true', help="Format samples on one line.")
    parser.add_argument('--estimate', action='store_true')
    args = parser.parse_args()
    
    if args.estimate :
        result = estimate( args.filename, args.N )
        print_result((True,result), sys.stdout)
    else :
        sample( args.filename, args.N, args.with_facts, args.oneline )