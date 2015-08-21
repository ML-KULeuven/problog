#! /usr/bin/env python

"""Sample possible worlds from a given ProbLog model.

    Note: currently, evidence is ignored.

"""


from __future__ import print_function

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..' ) )

from problog.program import PrologFile
from problog.logic import Term
from problog.engine import DefaultEngine
from problog.formula import LogicFormula
import random

# TODO support evidence

class SampledFormula(LogicFormula) :
    
    def __init__(self) :
        LogicFormula.__init__(self)
        self.facts = {}
        self.groups = {}
        self.probability = 1.0
        
    def addAtom( self, identifier, probability, group=None ) :
        if group == None :  # Simple fact
            if not identifier in self.facts :
                if probability == None :    # Deterministically true
                    value = True
                else :
                    p = float(probability)
                    value = ( random.random() <= p)
                    if value :
                        self.probability *= p
                    else :
                        self.probability *= (1-p)
                self.facts[identifier] = value
            else :
                value = self.facts[identifier]
        else :
            group, result, choice = identifier
            origin = (group, result)
            
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
            
            if not identifier in self.facts :
                p = float(probability)
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
                
                self.facts[identifier] = value
            else :
                value = self.facts[identifier]
                
        if value :
            return 0    # True
        else :
            return None # False
            
    def computeProbability(self) :
        for k, p in self.groups.items() :
            if p != None :
                self.probability *= p
        self.groups = {}
            
    def toString(self, db, with_facts, oneline) :
        self.computeProbability()
        lines = []
        for k, v in self.queries() :
            if v == 0 : lines.append(str(k) + '.')
        if with_facts :
            for k, v in self.facts.items() :
                if v : lines.append(str(translate(db, k)) + '.')
            
        if oneline :
            sep = ' '
        else :
            sep = '\n'
        return '%s%s%% Probability: %.8g' % (sep.join(set(lines)),sep,self.probability)

def translate(db, atom_id) :
    if type(atom_id) == tuple :
        atom_id, args, choice = atom_id
        return Term('ad_%s_%s' % (atom_id, choice), *args)
    else :
        node = db.get_node(atom_id)
        return Term(node.functor, *node.args)

def sample( filename, N=1, with_facts=False, oneline=False ) :
    pl = PrologFile(filename)
    
    engine = DefaultEngine()
    db = engine.prepare(pl)
    
    for i in range(0, N) :
        result = engine.ground_all(db, target=SampledFormula())
        print ('====================')
        print (result.toString(db, with_facts, oneline))

def sample_object(pl, N=1):
    engine = DefaultEngine()
    db = engine.prepare(pl)
    result = [engine.ground_all(db, target=SampledFormula()) for i in range(N)]
    return result, db

def estimate( filename, N=1 ) :
    from collections import defaultdict
    pl = PrologFile(filename)
    
    engine = DefaultEngine()
    db = engine.prepare(pl)
    
    estimates = defaultdict(float)
    counts = 0.0
    for i in range(0, N) :
        result = engine.ground_all(db, target=SampledFormula())
        for k, v in result.queries() :
            if v == 0 :
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
