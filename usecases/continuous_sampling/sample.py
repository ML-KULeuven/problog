#! /usr/bin/env python

# Distributional Clauses for ProbLog 2.1
# ======================================
#
#   Not really distributional clauses.
#   Uses query-based sampling.
#   
#   Concept:
#       Each term has a value which is stored in the SampledFormula.
#       When an probabilistic atom or choice atom is evaluated:
#           - if probability is boolean discrete:       determine Yes/No, store in formula and return 0 (True) or None (False)
#           - if probability is non-boolean discrete:   determine value, store value in formula and return key to value
#       Adds builtin sample(X,S) that calls X and returns the sampled value in S.

from __future__ import print_function

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))

from problog.program import PrologFile
from problog.logic import Term, Constant
from problog.engine import DefaultEngine
from problog.engine_builtin import check_mode
from problog.formula import LogicFormula
import random

def sample_value( term ) :
    # TODO add discrete distribution
    if term.functor == 'normal' :
        a,b = map(float, term.args)
        return Constant(random.normalvariate(a,b))
    elif term.functor == 'poisson' :
        import numpy.random
        a = map(float, term.args)
        return Constant(numpy.random.poisson(a)[0])
    elif term.functor == 'exponential' :
        a = map(float, term.args)
        return Constant(random.expovariate(a))
    elif term.functor == 'beta' :
        a,b = map(float, term.args)
        return Constant(random.betavariate(a,b))
    elif term.functor == 'gamma' :
        a,b = map(float, term.args)
        return Constant(random.gammavariate(a,b))
    elif term.functor == 'uniform' :
        a,b = map(float, term.args)
        return Constant(random.randrange(a,b))
    elif term.functor == 'constant' :
        return term.args[0]
    else :
        raise ValueError("Unknown distribution: '%s'" % term.functor)

class SampledFormula(LogicFormula) :
    
    def __init__(self) :
        LogicFormula.__init__(self)
        self.facts = {}
        self.groups = {}
        self.probability = 1.0  # Try to compute
        self.values = []
        
    def _isSimpleProbability(self, term) :
        return type(term) == float or term.isConstant()
        
    def addValue(self, value) :
        self.values.append(value)
        return len(self.values)
        
    def getValue(self, key) :
        if key == 0 :
            return None
        return self.values[key-1]
    
    def addAtom( self, identifier, probability, group=None ) :
        if probability is None : return 0
        
        if group == None :  # Simple fact
            if not identifier in self.facts :
                if self._isSimpleProbability(probability) :
                    p = random.random()
                    value = p < float(probability)
                    if value :
                        result_node = self.TRUE
                    else :
                        result_node = self.FALSE
                else :
                    value = sample_value( probability )
                    result_node = self.addValue( value )
                self.facts[identifier] = result_node
                return result_node
            else :
                return self.facts[identifier]
        else :
            group, result, choice = identifier
            origin = (group, result)
            if not identifier in self.facts :
                if self._isSimpleProbability(probability) :
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
                    if value :
                        result_node = self.TRUE
                    else :
                        result_node = self.FALSE
                else :
                    value = sample_value( probability )
                    result_node = self.addValue( value )
                self.facts[identifier] = result_node
                return result_node
            else :
                return self.facts[identifier]
                
    def addAnd( self, content ) :
        i = 0
        for c in content :
            if c != None and c != 0 : i += 1
            if i > 1 : raise ValueError("Can't combine sampled predicates. Use sample/2 to extract sample.")
        return LogicFormula.addAnd(self, content)
        
    def addOr( self, content, **kwd ) :
        i = 0
        for c in content :
            if c != None and c != 0 : i += 1
            if i > 1 : raise ValueError("Can't combine sampled predicates. Make sure bodies are mutually exclusive.")
        return LogicFormula.addOr(self, content, **kwd)
                
            
    def toString(self, db, with_facts, oneline) :
        #self.computeProbability()
        lines = []
        for k, v in self.queries() :
            if not v is None : 
                val = self.getValue(v)
                if val is None :
                    lines.append('%s.' % (str(k) ))
                else :
                    lines.append('%s = %s.' % (str(k), val ))
        if with_facts :
            for k, v in self.facts.items() :
                if v : lines.append(str(translate(db, k)) + '.')
            
        if oneline :
            sep = ' '
        else :
            sep = '\n'
        return sep.join(set(lines))
        #return '%s%s%% Probability: %.8g' % (sep.join(set(lines)),sep,self.probability)
        
    def toTuples(self, db):
        lines = []
        for k, v in self.queries() :
            if not v is None : 
                val = self.getValue(v)
                if val is None :
                    lines.append((k.functor,) + k.args + (None,))
                else :
                    lines.append((k.functor,) + k.args + (val,))
        # if with_facts :
        #     for k, v in self.facts.items() :
        #         if v : lines.append(str(translate(db, k)) + '.')
            
        # if oneline :
        #     sep = ' '
        # else :
        #     sep = '\n'
        return list(set(lines))
        

def translate(db, atom_id) :
    if type(atom_id) == tuple :
        atom_id, args, choice = atom_id
        return Term('ad_%s_%s' % (atom_id, choice), *args)
    else :
        node = db.getNode(atom_id)
        return Term(node.functor, *node.args)

def builtin_sample(term, result, target=None, engine=None, callback=None, **kwdargs ) :
    check_mode( (term,result), 'cv', functor='sample' )
    # Find the define node for the given query term.
    term_call = term.withArgs( *(term.args ))
    results = engine.call( term_call, subcall=True, target=target, **kwdargs )
    actions = []
    n = len(term.args)
    for res, node in results :
        res1 = res[:n]
        res_pass = (term.withArgs(*res1), target.getValue(node))
        actions += callback.notifyResult( res_pass, 0, False)
    actions += callback.notifyComplete()
    return True, actions

def sample(model, N=1, with_facts=False, oneline=False, tuples=False ) :
    engine = DefaultEngine()
    engine.add_builtin('sample',2, builtin_sample)
    db = engine.prepare(model)
    
    for i in range(0, N) :
        result = engine.ground_all(db, target=SampledFormula())
        if tuples:
            yield result.toTuples(db)
        else:
            yield result.toString(db, with_facts, oneline)

# def sample_object(pl, N=1):
#     engine = DefaultEngine()
#     db = engine.prepare(pl)
#     result = [engine.ground_all(db, target=SampledFormula()) for i in range(N)]
#     return result, db
#
# def estimate( filename, N=1 ) :
#     from collections import defaultdict
#     pl = PrologFile(filename)
#
#     engine = DefaultEngine()
#
#
#     db = engine.prepare(pl)
#
#     estimates = defaultdict(float)
#     counts = 0.0
#     for i in range(0, N) :
#         result = engine.ground_all(db, target=SampledFormula())
#         for k, v in result.queries() :
#             if v == 0 :
#                 estimates[k] += 1.0
#         counts += 1.0
#
#     for k in estimates :
#         estimates[k] = estimates[k] / counts
#     return estimates

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
        pl = PrologFile(args.filename)
        for s in sample(pl, args.N, args.with_facts, args.oneline):
            print ('====================')
            print (s)
            