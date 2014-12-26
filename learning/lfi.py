from __future__ import print_function

import sys, os, random, math
from collections import defaultdict

sys.path.append(os.path.abspath( os.path.join( os.path.dirname(__file__), '../' ) ) )

class GreedyParameterLearner(object) :
    pass
    
# We have a base program.
    
    
# LFI learning: examples are sets of evidence
# Add example:
#   - ground base program together with queries (e)
#   => compile & store
# Evaluate:
#   - queries are tunable facts
#   - evaluate queries with current probabilities => tunable probabilities?


# Ideas:
#   - compute 'super' example with all evidence atoms
#       => compile once

# For each example => compile

from problog.core import LABEL_QUERY
from problog.interface import ground
from problog.engine import DefaultEngine
# from problog.nnf_formula import NNF as knowledge
from problog.sdd_formula import SDD as knowledge
from problog.evaluator import SemiringProbability
from problog.logic import Term, Var, Constant, Clause, AnnotatedDisjunction
from problog.parser import PrologParser
from problog.program import PrologFactory, ClauseDB, PrologString, PrologFile

# interpretation = { atom : True/False }



class SemiringLearning(SemiringProbability) :
    """Same as probabilistic semiring, but initializes t(...) weights with value."""
    
    
    def __init__(self, weights) :
        SemiringProbability.__init__(self)
        self.weights = weights
    
    def value(self, a) :
        if isinstance(a, Term) and a.functor == 'lfi' :
            assert(len(a.args) == 1)
            index = int(a.args[0])
            return self.weights[index]
        else :
            return float(a)
    
def str2bool(s) :
    if str(s) == 'true' :
        return True
    elif str(s) == 'false' :
        return False
    else :
        return None
                
class LFIProblem(SemiringProbability) :
    
    def __init__(self, source, examples, max_iter=10000, min_improv=1e-10) :
        SemiringProbability.__init__(self)
        self.source = source
        self.names = []
        self.queries = []
        self.weights = []
        self.examples = examples
        self._compiled_examples = None
        
        self.max_iter = max_iter
        self.min_improv = min_improv
    
    # Overwrite SemiringProbability
    def value(self, a) :
        if isinstance(a, Term) and a.functor == 'lfi' :
            assert(len(a.args) == 1)
            index = int(a.args[0])
            return self.weights[index]
        else :
            return float(a)
         
    @property 
    def count(self) :
        return len(self.weights)
    
    
    def prepare(self) :
        self._compile_examples()
        
    def _process_examples( self ) :
        """Process examples by grouping together examples with similar structure.
    
        :param examples: all examples, where examples are represented as sets of evidence (atom,value) pair.
        :type examples: sequence of lists of pairs
        :return: example groups based on evidence atoms
        :rtype: dict of atoms : values for examples
        """
    
        # value can be True / False / None
        # ( atom ), ( ( value, ... ), ... ) 

        # Simple implementation: don't add neutral evidence.
        result = defaultdict(list)
        for example in self.examples :
            atoms, values = zip(*sorted(example))
            result[atoms].append( values )
        return result
    
    def _compile_examples( self ) :
        """Compile examples.
    
        :param examples: Output of ::func::`process_examples`.
        """
        baseprogram = DefaultEngine().prepare(self)        
        examples = self._process_examples()
    
        result = []
        for atoms, example_group in examples.items() :
            ground_program = None   # Let the grounder decide
            for example in example_group :
                ground_program = ground( baseprogram, ground_program, evidence=zip( atoms, example ) )
                compiled_program = knowledge.createFrom(ground_program)
                result.append( (atoms, example, compiled_program) )
        self._compiled_examples = result
    
     
    def _process_atom( self, atom ) :
        """Returns tuple ( prob_atom, [ additional clauses ] )"""
        
        if atom.probability and atom.probability.functor == 't' :
            # Learnable probability
            assert(len(atom.probability.args) == 1) 
            start_value = atom.probability.args[0]

            # 1) Introduce a new fact
            lfi_fact = Term('lfi_fact_%d' % self.count, *atom.args)
            lfi_prob = Term('lfi', Constant(self.count)) 
            
            # 2) Replacement atom
            replacement = lfi_fact.withProbability(lfi_prob)
            
            # 3) Create redirection clause
            extra_clauses = [ Clause( atom.withProbability(), lfi_fact ) ]
            
            # 4) Set initial weight
            if isinstance(start_value, Constant) :
                self.weights.append( float(start_value) )
            else :
                self.weights.append( random.random() )
                
            # 5) Add query
            self.queries.append(lfi_fact)
            extra_clauses.append(Term('query', lfi_fact))
            
            # 6) Add name
            self.names.append(atom)
            
            return replacement, extra_clauses
        else :
            return atom, []
        
    # Overwrite from LogicProgram    
    def __iter__(self) :
        """Iterate over the clauses."""
        
        # TODO remove queries?
        
        for clause in self.source :
            if isinstance(clause, Clause) :
                new_head, extra_clauses = self._process_atom( clause.head )
                yield Clause( new_head, clause.body )
                for extra in extra_clauses : yield extra                
            elif isinstance(clause, AnnotatedDisjunction) :
                new_heads = []
                extra_clauses_all = []
                for head in clause.heads :
                    new_head, extra_clauses = self._process_atom( head )
                    new_heads.append(new_head)
                    extra_clauses_all += extra_clauses                
                yield AnnotatedDisjunction( new_heads, clause.body )
                for extra in extra_clauses_all : yield extra                
            else :
                # Fact
                new_fact, extra_clauses = self._process_atom( clause )
                yield new_fact
                for extra in extra_clauses : yield extra

                    
        # Queries are: lfi_fact_0 ... lfi_fact_<count-1>
        # Semiring lookup lfi_0 lfi_fact_0

    def evaluate_examples( self ) :
        results = []
        i = 0
        for at, val, comp in self._compiled_examples :        
            evidence = dict(zip(map(str,at),map(str2bool,val)))

            # with open('nnf_%s.dot' % (i), 'w') as f :
            #     print( comp.toDot(), file=f)
        
            evaluator = comp.getEvaluator(semiring=self, evidence=evidence) 

            pQueries = {}
            # Probability of query given evidence
            for name, node in evaluator.getNames(LABEL_QUERY) :
                w = evaluator.evaluate(node)    
                if w < 1e-6 : 
                    pQueries[name] = 0.0
                else :
                    pQueries[name] = w
            pEvidence = evaluator.evaluateEvidence()
            i+=1

        
            #pQueries = comp.evaluate(semiring=SemiringLearning(), weights=weights, evidence=evidence) 
            #pEvidence = comp.evaluateEvidence(semiring=SemiringLearning(), weights=weights, evidence=evidence)
            results.append( (pEvidence, pQueries) )
        return results
    
    def update(self, results) :
        fact_marg = [0.0] * self.count
        fact_count = [0] * self.count
        score = 0.0
        for pEvidence, result in results :
            for fact, value in result.items() :
                index = int(fact.split('(')[0].rsplit('_',1)[1])
                fact_marg[index] += value
                fact_count[index] += 1
            score += math.log(pEvidence)

        output = {}
        for index in range(0, self.count) :
            if fact_count[index] > 0 :
                self.weights[index] = fact_marg[index] / fact_count[index]
        return score
        
    def step(self) :
        results = self.evaluate_examples()
        return self.update(results)
        
    def run(self) :
        self.prepare()
        delta = 1000
        prev_score = -1e10
        iteration = 0
        while iteration < self.max_iter and delta > self.min_improv :
            score = self.step()
            iteration += 1
            delta = score - prev_score
            prev_score = score
        return prev_score
    
def test1() :
    
    program = """
        t(0.5)::burglary.
        0.2::earthquake.
        t(_)::p_alarm1.
        t(_)::p_alarm2.
        t(_)::p_alarm3.

        alarm :- burglary, earthquake, p_alarm1.
        alarm :- burglary, \+earthquake, p_alarm2.
        alarm :- \+burglary, earthquake, p_alarm3.
    """

    examples = [ 
        ( ( Term('burglary'), 'false' ), ( Term('alarm'), 'false' ) ),
        ( ( Term('earthquake'), 'false' ), ( Term('alarm'), 'true' ), ( Term('burglary'), 'true' ) ),
        ( ( Term('burglary'), 'false' ), )
    ]
    
    print (examples)

    lfi = LFIProblem(PrologString(program), examples)
    score = lfi.run()

    return run_lfi(PrologString(program), examples)

    
def test2() :
    
    program = """
       t(_)::stress(X) :- person(X).
       t(_)::influences(X,Y) :- person(X), person(Y).

       smokes(X) :- stress(X).
       smokes(X) :- friend(X,Y), influences(Y,X), smokes(Y).

       person(1).
       person(2).
       person(3).
       person(4).

       friend(1,2).
       friend(2,1).
       friend(2,4).
       friend(3,2).
       friend(4,2).
    """

    examples = [ 
        ( ( Term('smokes', Constant(2)), 'false' ), 
          ( Term('smokes', Constant(4)), 'true'  ),
          ( Term('influences', Constant(1), Constant(2)), 'false'  ),
          ( Term('influences', Constant(4), Constant(2)), 'false'  ),
          ( Term('influences', Constant(2), Constant(3)), 'true'  ),          
          ( Term('stress', Constant(1)), 'true'  )
        )
    ]

    return run_lfi(PrologString(program), examples)

def read_examples( *filenames ) :
    
    for filename in filenames :
        
        engine = DefaultEngine()
        
        with open(filename) as f :
            example = ''
            for line in f :
                if line.strip().startswith('---') :
                    pl = PrologString(example)
                    yield engine.query(pl, Term('evidence',None,None))
                    example = ''
                else :
                    example += line
            if example :
                pl = PrologString(example)
                yield engine.query(pl, Term('evidence',None,None))
    
    
def run_lfi( program, examples, max_iter=10000, min_improv=1e-10 ) :
    lfi = LFIProblem( program, examples, max_iter=max_iter, min_improv=min_improv )
    score = lfi.run()
    return score, lfi.weights, lfi.names
    
    
if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser("LFI learning with ProbLog")
    parser.add_argument('model')
    parser.add_argument('examples', nargs='+')
    parser.add_argument('-n', dest='max_iter', default=10000 )
    parser.add_argument('-d', dest='min_improv', default=1e-10 )
    args = parser.parse_args()
    
    program = PrologFile(args.model)
    examples = list( read_examples(*args.examples ) )
    score, weights, names = run_lfi( program, examples, args.max_iter, args.min_improv)
    
    print (score, weights, names)
    
    