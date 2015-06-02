#! /usr/bin/env python

"""
Learning from interpretations
-----------------------------

Parameter learning for ProbLog.

Given a probabilistic program with parameterized weights and a set of partial implementations, learns appropriate values of the parameters.

Algorithm
+++++++++

The algorithm operates as follows:

    0. Set initial values for the weights to learn.
    1. Set the evidence present in the example.
    2. Query the model for the weights of the atoms to be learned.
    3. Update the weights to learn by taking the mean value over all examples and queries.
    4. Repeat steps 1 to 4 until convergence (or a maximum number of iterations).
    
The score of the model for a given example is obtained by calculating the probability of the evidence in the example.

Implementation
++++++++++++++

The algorithm is implemented on top of the ProbLog toolbox.

It uses the following extensions of ProbLog's classes:

    * a LogicProgram implementation that rewrites the model and extracts the weights to learn (see :py:func:`learning.lfi.LFIProblem.__iter__`)
    * a custom semiring that looks up the current value of a weight to learn (see :py:func:`learning.lfi.LFIProblem.value`)


.. autoclass:: learning.lfi.LFIProblem
    :members: __iter__, value


"""


from __future__ import print_function

import sys, os, random, math
from collections import defaultdict

sys.path.append(os.path.abspath( os.path.join( os.path.dirname(__file__), '../' ) ) )

from problog.engine import DefaultEngine, ground
# from problog.nnf_formula import NNF as knowledge
from problog.sdd_formula import SDD as knowledge
from problog.evaluator import SemiringProbability
from problog.logic import Term, Var, Constant, Clause, AnnotatedDisjunction, LogicProgram, Or
from problog.parser import PrologParser
from problog.program import PrologFactory, PrologString, PrologFile
from problog.core import ProbLogError

    
def str2bool(s) :
    if str(s) == 'true' :
        return True
    elif str(s) == 'false' :
        return False
    else :
        return None
                
class LFIProblem(SemiringProbability, LogicProgram) :
    
    def __init__(self, source, examples, max_iter=10000, min_improv=1e-10) :
        SemiringProbability.__init__(self)
        LogicProgram.__init__(self)
        self.source = source
        self.names = []
        self.queries = []
        self.weights = []
        self.examples = examples
        self._compiled_examples = None
        
        self.max_iter = max_iter
        self.min_improv = min_improv
        self.iteration = 0

        self.output_mode = False
    
    def value(self, a) :
        """Overrides from SemiringProbability.
        Replaces weights of the form ``lfi(i)`` by their current estimated value.
        """
        
        if isinstance(a, Term) and a.functor == 'lfi' :
            assert(len(a.args) == 1)
            index = int(a.args[0])
            return self.weights[index]
        else :
            return float(a)
         
    @property 
    def count(self) :
        """Number of parameters to learn."""
        return len(self.weights)
    
    def prepare(self) :
        """Prepare for learning."""
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
            atoms, values = zip(*example)
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
    
     
    def _process_atom(self, atom, body):
        """Returns tuple ( prob_atom, [ additional clauses ] )"""
        if isinstance(atom, Or):
            # Annotated disjunction
            atoms = atom.toList()
        else:
            atoms = [atom]

        atoms_out = []
        extra_clauses = []

        has_lfi_fact = False
        for atom in atoms:
            if atom.probability and atom.probability.functor == 't':
                # t(_)::p(X) :- body.
                #
                # Translate to
                #   lfi(1)::lfi_fact_1(X).
                #   p(X) :- body, lfi_fact1(X).
                # For annotated disjunction: t(_)::p1(X); t(_)::p2(X) :- body.
                #   lfi1::lfi_fact1(X); lfi2::lfi_fact2(X); ... .
                #   p1(X) :- body, lfi_fact1(X).
                #   p2(X) :- body, lfi_fact2(X).
                #  ....
                has_lfi_fact = True

                # Learnable probability
                assert(len(atom.probability.args) == 1)
                start_value = atom.probability.args[0]

                # 1) Introduce a new fact
                lfi_fact = Term('lfi_fact_%d' % self.count, *atom.args)
                lfi_prob = Term('lfi', Constant(self.count))

                # 2) Replacement atom
                replacement = lfi_fact.withProbability(lfi_prob)
                if body is None:
                    new_body = lfi_fact
                else:
                    new_body = body & lfi_fact

                # 3) Create redirection clause
                extra_clauses += [Clause(atom.withProbability(), new_body)]

                # 4) Set initial weight
                if isinstance(start_value, Constant):
                    self.weights.append(float(start_value))
                else:
                    self.weights.append(random.random())

                # 5) Add query
                self.queries.append(lfi_fact)
                extra_clauses.append(Term('query', lfi_fact))

                # 6) Add name
                self.names.append(atom)

                atoms_out.append(replacement)
            else:
                atoms_out.append(atom)

        if has_lfi_fact:
            if len(atoms) == 1:     # Simple clause
                #if body is None:
                return [atoms_out[0]] + extra_clauses
                #else:
                #    return [Clause(atoms_out[0], body)] + extra_clauses
            else:
                return [AnnotatedDisjunction(Or.fromList(atoms_out), Term('true'))] + extra_clauses
        else:
            if len(atoms) == 1:
                if body is None:
                    return [atoms_out[0]]
                else:
                    return [Clause(atoms_out[0], body)]
            else:
                return [AnnotatedDisjunction(atoms_out, body)]

    def _process_atom_output(self, atom, body):
        """Returns tuple ( prob_atom, [ additional clauses ] )"""

        if isinstance(atom, Or):
            atoms = atom.toList()
        else:
            atoms = [atom]

        atoms_out = []
        for atom in atoms:
            if atom.probability and atom.probability.functor == 't' :
                assert (atom in self.names)
                index = self.output_names.index(atom)
                self.output_names[index] = None
                weight = self.weights[index]
                result = atom.withProbability(weight)
                atoms_out.append(result)
            else:
                atoms_out.append(atom)
        if len(atoms_out) == 1:
            if body is None:
                return [atoms_out[0]]
            else:
                return [Clause(atoms_out[0], body)]
        else:
            return [AnnotatedDisjunction(Or.fromList(atoms_out), body)]
        
    # Overwrite from LogicProgram    
    def __iter__(self) :
        """
        Iterate over the clauses of the source model.
        This object can be used as a LogicProgram to be passed to the grounding Engine.
        
        Extracts and processes all ``t(...)`` weights.
        This
        
            * replaces each probabilistic atom ``t(...)::p(X)`` by a unique atom ``lfi(i) :: lfi_fact_i(X)``;
            * adds a new clause ``p(X) :- lfi_fact_i(X)``;
            * adds a new query ``query( lfi_fact_i(X) )``;
            * initializes the weight of ``lfi(i)`` based on the ``t(...)`` specification;
        
        This also removes all existing queries from the model.
        
        Example:
        
        .. code-block:: prolog
        
            t(_) :: p(X) :- b(X).
            t(_) :: p(X) :- c(X).

        is transformed into
        
        .. code-block:: prolog
        
            lfi(0) :: lfi_fact_0(X) :- b(X).
            p(X) :- lfi_fact_0(X).
            lfi(1) :: lfi_fact_1(X) :- c(X).
            p(X) :- lfi_fact_1(X).
            query(lfi_fact_0(X)).
            query(lfi_fact_1(X)).
        
        """

        if self.output_mode:
            process_atom = self._process_atom_output
            self.output_names = self.names[:]
        else:
            process_atom = self._process_atom

        for clause in self.source:
            if isinstance(clause, Clause) or isinstance(clause, AnnotatedDisjunction):
                if clause.head.functor == 'query' and clause.head.arity == 1:
                    continue                
                extra_clauses = process_atom(clause.head, clause.body)
                #yield Clause(new_head, clause.body)
                for extra in extra_clauses:
                    yield extra
            # elif isinstance(clause, AnnotatedDisjunction):
            #     new_heads = []
            #     extra_clauses_all = []
            #     for head in clause.heads:
            #         new_head, extra_clauses = process_atom(head, clause.body)
            #         new_heads.append(new_head)
            #         extra_clauses_all += extra_clauses
            #     yield AnnotatedDisjunction(new_heads, clause.body)
            #     for extra in extra_clauses_all:
            #         yield extra
            else:
                if clause.functor == 'query' and clause.arity == 1:
                    continue
                # Fact
                extra_clauses = process_atom(clause, None)
                # yield new_fact
                for extra in extra_clauses:
                    yield extra

    def _evaluate_examples( self ) :
        """Evaluate the model with its current estimates for all examples."""
        
        results = []
        i = 0
        for at, val, comp in self._compiled_examples :        
            evidence = dict(zip(at,map(str2bool,val)))

            evaluator = comp.getEvaluator(semiring=self, evidence=evidence) 

            pQueries = {}
            # Probability of query given evidence
            for name, node in evaluator.getNames('query') :
                w = evaluator.evaluate(node)    
                if w < 1e-6 : 
                    pQueries[name] = 0.0
                else :
                    pQueries[name] = w
            pEvidence = evaluator.evaluateEvidence()
            i+=1
            results.append( (pEvidence, pQueries) )
        return results
    
    def _update(self, results) :
        """Update the current estimates based on the latest evaluation results."""
        
        fact_marg = [0.0] * self.count
        fact_count = [0] * self.count
        score = 0.0
        for pEvidence, result in results :
            for fact, value in result.items() :
                fact = str(fact)
                index = int(fact.split('(')[0].rsplit('_',1)[1])
                fact_marg[index] += value
                fact_count[index] += 1
            try:
                score += math.log(pEvidence)
            except ValueError:
                raise ProbLogError('Inconsistent evidence.')

        output = {}
        for index in range(0, self.count) :
            if fact_count[index] > 0 :
                self.weights[index] = fact_marg[index] / fact_count[index]
        return score
        
    def step(self) :
        self.iteration += 1
        results = self._evaluate_examples()
        return self._update(results)

    def get_model(self):
        self.output_mode = True
        lines = []
        for l in self:
            lines.append('%s.' % l)
        lines.append('')
        self.output_mode = False
        return '\n'.join(lines)
        
    def run(self) :
        self.prepare()
        delta = 1000
        prev_score = -1e10
        while self.iteration < self.max_iter and delta > self.min_improv :
            score = self.step()
            delta = score - prev_score
            prev_score = score
        return prev_score

def read_examples( *filenames ) :
    
    for filename in filenames :
        
        engine = DefaultEngine()
        
        with open(filename) as f :
            example = ''
            for line in f :
                if line.strip().startswith('---') :
                    pl = PrologString(example)
                    atoms =  engine.query(pl, Term('evidence',None,None))
                    if len(atoms) > 0:
                        yield atoms
                    example = ''
                else :
                    example += line
            if example :
                pl = PrologString(example)
                atoms = engine.query(pl, Term('evidence',None,None))
                if len(atoms) > 0:
                    yield atoms
    
    
def run_lfi( program, examples, max_iter=10000, min_improv=1e-10, output=None):
    lfi = LFIProblem( program, examples, max_iter=max_iter, min_improv=min_improv )
    score = lfi.run()

    if output is not None:
        with open(output,'w') as f:
            f.write(lfi.get_model())
    return score, lfi.weights, lfi.names, lfi.iteration

    
if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser(description="Learning from interpretations with ProbLog")
    parser.add_argument('model')
    parser.add_argument('examples', nargs='+')
    parser.add_argument('-n', dest='max_iter', default=10000 )
    parser.add_argument('-d', dest='min_improv', default=1e-10 )
    parser.add_argument('-o', '--output', type=str, default=None, help='write resulting model to given file')
    args = parser.parse_args()
    
    program = PrologFile(args.model)
    examples = list(read_examples(*args.examples))
    score, weights, names, iterations = run_lfi( program, examples, args.max_iter, args.min_improv, args.output)
    
    print (score, weights, names, iterations)
    
    