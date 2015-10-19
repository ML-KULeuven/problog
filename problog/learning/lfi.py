#! /usr/bin/env python

"""
Learning from interpretations
-----------------------------

Parameter learning for ProbLog.

Given a probabilistic program with parameterized weights and a set of partial implementations,
learns appropriate values of the parameters.

Algorithm
+++++++++

The algorithm operates as follows:

    0. Set initial values for the weights to learn.
    1. Set the evidence present in the example.
    2. Query the model for the weights of the atoms to be learned.
    3. Update the weights to learn by taking the mean value over all examples and queries.
    4. Repeat steps 1 to 4 until convergence (or a maximum number of iterations).
    
The score of the model for a given example is obtained by calculating the probability of the
evidence in the example.

Implementation
++++++++++++++

The algorithm is implemented on top of the ProbLog toolbox.

It uses the following extensions of ProbLog's classes:

    * a LogicProgram implementation that rewrites the model and extracts the weights to learn
    (see :py:func:`learning.lfi.LFIProblem.__iter__`)
    * a custom semiring that looks up the current value of a weight to learn
    (see :py:func:`learning.lfi.LFIProblem.value`)


.. autoclass:: learning.lfi.LFIProblem
    :members: __iter__, value

"""

from __future__ import print_function

import sys
import os
import random
import math
import logging

from collections import defaultdict

# Make sure the ProbLog module is on the path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from problog.engine import DefaultEngine, ground
from problog.evaluator import SemiringProbability
from problog.logic import Term, Constant, Clause, AnnotatedDisjunction, Or
from problog.program import PrologString, PrologFile, LogicProgram
from problog.core import ProbLogError
from problog.errors import process_error
from problog.sdd_formula import SDD

from problog import get_evaluatable, get_evaluatables
import traceback


def str2bool(s):
    if str(s) == 'true':
        return True
    elif str(s) == 'false':
        return False
    else:
        return None


class LFIProblem(SemiringProbability, LogicProgram) :
    
    def __init__(self, source, examples, max_iter=10000, min_improv=1e-10, verbose=0, knowledge=SDD, **extra):
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
        self.verbose = verbose
        self.iteration = 0

        self.knowledge = knowledge

        self.output_mode = False
    
    def value(self, a):
        """Overrides from SemiringProbability.
        Replaces weights of the form ``lfi(i)`` by their current estimated value.
        """
        
        if isinstance(a, Term) and a.functor == 'lfi':
            assert(len(a.args) == 1)
            index = int(a.args[0])
            return self.weights[index]
        else :
            return float(a)
         
    @property 
    def count(self):
        """Number of parameters to learn."""
        return len(self.weights)
    
    def prepare(self):
        """Prepare for learning."""
        self._compile_examples()
        
    def _process_examples(self):
        """Process examples by grouping together examples with similar structure.
    
        :return: example groups based on evidence atoms
        :rtype: dict of atoms : values for examples
        """
    
        # value can be True / False / None
        # ( atom ), ( ( value, ... ), ... ) 

        # Simple implementation: don't add neutral evidence.
        result = defaultdict(list)
        for example in self.examples:
            atoms, values = zip(*example)
            result[atoms].append(values)
        return result
    
    def _compile_examples( self ) :
        """Compile examples.
    
        :param examples: Output of ::func::`process_examples`.
        """
        logger = logging.getLogger('problog_lfi')

        baseprogram = DefaultEngine().prepare(self)
        examples = self._process_examples()

        result = []
        n = 0
        for atoms, example_group in examples.items():
            ground_program = None   # Let the grounder decide
            for example in example_group :
                if self.verbose:
                    n += 1
                    logger.debug('Compiling example %s ...' % n)

                ground_program = ground(baseprogram, ground_program,
                                        evidence=list(zip(atoms, example)))
                compiled_program = self.knowledge.create_from(ground_program)
                result.append((atoms, example, compiled_program))
        self._compiled_examples = result

    def _process_atom(self, atom, body):
        """Returns tuple ( prob_atom, [ additional clauses ] )"""
        if isinstance(atom, Or):
            # Annotated disjunction
            atoms = atom.to_list()
        else:
            atoms = [atom]

        atoms_out = []
        extra_clauses = []

        has_lfi_fact = False
        available_probability = 1.0

        num_random_weights = 0
        for atom in atoms:
            if atom.probability and atom.probability.functor == 't':
                start_value = atom.probability.args[0]
                if isinstance(start_value, Constant):
                    available_probability -= float(start_value)
                else:
                    num_random_weights += 1
            elif atom.probability and atom.is_constant():
                available_probability -= float(atom.probability)

        random_weights = [random.random() for i in range(0, num_random_weights+1)]
        norm_factor = available_probability / sum(random_weights)
        random_weights = [r*norm_factor for r in random_weights]

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
                replacement = lfi_fact.with_probability(lfi_prob)
                if body is None:
                    new_body = lfi_fact
                else:
                    new_body = body & lfi_fact

                # 3) Create redirection clause
                extra_clauses += [Clause(atom.with_probability(), new_body)]

                # 4) Set initial weight
                if isinstance(start_value, Constant):
                    self.weights.append(float(start_value))
                else:
                    self.weights.append(random_weights.pop(-1))

                # 5) Add query
                self.queries.append(lfi_fact)
                if body:
                    extra_clauses.append(Clause(Term('query', lfi_fact), body))
                else:
                    extra_clauses.append(Term('query', lfi_fact))

                # 6) Add name
                self.names.append(atom)

                atoms_out.append(replacement)
            else:
                atoms_out.append(atom)

        if has_lfi_fact:
            if len(atoms) == 1:     # Simple clause
                return [atoms_out[0]] + extra_clauses
            else:
                return [AnnotatedDisjunction(atoms_out, Term('true'))] + extra_clauses
        else:
            if len(atoms) == 1:
                if body is None:
                    return [atoms_out[0]]
                else:
                    return [Clause(atoms_out[0], body)]
            else:
                if body is None:
                    body = Term('true')
                return [AnnotatedDisjunction(atoms_out, body)]

    def _process_atom_output(self, atom, body):
        """Returns tuple ( prob_atom, [ additional clauses ] )"""

        if isinstance(atom, Or):
            atoms = atom.to_list()
        else:
            atoms = [atom]

        atoms_out = []
        for atom in atoms:
            if atom.probability and atom.probability.functor == 't':
                assert (atom in self.names)
                index = self.output_names.index(atom)
                self.output_names[index] = None
                weight = self.weights[index]
                result = atom.with_probability(weight)
                atoms_out.append(result)
            else:
                atoms_out.append(atom)
        if len(atoms_out) == 1:
            if body is None:
                return [atoms_out[0]]
            else:
                return [Clause(atoms_out[0], body)]
        else:
            return [AnnotatedDisjunction(atoms_out, body)]
        
    # Overwrite from LogicProgram    
    def __iter__(self) :
        """
        Iterate over the clauses of the source model.
        This object can be used as a LogicProgram to be passed to the grounding Engine.
        
        Extracts and processes all ``t(...)`` weights.
        This
        
            * replaces each probabilistic atom ``t(...)::p(X)`` by a unique atom \
            ``lfi(i) :: lfi_fact_i(X)``;
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
            if isinstance(clause, Clause):
                if clause.head.functor == 'query' and clause.head.arity == 1:
                    continue                
                extra_clauses = process_atom(clause.head, clause.body)
                for extra in extra_clauses:
                    yield extra
            elif isinstance(clause, AnnotatedDisjunction):
                extra_clauses = process_atom(Or.from_list(clause.heads), clause.body)
                for extra in extra_clauses:
                    yield extra
            else:
                if clause.functor == 'query' and clause.arity == 1:
                    continue
                # Fact
                extra_clauses = process_atom(clause, None)
                for extra in extra_clauses:
                    yield extra

    def _evaluate_examples( self ) :
        """Evaluate the model with its current estimates for all examples."""
        
        results = []
        i = 0
        logging.getLogger('problog_lfi').debug('Evaluating examples ...')
        for at, val, comp in self._compiled_examples:
            evidence = dict(zip(at, map(str2bool, val)))
            evaluator = comp.get_evaluator(semiring=self, evidence=evidence)
            p_queries = {}
            # Probability of query given evidence
            for name, node in evaluator.formula.queries():
                w = evaluator.evaluate_fact(node)
                if w < 1e-6:
                    p_queries[name] = 0.0
                else:
                    p_queries[name] = w
            p_evidence = evaluator.evaluate_evidence()
            i += 1
            results.append((p_evidence, p_queries))
        return results
    
    def _update(self, results) :
        """Update the current estimates based on the latest evaluation results."""
        
        fact_marg = [0.0] * self.count
        fact_count = [0] * self.count
        score = 0.0
        for pEvidence, result in results:
            for fact, value in result.items():
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
        logging.getLogger('problog_lfi').info('Weights to learn: %s' % self.names)
        logging.getLogger('problog_lfi').info('Initial weights: %s' % self.weights)
        delta = 1000
        prev_score = -1e10
        while self.iteration < self.max_iter and (delta < 0 or delta > self.min_improv):
            score = self.step()
            logging.getLogger('problog_lfi').info('Weights after iteration %s: %s' % (self.iteration, self.weights))
            delta = score - prev_score
            prev_score = score
        return prev_score

def extract_evidence(pl):
    engine = DefaultEngine()
    atoms = engine.query(pl, Term('evidence', None, None))
    atoms1 = engine.query(pl, Term('evidence', None))
    for atom in atoms1:
        atom = atom[0]
        if atom.is_negated():
            atoms.append((-atom, Term('false')))
        else:
            atoms.append((atom, Term('true')))
    return atoms

def read_examples(*filenames):
    
    for filename in filenames:
        engine = DefaultEngine()
        
        with open(filename) as f:
            example = ''
            for line in f:
                if line.strip().startswith('---') :
                    pl = PrologString(example)
                    atoms = extract_evidence(pl)
                    if len(atoms) > 0:
                        yield atoms
                    example = ''
                else:
                    example += line
            if example:
                pl = PrologString(example)
                atoms = extract_evidence(pl)
                if len(atoms) > 0:
                    yield atoms
    
    
def run_lfi( program, examples, output_model=None, **kwdargs):
    lfi = LFIProblem(program, examples, **kwdargs)
    score = lfi.run()

    if output_model is not None:
        with open(output_model, 'w') as f:
            f.write(lfi.get_model())
    return score, lfi.weights, lfi.names, lfi.iteration


def argparser():
    import argparse
    parser = argparse.ArgumentParser(description="Learning from interpretations with ProbLog")
    parser.add_argument('model')
    parser.add_argument('examples', nargs='+')
    parser.add_argument('-n', dest='max_iter', default=10000, type=int )
    parser.add_argument('-d', dest='min_improv', default=1e-10, type=float )
    parser.add_argument('-O', '--output-model', type=str, default=None,
                        help='write resulting model to given file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='write output to file')
    parser.add_argument('-k', '--knowledge', dest='koption', choices=get_evaluatables(),
                        default=None, help='knowledge compilation tool')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--web', action='store_true', help=argparse.SUPPRESS)
    return parser


def create_logger(name, verbose):
    levels = [logging.WARNING, logging.INFO, logging.DEBUG] + list(range(9, 0, -1))
    verbose = max(0, min(len(levels)-1, verbose))
    logger = logging.getLogger(name)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(levels[verbose])


def main(argv, result_handler=None):
    parser = argparser()
    args = parser.parse_args(argv)

    if result_handler is None:
        if args.web:
            result_handler = print_result_json
        else:
            result_handler = print_result

    knowledge = get_evaluatable(args.koption)

    if args.output is None:
        outf = sys.stdout
    else:
        outf = open(args.output, 'w')

    create_logger('problog_lfi', args.verbose)
    create_logger('problog', args.verbose - 1)

    program = PrologFile(args.model)
    examples = list(read_examples(*args.examples))
    if len(examples) == 0:
        logging.getLogger('problog_lfi').warn('no examples specified')
    else:
        logging.getLogger('problog_lfi').info('Number of examples: %s' % len(examples))
    options = vars(args)
    del options['examples']

    try:
        results = run_lfi(program, examples, knowledge=knowledge, **options)

        for n in results[2]:
            n.loc = program.lineno(n.location)
        retcode = result_handler((True, results), output=outf)
    except Exception as err:
        trace = traceback.format_exc()
        err.trace = trace
        retcode = result_handler((False, err), output=outf)

    if args.output is not None:
        outf.close()

    if retcode:
        sys.exit(retcode)


def print_result(d, output, precision=8):
    success, d = d
    if success:
        score, weights, names, iterations = d
        weights = list(map(lambda x: round(x, precision), weights))
        print (score, weights, names, iterations, file=output)
        return 0
    else:
        print (process_error(d), file=output)
        return 1


def print_result_json(d, output, precision=8):
    import json
    success, d = d
    if success:
        score, weights, names, iterations = d
        results = {'SUCCESS': True,
                   'score': score,
                   'iterations': iterations,
                   'weights': [[str(n.with_probability()), round(w, precision), n.loc[1], n.loc[2]]
                               for n, w in zip(names, weights)],
                   }
        print (json.dumps(results), file=output)
    else:
        results = {'SUCCESS': False, 'err': d}
        print (json.dumps(results), file=output)
    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
