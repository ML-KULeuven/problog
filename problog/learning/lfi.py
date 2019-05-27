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
import random
import math
import logging

from collections import defaultdict

from problog.engine import DefaultEngine, ground
from problog.evaluator import SemiringProbability
from problog.logic import Term, Constant, Clause, AnnotatedDisjunction, Or, Var,\
    InstantiationError, ArithmeticError
from problog.program import PrologString, PrologFile, LogicProgram
from problog.core import ProbLogError
from problog.errors import process_error, InconsistentEvidenceError


from problog import get_evaluatable, get_evaluatables
import traceback


def str2bool(s):
    if str(s) == 'true':
        return True
    elif str(s) == 'false':
        return False
    else:
        return None


class LFIProblem(SemiringProbability, LogicProgram):

    def __init__(self, source, examples, max_iter=10000, min_improv=1e-10, verbose=0, knowledge=None,
                 leakprob=None, propagate_evidence=True, normalize=False, **extra):
        """
        :param source: filename of file containing input model
        :type source: str
        :param examples: list of observed terms / value
        :type examples: list[tuple(Term, bool)]
        :param max_iter: maximum number of iterations to run
        :type max_iter: int
        :param min_improv: minimum improvement in log-likelihood for convergence detection
        :type min_improv: float
        :param verbose: verbosity level
        :type verbose: int
        :param knowledge: class to use for knowledge compilation
        :type knowledge: class
        :param leakprob: Add all true evidence atoms with the given probability
                         to avoid 'inconsistent evidence' errors. This also
                         allows to learn a program without constants and
                         retrieve the constants from the evidence file.
                         (default: None)
        :type leakprob: float or None
        :param extra: catch all for additional parameters (not used)
        """
        SemiringProbability.__init__(self)
        LogicProgram.__init__(self)
        self.source = source

        # The names of the atom for which we want to learn weights.
        self.names = []

        # The weights to learn.
        # The initial weights are of type 'float'.
        # When necessary they are replaced by a dictionary [t(arg1, arg2, ...) -> float]
        #  for weights of form t(SV, arg1, arg2, ...).
        self._weights = []

        self.examples = examples
        self.leakprob = leakprob
        self.leakprobatoms = None
        self.propagate_evidence = propagate_evidence
        self._compiled_examples = None
        
        self.max_iter = max_iter
        self.min_improv = min_improv
        self.verbose = verbose
        self.iteration = 0

        if knowledge is None:
            knowledge = get_evaluatable()
        self.knowledge = knowledge

        self.output_mode = False
        self.extra = extra

        self._enable_normalize = normalize
        self._adatoms = []
    
    def value(self, a):
        """Overrides from SemiringProbability.
        Replaces a weight of the form ``lfi(i, t(...))`` by its current estimated value.
        Other weights are passed through unchanged.

        :param a: term representing the weight
        :type a: Term
        :return: current weight
        :rtype: float
        """
        if isinstance(a, Term) and a.functor == 'lfi':
            # index = int(a.args[0])
            return self._get_weight(*a.args)
        else:
            return float(a)
         
    @property 
    def count(self):
        """Number of parameters to learn."""
        return len(self.names)
    
    def prepare(self):
        """Prepare for learning."""
        self._compile_examples()

    def _get_weight(self, index, args, strict=True):
        index = int(index)
        weight = self._weights[index]
        if isinstance(weight, dict):
            if strict:
                return weight[args]
            else:
                return weight.get(args, 0.0)
        else:
            return weight

    def get_weights(self, index):
        """Get a list of key, weight pairs for the given input fact.

        :param index: identifier of the fact
        :return: list of key, weight pairs where key refers to the additional variables
        on which the weight is based
        :rtype: list[tuple[Term, float]]
        """
        weight = self._weights[index]
        if isinstance(weight, dict):
            return list(weight.items())
        else:
            return [(Term('t'), weight)]

    def _set_weight(self, index, args, weight):
        index = int(index)
        if not args:
            assert not isinstance(self._weights[index], dict)
            self._weights[index] = weight
        elif isinstance(self._weights[index], dict):
            self._weights[index][args] = weight
        else:
            self._weights[index] = {args: weight}

    def _add_weight(self, weight):
        self._weights.append(weight)

    def _process_examples(self):
        """Process examples by grouping together examples with similar structure.
    
        :return: example groups based on evidence atoms
        :rtype: dict of atoms : values for examples
        """
    
        # value can be True / False / None
        # ( atom ), ( ( value, ... ), ... ) 

        # Simple implementation: don't add neutral evidence.

        if self.propagate_evidence:
            result = ExampleSet()
            for index, example in enumerate(self.examples):
                atoms, values = zip(*example)
                result.add(index, atoms, values)
            return result
        else:
            # smarter: compile-once all examples with same atoms
            result = ExampleSet()
            for index, example in enumerate(self.examples):
                atoms, values = zip(*example)
                result.add(index, atoms, values)
            return result
    
    def _compile_examples(self):
        """Compile examples.
    
        :param examples: Output of ::func::`process_examples`.
        """
        logger = logging.getLogger('problog_lfi')

        baseprogram = DefaultEngine(**self.extra).prepare(self)
        examples = self._process_examples()
        result = []
        for example in examples:
            example.compile(self, baseprogram)
        self._compiled_examples = examples

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
                try:
                    start_value = float(atom.probability.args[0])
                    available_probability -= float(start_value)
                except InstantiationError:
                    # Can't be converted to float => take random
                    num_random_weights += 1
                except ArithmeticError:
                    num_random_weights += 1
            elif atom.probability and atom.probability.is_constant():
                available_probability -= float(atom.probability)

        random_weights = [random.random() for i in range(0, num_random_weights + 1)]
        norm_factor = available_probability / sum(random_weights)
        random_weights = [r * norm_factor for r in random_weights]

        self._adatoms.append((available_probability, []))

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
                try:
                    start_value = float(atom.probability.args[0])
                except InstantiationError:
                    start_value = None
                except ArithmeticError:
                    start_value = None

                # Replace anonymous variables with non-anonymous variables.
                class ReplaceAnon(object):

                    def __init__(self):
                        self.cnt = 0

                    def __getitem__(self, key):
                        if key == '_':
                            self.cnt += 1
                            return Var('anon_%s' % self.cnt)
                        else:
                            return Var(key)

                atom1 = atom.apply(ReplaceAnon())
                prob_args = atom.probability.args[1:]

                # 1) Introduce a new fact
                lfi_fact = Term('lfi_fact', Constant(self.count), Term('t', *prob_args), *atom1.args)
                lfi_prob = Term('lfi', Constant(self.count), Term('t', *prob_args))

                # 2) Replacement atom
                replacement = lfi_fact.with_probability(lfi_prob)
                if body is None:
                    new_body = lfi_fact
                else:
                    new_body = body & lfi_fact

                # 3) Create redirection clause
                extra_clauses += [Clause(atom1.with_probability(), new_body)]

                self._adatoms[-1][1].append(len(self._weights))
                # 4) Set initial weight
                if start_value is None:
                    self._add_weight(random_weights.pop(-1))
                else:
                    self._add_weight(start_value)

                # 5) Add name
                self.names.append(atom)
                atoms_out.append(replacement)
            else:
                atoms_out.append(atom)

        if len(self._adatoms[-1][1]) < 2:
            self._adatoms.pop(-1)

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

        transforms = defaultdict(list)

        clauses = []
        atoms_fixed = []
        t_args = None
        fixed_only = True
        for atom in atoms:
            if atom.probability and atom.probability.functor == 't':
                assert (atom in self.names)
                # assert (t_args is None or atom.probability.args == t_args)
                # t_args = atom.probability.args

                index = self.output_names.index(atom)
                weights = self.get_weights(index)

                for w_args, w_val in weights:
                    translate = tuple(zip(atom.probability.args[1:], w_args.args))
                    transforms[translate].append(atom.with_probability(Constant(w_val)))
                self.output_names[index] = None
                fixed_only = False
            else:
                atoms_fixed.append(atom)

        if not fixed_only:
            clauses = []
            for tr, atoms in transforms.items():
                tr = DefaultDict({k: v for k, v in tr})
                atoms_out = [at.apply(tr) for at in atoms] + atoms_fixed
                if len(atoms_out) == 1:
                    if body is None:
                        clauses.append(atoms_out[0])
                    else:
                        clauses.append(Clause(atoms_out[0], body.apply(tr)))
                else:
                    if body is None:
                        clauses.append(AnnotatedDisjunction(atoms_out, None))
                    else:
                        clauses.append(AnnotatedDisjunction(atoms_out, body.apply(tr)))
            return clauses
        else:
            atoms_out = atoms_fixed
            if len(atoms_out) == 1:
                if body is None:
                    return [atoms_out[0]]
                else:
                    return [Clause(atoms_out[0], body)]
            else:
                return [AnnotatedDisjunction(atoms_out, body)]

    # Overwrite from LogicProgram    
    def __iter__(self):
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

        If ``self.leakprobs`` is a value, then during learning all true
        examples are added to the program with the given leak probability.

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

        if self.leakprob is not None:
            leakprob_atoms = self._get_leakprobatoms()
            for example_atom in leakprob_atoms:
                yield example_atom.with_probability(Constant(self.leakprob))

    def _get_leakprobatoms(self):
        if self.leakprobatoms is not None:
            return self.leakprobatoms
        self.leakprobatoms = set()
        for examples in self.examples:
            for example, obs in examples:
                if obs:
                    self.leakprobatoms.add(example)
        return self.leakprobatoms

    def _evaluate_examples(self):
        """Evaluate the model with its current estimates for all examples."""
        
        results = []
        i = 0
        logging.getLogger('problog_lfi').debug('Evaluating examples ...')

        evaluator = ExampleEvaluator(self._weights)

        return map(evaluator, self._compiled_examples)
    
    def _update(self, results):
        """Update the current estimates based on the latest evaluation results."""
        
        fact_marg = defaultdict(float)
        fact_count = defaultdict(int)
        score = 0.0
        for m, pEvidence, result in results:
            for fact, value in result.items():
                index = fact.args[0:2]
                fact_marg[index] += value * m
                fact_count[index] += m
            try:
                score += math.log(pEvidence)
            except ValueError:
                raise ProbLogError('Inconsistent evidence.')

        for index in fact_marg:
            if fact_count[index] > 0:
                self._set_weight(index[0], index[1], fact_marg[index] / fact_count[index])

        if self._enable_normalize:
            self._normalize_weights()
        return score

    def _normalize_weights(self):

        for p, idx in self._adatoms:
            keys = set()
            for i in idx:
                for key, val in self.get_weights(i):
                    keys.add(key)
            if len(keys) > 1:
                try:
                    keys.remove(Term('t'))
                except KeyError:
                    pass
            for key in keys:
                w = sum(self._get_weight(i, key, strict=False) for i in idx)
                n = p / w
                for i in idx:
                    self._set_weight(i, key, self._get_weight(i, key, strict=False) * n)
        
    def step(self):
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
        
    def run(self):
        self.prepare()
        logging.getLogger('problog_lfi').info('Weights to learn: %s' % self.names)
        logging.getLogger('problog_lfi').info('Initial weights: %s' % self._weights)
        delta = 1000
        prev_score = -1e10
        while self.iteration < self.max_iter and (delta < 0 or delta > self.min_improv):
            score = self.step()
            logging.getLogger('problog_lfi').info('Weights after iteration %s: %s' % (self.iteration, self._weights))
            logging.getLogger('problog_lfi').info('Score after iteration %s: %s' % (self.iteration, score))
            delta = score - prev_score
            prev_score = score
        return prev_score


class ExampleSet(object):

    def __init__(self):
        self._examples = {}

    def add(self, index, atoms, values):
        ex = self._examples.get((atoms, values))
        if ex is None:
            self._examples[(atoms,values)] = Example(index, atoms, values)
        else:
            ex.add_index(index)

    def __iter__(self):
        return iter(self._examples.values())


class Example(object):

    def __init__(self, index, atoms, values):
        """An example consists of a list of atoms and their corresponding values (True/False)."""
        self.atoms = tuple(atoms)
        self.values = tuple(values)
        self.compiled = []
        self.n = [index]

    def __hash__(self):
        return hash((self.atoms, self.values))

    def __eq__(self, other):
        if other is None:
            return False
        return self.atoms == other.atoms and self.values == other.values

    def compile(self, lfi, baseprogram):
        ground_program = None  # Let the grounder decide
        ground_program = ground(baseprogram, ground_program,
                                evidence=list(zip(self.atoms, self.values)),
                                propagate_evidence=lfi.propagate_evidence)
        for i, node, t in ground_program:
            if t == 'atom' and isinstance(node.probability, Term) and node.probability.functor == 'lfi':
                factargs = ()
                if type(node.identifier) == tuple:
                    factargs = node.identifier[1]
                fact = Term('lfi_fact', node.probability.args[0], node.probability.args[1], *factargs)
                ground_program.add_query(fact, i)
        self.compiled = lfi.knowledge.create_from(ground_program)

    def add_index(self, index):
        self.n.append(index)


class ExampleEvaluator(SemiringProbability):

    def __init__(self, weights):
        SemiringProbability.__init__(self)
        self._weights = weights

    def _get_weight(self, index, args, strict=True):
        index = int(index)
        weight = self._weights[index]
        if isinstance(weight, dict):
            if strict:
                return weight[args]
            else:
                return weight.get(args, 0.0)
        else:
            return weight


    def value(self, a):
        """Overrides from SemiringProbability.
        Replaces a weight of the form ``lfi(i, t(...))`` by its current estimated value.
        Other weights are passed through unchanged.

        :param a: term representing the weight
        :type a: Term
        :return: current weight
        :rtype: float
        """
        if isinstance(a, Term) and a.functor == 'lfi':
            # index = int(a.args[0])
            return self._get_weight(*a.args)
        else:
            return float(a)

    def __call__(self, example):
        """Evaluate the model with its current estimates for all examples."""

        at = example.atoms
        val = example.values
        comp = example.compiled
        n = example.n
        # if type(n) == list:
        #     n = n[0]

        evidence = {}
        for a, v in zip(at, val):
            if a in evidence:
                if evidence[a] != v:
                    context = ' (found evidence({},{}) and evidence({},{}) in example {})'.format(
                        a, evidence[a], a, v, n[0] + 1)
                    raise InconsistentEvidenceError(source=a, context=context)
            else:
                evidence[a] = v
        try:
            evaluator = comp.get_evaluator(semiring=self, evidence=evidence)
        except InconsistentEvidenceError as err:
            if err.context == '':
                context = ' (example {})'.format(n[0] + 1)
            else:
                context = err.context + ' (example {})'.format(n[0] + 1)
            raise InconsistentEvidenceError(err.source, context)
        p_queries = {}
        # Probability of query given evidence
        for name, node, label in evaluator.formula.labeled():
            w = evaluator.evaluate_fact(node)
            if w < 1e-6:
                p_queries[name] = 0.0
            else:
                p_queries[name] = w
        p_evidence = evaluator.evaluate_evidence()
        return len(n), p_evidence, p_queries


def extract_evidence(pl):
    engine = DefaultEngine()
    atoms = engine.query(pl, Term('evidence', None, None))
    atoms1 = engine.query(pl, Term('evidence', None))
    atoms2 = engine.query(pl, Term('observe', None))
    for atom in atoms1 + atoms2:
        atom = atom[0]
        if atom.is_negated():
            atoms.append((-atom, Term('false')))
        else:
            atoms.append((atom, Term('true')))
    return [(at, str2bool(vl)) for at, vl in atoms]


def read_examples(*filenames):
    
    for filename in filenames:
        engine = DefaultEngine()
        
        with open(filename) as f:
            example = ''
            for line in f:
                if line.strip().startswith('---'):
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


class DefaultDict(object):

    def __init__(self, base):
        self.base = base

    def __getitem__(self, key):
        return self.base.get(key, Var(key))

    
def run_lfi(program, examples, output_model=None, **kwdargs):
    lfi = LFIProblem(program, examples, **kwdargs)
    score = lfi.run()

    if output_model is not None:
        with open(output_model, 'w') as f:
            f.write(lfi.get_model())

    names = []
    weights = []
    for i, name in enumerate(lfi.names):
        weights_i = lfi.get_weights(i)
        for w_args, w_val in weights_i:
            translate = {k: v for k, v in zip(name.probability.args[1:], w_args.args)}
            names.append(name.apply(DefaultDict(translate)))
            weights.append(w_val)

    return score, weights, names, lfi.iteration, lfi


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
    parser.add_argument('-l', '--leak-probabilities', dest='leakprob', type=float,
                        help='Add leak probabilities for evidence atoms.')
    parser.add_argument('--propagate-evidence', action='store_true',
                        dest='propagate_evidence',
                        default=True,
                        help="Enable evidence propagation")
    parser.add_argument('--dont-propagate-evidence', action='store_false',
                        dest='propagate_evidence',
                        default=True,
                        help="Disable evidence propagation")
    parser.add_argument('--normalize', action='store_true', help="Normalize AD-weights.")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--web', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('-a', '--arg', dest='args', action='append',
                        help='Pass additional arguments to the cmd_args builtin.')

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
        score, weights, names, iterations, lfi = d
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
        score, weights, names, iterations, lfi = d
        results = {'SUCCESS': True,
                   'score': score,
                   'iterations': iterations,
                   'weights': [[str(n), round(w, precision), n.loc[1], n.loc[2]]
                               for n, w in zip(names, weights)],
                   'model': lfi.get_model()
                   }
        print (json.dumps(results), file=output)
    else:
        results = {'SUCCESS': False, 'err': vars(d)}
        print (json.dumps(results), file=output)
    return 0


if __name__ == '__main__':
    main(sys.argv[1:])
