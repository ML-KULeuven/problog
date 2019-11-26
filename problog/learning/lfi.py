#! /usr/bin/env python

"""
Learning from interpretations
-----------------------------

Parameter learning for ProbLog.

Given a probabilistic program with parameterized weights and a set of partial implementations,
learns appropriate values of the parameters.


Continuous distributions
++++++++++++++++++++++++

A parametrized weight can also be a continuous normal distribution if the atom it is associated
with only appears as a head (thus is not used in any bodies of other ProbLog rules).

For example, the following GMM:

.. code-block:: prolog::
    t(0.5)::c.
    t(normal(1,10))::fa :- c.
    t(normal(10,10))::fa :- \+c.

with evidence:

.. code-block:: prolog::
    evidence(fa, 10).
    ---
    evidence(fa, 18).
    ---
    evidence(fa, 8).


Or a multivariate GMM:

.. code-block:: prolog
    t(0.5)::c.
    t(normal([1,1],[10,1,1,10]))::fa :- c.
    t(normal([10,10],[10,1,1,10]))::fa :- \+c.

with evidence:

.. code-block:: prolog
    evidence(fa, [10,11]).
    ---
    evidence(fa, [18,12]).
    ---
    evidence(fa, [8,7]).

The covariance matrix is represented as a row-based list ([[10,1],[1,10]] is [10,1,1,10]).

The GMM can also be represent compactly and as one examples:

.. code-block:: prolog
    t(0.5)::c(ID,1); t(0.5)::c(ID,2).
    comp(1). comp(2).
    t(normal(_,_),C)::fa(ID) :- comp(C), c(ID,C).

with evidence:

.. code-block:: prolog::
    evidence(fa(1), 10).
    evidence(fa(2), 18).
    evidence(fa(3), 8).


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

try:
    from typing import List, Union
except ImportError:
    List, Union = None, None

from collections import defaultdict
from itertools import chain

from problog.engine import DefaultEngine, ground
from problog.evaluator import (
    SemiringProbability,
    SemiringLogProbability,
    SemiringDensity,
    DensityValue,
)
from problog.logic import (
    Term,
    Constant,
    Clause,
    AnnotatedDisjunction,
    Or,
    Var,
    InstantiationError,
    ArithmeticError,
    term2list,
    list2term,
    Not,
)
from problog.program import PrologString, PrologFile, LogicProgram
from problog.core import ProbLogError
from problog.errors import process_error, InconsistentEvidenceError

# Scipy and Numpy are optional installs (only required for continuous variables)
try:
    import scipy.stats as stats
except ImportError:
    stats = None
try:
    import numpy as np
except ImportError:
    np = None

from problog import get_evaluatable, get_evaluatables
import traceback


def str2bool(s):
    if str(s) == "true":
        return True
    elif str(s) == "false":
        return False
    else:
        return None


def str2num(s):
    """Translate a Term that represents a number or list of numbers to observations (as Python primitives).

    :return: Tuple of (isobserved?, values)
    """
    if s.is_constant() and (s.is_float() or s.is_integer()):
        return True, s.compute_value()
    elif s.functor == ".":
        values = term2list(s)
        numvalues = []
        for value in values:
            if isinstance(value, int) or isinstance(value, float):
                numvalues.append(value)
            else:
                return None, None
        return True, tuple(numvalues)
    else:
        return None, None


cdist_names = ["normal"]


def dist_prob(d, x, eps=None, log=False, density=True):
    """Compute the density of the value x given the distribution d (use interval 2*eps around x).
    Returns a polynomial

    :param d: Distribution Term
    :param x: Value
    :param log: use log-scale computations
    :return: Probability
    """

    if stats is None or np is None:
        raise ProbLogError(
            "Continuous variables require Scipy and Numpy to be installed."
        )

    if d.functor == "normal":
        if isinstance(d.args[0], Term) and d.args[0].functor == ".":
            args = (term2list(d.args[0]), term2list(d.args[1]))
        else:
            args = d.args
        if isinstance(args[0], list):  # multivariate
            m = args[0]
            ndim = len(m)
            cov = args[1]
            if len(cov) != ndim * ndim:
                raise ValueError("Distribution parameters do not match: {}".format(d))
            cov = np.reshape(cov, (ndim, ndim))
            try:
                rv = stats.multivariate_normal(m, cov)
            except np.linalg.linalg.LinAlgError as exc:
                logger = logging.getLogger("problog_lfi")
                logger.debug(
                    "Encountered a singular covariance matrix: N({},\n{})".format(
                        m, cov
                    )
                )
                raise exc
            if log:
                raise NotImplementedError("log computations not yet supported")
            else:
                result = [0, rv.pdf(x)]
            retval = DensityValue(result)
        else:  # univariate
            m, s = map(float, d.args)
            rv = stats.norm(m, s)
            # TODO: The multiplication with eps should be avoided by working with densities
            result = rv.pdf(x)
            if log:
                try:
                    result = math.log(result)
                except ValueError:
                    result = -math.inf
            # print('dist_prob({}, {}) -> {}'.format(d, x, result))
            retval = DensityValue([0, result])

        if density:
            return retval
        else:
            return float(retval)
    raise ValueError("Distribution not supported '%s'" % d.functor)


def dist_prob_set(d, values, eps=1e-4):
    """Fit parameters based on EM.

    :param d: Distribution Term
    :param values: List of (value, weight, count)
    """
    logger = logging.getLogger("problog_lfi")
    if stats is None or np is None:
        raise ProbLogError(
            "Continuous variables require Scipy and Numpy to be installed."
        )
    if d.functor == "normal":
        if isinstance(d.args[0], Term) and d.args[0].functor == ".":
            args = (term2list(d.args[0]), term2list(d.args[1]))
        else:
            args = d.args
        if isinstance(args[0], list):  # multivariate
            # TODO: cleanup (make nice with numpy, store numpy in Term to avoid conversions?)
            pf = 0.0
            ndim = len(args[0])
            mu = np.zeros(ndim)
            cov = np.zeros((ndim, ndim))
            for value, weight, count in values:
                weight = float(weight)
                pf += weight * count
                mu += weight * count * np.array(value)
            if pf == 0.0:
                # Reuse previous distribution, no samples found
                return d
            mu /= pf
            for value, weight, count in values:
                weight = float(weight)
                xmu = np.matrix(value) - mu
                cov += weight * count * xmu.T * xmu
            cov /= pf
            s_eps = eps ** 2
            # if np.linalg.matrix_rank(cov) != ndim:
            #     # The matrix is singular, reinitialise to random value
            #     # See Bishop 9.2.1 on singularities in GMM. Better solutions exist.
            #     logger.debug('Singular matrix, reset to random values')
            #     mu = np.random.random(ndim)
            #     cov = np.diagflat([1000.0]*ndim)
            # for i in range(cov.shape[0]):
            #     if cov[i, i] < s_eps:
            #         # Covariance is corrected to not have probabilities larger than 1
            #         # Pdf is multiplied with eps to translate to prob
            #         logger.debug('Corrected covar from {} to {}'.format(cov[i, i], s_eps))
            #         cov[i, i] = s_eps
            try:
                rv = stats.multivariate_normal(mu, cov)
                if rv.pdf(mu) > 1.0 / (2 * eps):
                    logger.debug("PDF larger than 1.0/(2*eps), assume singularity")
                    raise np.linalg.linalg.LinAlgError()
            except np.linalg.linalg.LinAlgError:
                logger.debug("Singular matrix for normal dist, reset to random values")
                logger.debug("mu = {}".format(mu))
                logger.debug("cov = \n{}".format(cov))
                # The matrix is singular, reinitialise to random value
                # See Bishop 9.2.1 on singularities in GMM. Better solutions exist.
                mu = np.random.random(ndim)
                cov = np.diagflat([1000.0] * ndim)
            cov = cov.reshape(-1)
            # print('Update: {} -> normal({},{})'.format(d, mu, cov))
            # values.sort(key=lambda t: t[0])
            # for value, weight, count in values:
            #     print('({:<4}, {:7.5f}, {:<4})'.format(value, weight, count))
            return d.with_args(list2term(mu.tolist()), list2term(cov.tolist()))
        else:  # univariate
            pf = 0.0
            mu = 0.0
            var = 0.0
            for value, weight, count in values:
                weight = float(weight)
                pf += weight * count
                mu += weight * count * value
            if pf == 0.0:
                # Reuse previous distribution, no samples found
                return d
            mu /= pf
            for value, weight, count in values:
                weight = float(weight)
                var += weight * count * (value - mu) ** 2
            var /= pf
            if var < eps ** 2:
                # TODO: Is this a good approach? Should also take singularity into account
                # Std is corrected to not have probabilities larger than 1
                # Pdf is multiplied with eps to translate to prob
                std = eps
                logger.debug("Corrected std to {}".format(std))
            else:
                std = math.sqrt(
                    var
                )  # TODO: should we make this also variance to be consistent with multivariate?
            # print('Update: {} -> normal({},{})'.format(d, mu, std))
            # values.sort(key=lambda t: t[0])
            # for value, weight, count in values:
            #     print('({:<4}, {:7.5f}, {:<4})'.format(value, weight, count))
            return d.with_args(Constant(mu), Constant(std))
    raise ValueError("Distribution not supported '%s'" % d.functor)


def dist_perturb(d):
    if stats is None or np is None:
        raise ProbLogError(
            "Continuous variables require Scipy and Numpy to be installed."
        )
    if d.functor == "normal":
        if isinstance(d.args[0], Term) and d.args[0].functor == ".":
            args = (term2list(d.args[0]), term2list(d.args[1]))
        else:
            args = d.args
        if isinstance(args[0], list):  # multivariate
            mu = args[0]  # type: List[float]
            ndim = len(mu)
            cov = args[1]  # type: List[float]
            if len(cov) != ndim * ndim:
                raise ValueError("Distribution parameters do not match: {}".format(d))
            rv = stats.multivariate_normal(mu, np.reshape(cov, (ndim, ndim)) / 10)
            mu = rv.rvs()
            dn = d.with_args(list2term(mu.tolist()), list2term(cov))
            return dn
        else:  # univariate
            mu, std = map(float, d.args)
            rv = stats.norm(mu, std / 10)
            mu = float(rv.rvs())
            dn = d.with_args(Constant(mu), Constant(std))
            return dn
    raise ValueError("Distribution not supported '%s'" % d.functor)


class LFIProblem(LogicProgram):
    def __init__(
        self,
        source,
        examples,
        max_iter=10000,
        min_improv=1e-10,
        verbose=0,
        knowledge=None,
        leakprob=None,
        propagate_evidence=True,
        normalize=False,
        log=False,
        eps=1e-4,
        use_parents=False,
        **extra
    ):
        """
        Learn parameters using LFI.

        The atoms with to be learned continuous distributions can only appear in the head of a rule.

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
        :param eps: Epsilon value which is the smallest value that is used
        :type eps: float
        :param use_parents: Perform EM with P(x, parents | ev) instead of P(x | ev)
        :type use_parents: bool
        :param extra: catch all for additional parameters (not used)
        """
        # logger = logging.getLogger('problog_lfi')
        LogicProgram.__init__(self)
        self.source = source
        self._log = log
        self._eps = eps
        self._use_parents = use_parents

        # The names of the atom for which we want to learn weights.
        self.names = []
        self.bodies = []
        self.parents = []

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
        self._adatoms = []  # list AD atoms and total probability
        self._adatomc = {}  # complement of AD atom (complement that adds to prob 1.0)
        self._adparent = {}  # atom representing parent of AD
        self._catoms = set()  # Continuous atoms

    @property
    def count(self):
        """Number of parameters to learn."""
        return len(self.names)

    def add_ad(self, rem_prob, indices):
        """
        :param rem_prob: Remaining probability that can be learned (if no fixed probabilities given, this will be one)
        :param indices: Indices of atoms that together form an annotated disjunction.
        :return: None
        """
        self._adatoms.append((rem_prob, indices))
        for idx in indices:
            self._adatomc[idx] = [idxo for idxo in indices if idxo != idx]

    def count_ad(self):
        return len(self._adatoms)

    def append_ad(self, atom_index, ad_index=None):
        if ad_index is None:
            ad_index = -1
        self._adatoms[ad_index][1].append(atom_index)
        indices = self._adatoms[ad_index][1]
        for idx in indices:
            self._adatomc[idx] = [idxo for idxo in indices if idxo != idx]

    def verify_ad(self, ad_index=None):
        if ad_index is None:
            ad_index = -1
        if len(self._adatoms[ad_index][1]) == 1:
            indices = self._adatoms[ad_index][1]
            if self._use_parents:
                self._adatomc[indices[0]] = [
                    -1 - indices[0]
                ]  # No AD, complement is negative variable
            else:
                self._adatoms.pop(ad_index)
                for idx in indices:
                    del self._adatomc[idx]

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
            return [(Term("t"), weight)]

    def _set_weight(self, index, args, weight, weight_changed=None):
        # print(self._weights)
        index = int(index)
        if not args:
            assert not isinstance(self._weights[index], dict)
            self._weights[index] = weight
        elif isinstance(self._weights[index], dict):
            # self._weights[index][args] = weight
            if weight_changed and weight_changed[index]:
                self._weights[index][Term(args.functor)] += weight
            else:
                self._weights[index][Term(args.functor)] = weight
        else:
            if index in self._catoms:
                # If new t args, perturb the distribution a bit to avoid identical ones
                weight = dist_perturb(weight)
            # TODO: Shouldn't all weights be perturbed to avoid identical updates?
            # self._weights[index] = {args: weight}
            self._weights[index] = {Term(args.functor): weight}
        print(self._weights)

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

        use_parents = None
        if self._use_parents:
            use_parents = {"adatomc": self._adatomc}

        if self.propagate_evidence:
            result = ExampleSet()
            for index, example in enumerate(self.examples):
                atoms, values, cvalues = zip(*example)
                result.add(index, atoms, values, cvalues, use_parents=use_parents)
            return result
        else:
            # smarter: compile-once all examples with same atoms
            result = ExampleSet()
            for index, example in enumerate(self.examples):
                atoms, values, cvalues = zip(*example)
                result.add(index, atoms, values, cvalues, use_parents=use_parents)
            return result

    def _compile_examples(self):
        """Compile examples."""
        baseprogram = DefaultEngine(**self.extra).prepare(self)
        examples = self._process_examples()
        for i, example in enumerate(examples):
            print("Compiling example {}/{}".format(i + 1, len(examples)))
            example.compile(self, baseprogram)
        self._compiled_examples = examples

    def _process_atom(self, atom, body):
        """Returns tuple ( prob_atom, [ additional clauses ] )"""
        result = None
        if isinstance(atom, Or):
            # Annotated disjuntions are always discrete distributions
            result = self._process_atom_discr(atom, body)
        if (
            result is None
            and atom.probability
            and isinstance(atom.probability, Term)
            and len(atom.probability.args) > 0
        ):
            cdist = atom.probability.args[0]
            if isinstance(cdist, Term) and not isinstance(cdist, Var):
                if cdist.functor in cdist_names:
                    result = self._process_atom_cont(atom, body)
        if result is None:
            result = self._process_atom_discr(atom, body)
        print(str(atom) + " got processed to " + str(result))
        return result

    def _process_atom_cont(self, atom, body):
        """Returns tuple ( prob_atom, [ additional clauses ] )"""
        logger = logging.getLogger("problog_lfi")
        atoms_out = []
        extra_clauses = []

        has_lfi_fact = False

        if atom.probability and atom.probability.functor == "t":
            has_lfi_fact = True
            cdist = atom.probability.args[0]
            if isinstance(cdist, Term) and cdist.functor in cdist_names:
                start_dist = cdist
                if cdist.functor == "normal":
                    start_params = [None, None]
                    try:
                        if cdist.args[0].functor == ".":
                            start_params[0] = term2list(cdist.args[0])  # multivariate
                        else:
                            start_params[0] = float(cdist.args[0])  # univariate
                    except InstantiationError:
                        start_params[0] = None
                    except ArithmeticError:
                        start_params[0] = None
                    try:
                        if cdist.args[1].functor == ".":
                            start_params[1] = term2list(cdist.args[1])  # multivariate
                        else:
                            start_params[1] = float(cdist.args[1])  # univariate
                    except InstantiationError:
                        start_params[1] = None
                    except ArithmeticError:
                        start_params[1] = None
                else:
                    start_params = None
            else:
                start_dist = None
                start_params = None

            # Learnable probability
            # print('get start_value from {}'.format(cdist))

            # Replace anonymous variables with non-anonymous variables.
            class ReplaceAnon(object):
                def __init__(self):
                    self.cnt = 0

                def __getitem__(self, key):
                    if key == "_":
                        self.cnt += 1
                        return Var("anon_%s" % self.cnt)
                    else:
                        return Var(key)

            atom1 = atom.apply(ReplaceAnon())
            prob_args = atom.probability.args[1:]

            # 1) Introduce a new fact
            #  lfi_fact(0, t(1), 2, 3)
            #           |    |   |
            #           |    |   `-> Arguments for atom in head
            #           |    `-> Arguments for prob in head in t(_, 1)
            #           `-> Identifier for original to learn term
            # TODO: naming it clfi_fact instead of lfi_fact is not really necessary
            lfi_fact = Term(
                "clfi_fact", Constant(self.count), Term("t", *prob_args), *atom1.args
            )
            lfi_prob = Term(
                "clfi",
                Constant(self.count),
                Term("t", *prob_args),
                atom.with_probability(),
            )

            # 2) Replacement atom
            replacement = lfi_fact.with_probability(lfi_prob)
            if body is None:
                new_body = lfi_fact
            else:
                new_body = body & lfi_fact

            # 3) Create redirection clause
            extra_clauses += [Clause(atom1.with_probability(), new_body)]

            # 4) Set initial weight
            if start_dist is None:
                raise ProbLogError("No correct initial distribution defined")
            elif start_dist.functor == "normal":
                if start_params[0] is None:
                    start_params[0] = Constant(random.gauss(0, 10))
                if start_params[1] is None:
                    start_params[1] = Constant(
                        1000000
                    )  # TODO: What is a good choice here?
                start_dist = start_dist.with_args(start_params[0], start_params[1])
                self._add_weight(start_dist)

            # 5) Add name
            self._catoms.add(len(self.names))
            self.names.append(atom)
            atoms_out.append(replacement)
        else:
            # TODO: process continuous distribution for not to be learned distributions
            atoms_out.append(atom)
            raise ProbLogError(
                "Continuous distributions that do not have to be learned is not yet supported."
            )

        if has_lfi_fact:
            result = [atoms_out[0]] + extra_clauses
        else:
            if body is None:
                result = [atoms_out[0]]
            else:
                result = [Clause(atoms_out[0], body)]
        logger.debug("New clauses: " + str(result))
        return result

    def _process_atom_discr(self, atom, body):
        """Returns tuple ( prob_atom, [ additional clauses ] )"""
        if isinstance(atom, Or):
            # Annotated disjunction
            atoms = atom.to_list()
        else:
            atoms = [atom]

        atoms_out = []
        extra_clauses = []

        has_lfi_fact = False
        prior_probability = 0.0  # Sum of prior weights in AD.
        fixed_probability = 0.0  # Sum of fixed (i.e. non-learnable) weights in AD.

        num_random_weights = 0
        for atom in atoms:
            if atom.probability and atom.probability.functor == "t":
                try:
                    start_value = float(atom.probability.args[0])
                    prior_probability += float(start_value)
                except InstantiationError:
                    # Can't be converted to float => take random
                    num_random_weights += 1
                except ArithmeticError:
                    num_random_weights += 1
            elif atom.probability and atom.probability.is_constant():
                fixed_probability += float(atom.probability)

        random_weights = [random.random() for i in range(0, num_random_weights + 1)]
        norm_factor = (1.0 - prior_probability - fixed_probability) / sum(
            random_weights
        )
        random_weights = [r * norm_factor for r in random_weights]

        # First argument is probability available for learnable weights in the AD.
        self.add_ad(1.0 - fixed_probability, [])  # TODO : this adds extra ad

        for atom in atoms:
            if atom.probability and atom.probability.functor == "t":
                # t(_)::p(X) :- body.
                #
                # Translate to
                #   lfi(1)::lfi_fact_1(X).
                #   p(X) :- lfi_body_1(X).
                #   lfi_body_1(X) :- body,   lfi_fact_1(X).
                #   lfi_body_2(X) :- body, \+lfi_fact_1(X).
                #
                # For annotated disjunction: t(_)::p1(X); t(_)::p2(X) :- body.
                #   lfi1::lfi_fact_1(X); lfi2::lfi_fact_2(X); ... .
                #   p1(X) :- lfi_body_1(X).
                #   lfi_body_1(X) :- body, lfi_fact_1(X).
                #   p2(X) :- lfi_body_2(X).
                #   lfi_body_2(X) :- body, lfi_fact_2(X).
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
                class ReplaceAnon(object):  # TODO: can be defined outside of for loop?
                    def __init__(self):
                        self.cnt = 0

                    def __getitem__(self, key):
                        if key == "_":
                            self.cnt += 1
                            return Var("anon_%s" % self.cnt)
                        else:
                            return Var(key)

                atom1 = atom.apply(ReplaceAnon())
                prob_args = atom.probability.args[1:]

                # 1) Introduce a new fact
                # lfi_fact = Term('lfi_fact', Constant(self.count),      Term('t', *prob_args), *atom1.args)
                # lfi_body = Term('lfi_body', Constant(self.count),      Term('t', *prob_args), *atom1.args)
                lfi_fact = Term(
                    "lfi_fact", Constant(self.count), Term("t", *prob_args, *atom1.args)
                )
                lfi_body = Term(
                    "lfi_body", Constant(self.count), Term("t", *prob_args, *atom1.args)
                )
                # lfi_par  = Term('lfi_par',  Constant(self.count_ad()), Term('t', *prob_args), *atom1.args)
                # TODO: lfi_par should be unique for rule, not per disjunct
                # lfi_par = Term('lfi_par',   Constant(self.count),      Term('t', *prob_args), *atom1.args)
                lfi_par = Term(
                    "lfi_par", Constant(self.count), Term("t", *prob_args, *atom1.args)
                )
                # lfi_prob = Term('lfi', Constant(self.count), Term('t', *prob_args, *atom1.args))
                lfi_prob = Term("lfi", Constant(self.count), Term("t"))

                # 2) Replacement atom
                replacement = lfi_fact.with_probability(lfi_prob)
                if self._use_parents:
                    if body is None:
                        new_body = Term("true")
                    else:
                        new_body = body
                else:
                    if body is None:
                        new_body = lfi_fact
                    else:
                        new_body = body & lfi_fact

                # 3) Create redirection clause
                if self._use_parents:
                    extra_clauses += [
                        Clause(atom1.with_probability(), lfi_body),
                        Clause(lfi_body, lfi_par & lfi_fact),
                        Clause(lfi_par, new_body),
                    ]
                else:
                    extra_clauses += [Clause(atom1.with_probability(), new_body)]

                self.append_ad(len(self._weights))
                # 4) Set initial weight
                if start_value is None:
                    start_value = random_weights.pop(-1)
                    self._add_weight(start_value)
                else:
                    self._add_weight(start_value)

                # 5) Add name
                self.names.append(atom)
                if self._use_parents:
                    self.bodies.append(lfi_body)
                    self.parents.append(lfi_par)
                atoms_out.append(replacement)
            else:
                atoms_out.append(atom)

        self.verify_ad()

        if has_lfi_fact:
            if len(atoms) == 1:  # Simple clause
                return [atoms_out[0]] + extra_clauses
            else:
                return [AnnotatedDisjunction(atoms_out, Term("true"))] + extra_clauses
        else:
            if len(atoms) == 1:
                if body is None:
                    return [atoms_out[0]]
                else:
                    return [Clause(atoms_out[0], body)]
            else:
                if body is None:
                    body = Term("true")
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
            if atom.probability and atom.probability.functor == "t":
                assert atom in self.names
                # assert (t_args is None or atom.probability.args == t_args)
                # t_args = atom.probability.args

                index = self.output_names.index(atom)
                weights = self.get_weights(index)

                for w_args, w_val in weights:
                    translate = tuple(zip(atom.probability.args[1:], w_args.args))
                    if isinstance(w_val, Term) and w_val.functor in cdist_names:
                        # Keep the complex structure that represents the distribution
                        transforms[translate].append(atom.with_probability(w_val))
                    else:
                        transforms[translate].append(
                            atom.with_probability(Constant(w_val))
                        )
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
                if clause.head.functor == "query" and clause.head.arity == 1:
                    continue
                extra_clauses = process_atom(clause.head, clause.body)
                for extra in extra_clauses:
                    print("rule", extra)
                    yield extra
            elif isinstance(clause, AnnotatedDisjunction):
                extra_clauses = process_atom(Or.from_list(clause.heads), clause.body)
                for extra in extra_clauses:
                    print("rule", extra)
                    yield extra
            else:
                if clause.functor == "query" and clause.arity == 1:
                    continue
                # Fact
                extra_clauses = process_atom(clause, None)
                for extra in extra_clauses:
                    print("rule", extra)
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
        logging.getLogger("problog_lfi").debug("Evaluating examples ...")

        if self._log:
            evaluator = ExampleEvaluatorLog(
                self._weights, eps=self._eps, use_parents=self._use_parents
            )
        else:
            evaluator = ExampleEvaluator(
                self._weights, eps=self._eps, use_parents=self._use_parents
            )

        return list(chain.from_iterable(map(evaluator, self._compiled_examples)))

    def _update(self, results):
        """Update the current estimates based on the latest evaluation results."""
        print("_update", results)
        logger = logging.getLogger("problog_lfi")
        # fact_marg = defaultdict(DensityValue)
        fact_marg = defaultdict(int)
        fact_body = defaultdict(int)
        fact_par = defaultdict(int)
        fact_count = defaultdict(int)
        fact_values = dict()
        score = 0.0
        for m, pEvidence, result, p_values in results:
            # if not isinstance(pEvidence, DensityValue):
            #     pEvidence = DensityValue.wrap(pEvidence)
            print("_update.result", result)
            par_marg = dict()
            # print('p_values', p_values)
            for fact, value in result.items():
                print(fact, value)
                index = fact.args[0:2]
                # if not index in fact_marg:
                #     fact_marg[index] = polynomial.polynomial.polyzero
                # if not isinstance(value, DensityValue):
                #     value = DensityValue.wrap(value)
                if fact.functor == "lfi_fact":
                    fact_marg[index] += value * m
                if fact.functor == "lfi_body":
                    fact_body[index] += value * m
                elif fact.functor == "lfi_par":
                    if index in par_marg:
                        if par_marg[index] != value:
                            raise Exception(
                                "Different parent margs for {}={} and previous {}={}".format(
                                    fact, value, index, par_marg[index]
                                )
                            )
                    par_marg[index] = value
                    for o_index in self._adatomc[index[0]]:
                        if o_index >= 0:
                            par_marg[(o_index, index[1])] = value
                fact_count[index] += m
                if fact in p_values:
                    # print('fact in p_values', fact)
                    k = (index[0], index[1])
                    if k not in fact_values:
                        fact_values[k] = (self._get_weight(index[0], index[1]), list())
                    p_value = p_values[fact]
                    fact_values[k][1].append((p_value, value, m))
            for index, value in par_marg.items():
                print("value[{}]={} ({})".format(index, value, m))
                fact_par[index] += value * m
            try:
                if isinstance(pEvidence, DensityValue):
                    pEvidence = pEvidence.value()
                score += math.log(pEvidence)
            except ValueError:
                logger.debug("Pr(evidence) == 0.0")
                # raise ProbLogError('Inconsistent evidence when updating')

        if self._use_parents:
            update_list = fact_body
        else:
            update_list = fact_marg

        weight_changed = []
        for index in update_list:
            if len(weight_changed) <= float(index[0]):
                weight_changed.append(False)
        print(weight_changed)
        for index in update_list:
            k = (index[0], index[1])
            if k in fact_values:
                logger.debug(
                    "Update continuous distribution {}: ".format(index)
                    + ", ".join([str(v) for v in fact_values[k]])
                )
                self._set_weight(
                    index[0],
                    index[1],
                    dist_prob_set(
                        *fact_values[k], eps=self._eps, weight_changed=weight_changed
                    ),
                )
                weight_changed[int(index[0])] = True
            else:
                if self._use_parents:
                    if float(fact_body[index]) == 0.0:
                        prob = 0.0
                    else:
                        print(fact_par[index])
                        temp = dict()
                        ids, vars = zip(*list(fact_par.keys()))
                        for id in set(ids):
                            temp[id] = 0
                            for var in set(vars):
                                temp[id] += fact_par[(id, var)]
                        prob = float(fact_body[index]) / float(temp[index[0]])
                        # prob = float(fact_body[index]) / float(fact_par[index])
                    logger.debug(
                        "Update probabilistic fact {}: {} / {} = {}".format(
                            index, fact_body[index], fact_par[index], prob
                        )
                    )
                    self._set_weight(
                        index[0], index[1], prob, weight_changed=weight_changed
                    )
                    weight_changed[int(index[0])] = True
                elif fact_count[index] > 0:
                    # TODO: This assumes the estimate for true and false add up to one (not true for AD)
                    prob = float(fact_marg[index]) / float(fact_count[index])
                    logger.debug(
                        "Update probabilistic fact {}: {} / {} = {}".format(
                            index, fact_marg[index], fact_count[index], prob
                        )
                    )
                    self._set_weight(
                        index[0], index[1], prob, weight_changed=weight_changed
                    )
                    weight_changed[int(index[0])] = True

        if self._enable_normalize:
            self._normalize_weights()

        return score

    def _normalize_weights(self):
        # TODO: too late here, AD should be taken into account in _update
        # Derivation is sum(all values for var=k) / sum(all values for i sum(all values for var=i))
        print("_adatoms", self._adatoms)

        for available_prob, idx in self._adatoms:
            if len(idx) == 1:
                continue
            keys = set()
            for i in idx:
                for key, val in self.get_weights(i):
                    keys.add(key)
            if len(keys) > 1:
                try:
                    keys.remove(Term("t"))
                except KeyError:
                    pass

            keys = list(keys)
            if len(keys) > 1:
                w = 0.0
                for key in keys:
                    w += sum(self._get_weight(i, key, strict=False) for i in idx)
                if w != 0:
                    n = (
                        available_prob / w
                    )  # Some part of probability might be taken by non-learnable weights in AD.
                else:
                    n = available_prob
                for i in idx:
                    self._set_weight(
                        i,
                        list(self._weights[i].keys())[0],
                        self._get_weight(
                            i, list(self._weights[i].keys())[0], strict=False
                        )
                        * n,
                    )
            else:
                w = sum(self._get_weight(i, keys[0], strict=False) for i in idx)
                if w != 0:
                    n = (
                        available_prob / w
                    )  # Some part of probability might be taken by non-learnable weights in AD.
                else:
                    n = available_prob
                for i in idx:
                    self._set_weight(
                        i, keys[0], self._get_weight(i, keys[0], strict=False) * n
                    )

    def step(self):
        self.iteration += 1
        results = self._evaluate_examples()
        logging.getLogger("problog_lfi").info(
            "Step {}: {}".format(self.iteration, results)
        )
        return self._update(results)

    def get_model(self):
        self.output_mode = True
        lines = []
        for l in self:
            lines.append("%s." % l)
        lines.append("")
        self.output_mode = False
        return "\n".join(lines)

    def run(self):
        self.prepare()
        if self._use_parents:
            logging.getLogger("problog_lfi").info("Weights to learn: %s" % self.names)
            logging.getLogger("problog_lfi").info("Bodies: %s" % self.bodies)
            logging.getLogger("problog_lfi").info("Parents: %s" % self.parents)
        else:
            logging.getLogger("problog_lfi").info("Weights to learn: %s" % self.names)
        logging.getLogger("problog_lfi").info("Initial weights: %s" % self._weights)
        delta = 1000
        prev_score = -1e10
        # TODO: isn't this comparing delta i logprob with min_improv in prob?
        while self.iteration < self.max_iter and (delta < 0 or delta > self.min_improv):
            score = self.step()
            logging.getLogger("problog_lfi").info(
                "Weights after iteration %s: %s" % (self.iteration, self._weights)
            )
            logging.getLogger("problog_lfi").info(
                "Score after iteration %s: %s" % (self.iteration, score)
            )
            delta = score - prev_score
            prev_score = score
        return prev_score


class ExampleSet(object):
    def __init__(self):
        self._examples = {}

    def add(self, index, atoms, values, cvalues, use_parents=None):
        ex = self._examples.get((atoms, values))
        if ex is None:
            self._examples[(atoms, values)] = Example(
                index, atoms, values, cvalues, use_parents=use_parents
            )
        else:
            ex.add_index(index, cvalues)

    def __iter__(self):
        return iter(self._examples.values())

    def __len__(self):
        return len(self._examples)


class Example(object):
    def __init__(self, index, atoms, values, cvalues, use_parents=None):
        """An example consists of a list of atoms and their corresponding values (True/False).

        Different continuous values are all mapped to True and stored in self.n.
        """
        self.atoms = tuple(atoms)
        self.values = tuple(values)
        self.compiled = []
        self.n = {tuple(cvalues): [index]}
        self._use_parents = use_parents

    def __hash__(self):
        return hash((self.atoms, self.values))

    def __eq__(self, other):
        if other is None:
            return False
        return self.atoms == other.atoms and self.values == other.values

    def compile(self, lfi, baseprogram):
        ground_program = None  # Let the grounder decide
        print("compile grounding:")
        print(baseprogram.to_prolog())
        print(baseprogram)
        print("...")
        print(ground_program)

        print(self.atoms)

        ground_program = ground(
            baseprogram,
            ground_program,
            evidence=list(zip(self.atoms, self.values)),
            propagate_evidence=lfi.propagate_evidence,
        )
        print("...")
        # print(ground_program.to_prolog())
        print(ground_program)

        lfi_queries = []
        for i, node, t in ground_program:
            if (
                t == "atom"
                and isinstance(node.probability, Term)
                and node.probability.functor == "lfi"
            ):
                factargs = ()
                print("node.identifier", node.identifier)
                if type(node.identifier) == tuple:
                    factargs = node.identifier[1]
                # fact = Term('lfi_fact', node.probability.args[0], node.probability.args[1], *factargs)
                # fact = Term('lfi_fact', node.probability.args[0], node.probability.args[1])
                fact = Term("lfi_fact", node.probability.args[0], Term("t", *factargs))
                print("Adding query: ", fact, i)
                ground_program.add_query(fact, i)
                if self._use_parents:
                    # tmp_body = Term('lfi_body', node.probability.args[0], node.probability.args[1], *factargs)
                    # tmp_body = Term('lfi_body', node.probability.args[0], node.probability.args[1])
                    tmp_body = Term(
                        "lfi_body", node.probability.args[0], Term("t", *factargs)
                    )
                    lfi_queries.append(tmp_body)
                    print("Adding query: ", tmp_body, i)
                    # tmp_par = Term('lfi_par', node.probability.args[0], node.probability.args[1], *factargs)
                    # tmp_par = Term('lfi_par', node.probability.args[0], node.probability.args[1])
                    tmp_par = Term(
                        "lfi_par", node.probability.args[0], Term("t", *factargs)
                    )
                    lfi_queries.append(tmp_par)
                    print("Adding query: ", tmp_par, i)
            elif (
                t == "atom"
                and isinstance(node.probability, Term)
                and node.probability.functor == "clfi"
            ):
                factargs = ()
                if type(node.identifier) == tuple:
                    factargs = node.identifier[1]
                # fact = Term('clfi_fact', node.probability.args[0], node.probability.args[1], *factargs)
                # fact = Term('clfi_fact', node.probability.args[0], node.probability.args[1])
                fact = Term("clfi_fact", node.probability.args[0], Term("t", *factargs))
                ground_program.add_query(fact, i)
            elif t == "atom":
                # TODO: check if non-lfi and continuous and save locations to replace later
                #       lfi continuous probs are associated with lfi/2
                pass

        if self._use_parents:
            ground_program = ground(
                baseprogram,
                ground_program,
                evidence=list(zip(self.atoms, self.values)),
                propagate_evidence=lfi.propagate_evidence,
                queries=lfi_queries,
            )
            print("New ground_program")
            print(ground_program)

        self.compiled = lfi.knowledge.create_from(ground_program)
        print("Compiled program")
        print(self.compiled)

    def add_index(self, index, cvalues):
        k = tuple(cvalues)
        if k in self.n:
            self.n[k].append(index)
        else:
            self.n[k] = [index]


class ExampleEvaluator(SemiringDensity):
    def __init__(self, weights, eps, use_parents=None):
        SemiringDensity.__init__(self)
        self._weights = weights
        self._cevidence = None
        self._eps = eps
        self._use_parents = use_parents

    def _get_weight(self, index, args, strict=True):
        index = int(index)
        weight = self._weights[index]
        if isinstance(weight, dict):
            if strict:
                return weight[args]
            else:
                return weight.get(args)
        else:
            return weight

    def _get_cweight(self, index, args, atom, strict=True):
        # TODO: Should we cache this? This method is called multiple times with the same arguments
        index = int(index)
        dist = self._weights[index]
        if isinstance(dist, dict):
            if strict:
                dist = dist[args]
            else:
                raise ProbLogError(
                    "Continuous distribution is not available for {}, {}".format(
                        index, args
                    )
                )
        if not isinstance(dist, Term):
            raise ProbLogError(
                "Expected a continuous distribution, got {}".format(dist)
            )
        value = self._cevidence.get(atom)
        if value is not None:
            p = dist_prob(dist, value, eps=self._eps)
        else:
            raise ProbLogError("Expected continuous evidence for {}")
        return p

    def is_dsp(self):
        """Indicates whether this semiring requires solving a disjoint sum problem."""
        return True

    def in_domain(self, a):
        return True  # TODO implement

    def value(self, a):
        """Overrides from SemiringProbability.
        Replaces a weight of the form ``lfi(i, t(...))`` by its current estimated value.
        Other weights are passed through unchanged.

        :param a: term representing the weight
        :type a: Term
        :return: current weight
        :rtype: float
        """
        if isinstance(a, Term) and a.functor == "lfi":
            # index = int(a.args[0])
            return self._get_weight(*a.args)
        elif isinstance(a, Term) and a.functor == "clfi":
            return self._get_cweight(*a.args)
        else:
            return float(a)

    def __call__(self, example):
        print("__call__")
        """Evaluate the model with its current estimates for all examples."""
        # print('=========>>>')
        at = example.atoms
        val = example.values
        comp = example.compiled
        results = []
        for cval, n in example.n.items():
            results.append(self._call_internal(at, val, cval, comp, n))
        # print('<<<=========')
        print("__call__.results = ", results)
        return results

    def _call_internal(self, at, val, cval, comp, n):
        print("__call_internal__")
        # print('=========')
        # print('ExampleEvaluator.__call__({},{},{},{})'.format(n, at, val, cval))
        # print('_weights: ', self._weights)
        evidence = {}
        self._cevidence = {}
        # p_values = {}
        for a, v, cv in zip(at, val, cval):
            if a in evidence:
                if cv is not None:
                    if self._cevidence[a] != cv:
                        context = " (found evidence({},{}) and evidence({},{}) in example {})".format(
                            a,
                            evidence[a],
                            a,
                            cv,
                            ",".join([str(ni) for ni in n])
                            if isinstance(n, list)
                            else n + 1,
                        )
                        raise InconsistentEvidenceError(source=a, context=context)
                if evidence[a] != v:
                    context = " (found evidence({},{}) and evidence({},{}) in example {})".format(
                        a,
                        evidence[a],
                        a,
                        v,
                        ",".join([str(ni) for ni in n])
                        if isinstance(n, list)
                        else n + 1,
                    )
                    raise InconsistentEvidenceError(source=a, context=context)
            else:
                if cv is not None:
                    self._cevidence[a] = cv
                evidence[a] = v

        p_values = {}
        # TODO: this loop is not required if there are no clfi_facts
        for idx, node, ty in comp:
            if ty == "atom":
                name = node.name
                if (
                    name is not None and name.functor == "clfi_fact"
                ):  # TODO: when is this wrapped in 'choice'? Before compilation?
                    clfi = node.probability
                    ev_atom = clfi.args[2]
                    value = self._cevidence.get(ev_atom)
                    if value is not None:
                        p_values[node.name] = value

        try:
            # TODO: The next step generates the entire formula if it is density and this is redone later (caching?)
            evaluator = comp.get_evaluator(semiring=self, evidence=evidence)
        except InconsistentEvidenceError as err:
            n = ",".join([str(ni + 1) for ni in n]) if isinstance(n, list) else n + 1
            context = err.context + " (example {})".format(n)
            raise InconsistentEvidenceError(err.source, context)

        p_queries = {}
        # Probability of query given evidence
        for name, node, label in evaluator.formula.labeled():
            if self._use_parents and name.functor not in ["lfi_body", "lfi_par"]:
                continue
            print("evaluate start {}".format(name), node)
            w = evaluator.evaluate(node)
            print("evaluate {}: {} ({})".format(name, w, len(n)))
            # w = evaluator.evaluate_fact(node)
            # print ("WWW", w, w1, w2)
            if isinstance(w, DensityValue):
                # print("{} => {}".format(w, float(w)))
                # w = float(w)
                p_queries[name] = w
            elif w < 1e-6:  # TODO: too high for multivariate dists?
                p_queries[name] = 0.0
            else:
                p_queries[name] = w
        print("_call_internal.evaluate_evidence")
        p_evidence = evaluator.evaluate_evidence()
        print(
            "_call_internal.result",
            p_evidence,
            "\n",
            p_queries,
            "\n\n ".join([str(v) for v in p_values.items()]),
        )

        return len(n), p_evidence, p_queries, p_values


class ExampleEvaluatorLog(SemiringLogProbability):
    def __init__(self, weights, eps, use_parents=None):
        SemiringLogProbability.__init__(self)
        self._weights = weights
        self._cevidence = None
        self._eps = eps
        self._use_parents = use_parents

    def _get_weight(self, index, args, strict=True):
        index = int(index)
        weight = self._weights[index]
        if isinstance(weight, dict):
            if strict:
                weight = weight[args]
            else:
                weight = weight.get(args, 0.0)
        try:
            result = math.log(weight)
        except ValueError:
            return float("-inf")
        return result

    def _get_cweight(self, index, args, atom, strict=True):
        # TODO: Should we cache this? This method is called multiple times with the same arguments
        index = int(index)
        dist = self._weights[index]
        if isinstance(dist, dict):
            if strict:
                dist = dist[args]
            else:
                raise ProbLogError(
                    "Continuous distribution is not available for {}, {}".format(
                        index, args
                    )
                )
        if not isinstance(dist, Term):
            raise ProbLogError(
                "Expected a continuous distribution, got {}".format(dist)
            )
        value = self._cevidence.get(atom)
        if value is not None:
            p = dist_prob(dist, value, log=True, eps=self._eps)
        else:
            raise ProbLogError("Expected continuous evidence for {}")
        return p

    def value(self, a):
        """Overrides from SemiringProbability.
        Replaces a weight of the form ``lfi(i, t(...))`` by its current estimated value.
        Other weights are passed through unchanged.

        :param a: term representing the weight
        :type a: Term
        :return: current weight
        :rtype: float
        """
        if isinstance(a, Term) and a.functor == "lfi":
            # index = int(a.args[0])
            rval = self._get_weight(*a.args)
        elif isinstance(a, Term) and a.functor == "clfi":
            rval = self._get_cweight(*a.args)
        else:
            rval = math.log(float(a))
        return rval

    def __call__(self, example):
        """Evaluate the model with its current estimates for all examples."""
        # print('=========>>>')
        at = example.atoms
        val = example.values
        comp = example.compiled
        results = []
        for cval, n in example.n.items():
            results.append(self._call_internal(at, val, cval, comp, n))
        # print('<<<=========')
        return results

    def _call_internal(self, at, val, cval, comp, n):
        # print('=========')
        # print('ExampleEvaluator.__call__({},{},{},{})'.format(n, at, val, cval))
        # print('_weights: ', self._weights)
        evidence = {}
        self._cevidence = {}
        # p_values = {}
        for a, v, cv in zip(at, val, cval):
            # print('__call__', a, v, cv)
            if a in evidence:
                if cv is not None:
                    if self._cevidence[a] != cv:
                        context = " (found evidence({},{}) and evidence({},{}) in example {})".format(
                            a,
                            evidence[a],
                            a,
                            cv,
                            ",".join([str(ni) for ni in n])
                            if isinstance(n, list)
                            else n + 1,
                        )
                        raise InconsistentEvidenceError(source=a, context=context)
                if evidence[a] != v:
                    context = " (found evidence({},{}) and evidence({},{}) in example {})".format(
                        a,
                        evidence[a],
                        a,
                        v,
                        ",".join([str(ni) for ni in n])
                        if isinstance(n, list)
                        else n + 1,
                    )
                    raise InconsistentEvidenceError(source=a, context=context)
            else:
                if cv is not None:
                    self._cevidence[a] = cv
                evidence[a] = v

        p_values = {}
        # TODO: this loop is not required if there are no clfi_facts
        for idx, node, ty in comp:
            if ty == "atom":
                name = node.name
                # TODO: when is this wrapped in 'choice'? Before compilation?
                if name is not None and name.functor == "clfi_fact":
                    clfi = node.probability
                    ev_atom = clfi.args[2]
                    value = self._cevidence.get(ev_atom)
                    if value is not None:
                        p_values[node.name] = value

        try:
            evaluator = comp.get_evaluator(semiring=self, evidence=evidence)
        except InconsistentEvidenceError as err:
            n = ",".join([str(ni + 1) for ni in n]) if isinstance(n, list) else n + 1
            context = err.context + " (example {})".format(n)
            raise InconsistentEvidenceError(err.source, context)

        p_queries = {}
        # Probability of query given evidence
        for name, node, label in evaluator.formula.labeled():
            w = evaluator.evaluate_fact(node)
            # if w < 1e-6:  # TODO: too high for multivariate dists
            #     print('Set w to 0: ', w)
            #     p_queries[name] = 0.0
            # else:
            p_queries[name] = w
        # TODO: p_evidence becomes too small for many continuous observations
        p_evidence = evaluator.evaluate_evidence()
        # print('__call__.result', p_evidence, '\n', p_queries, '\n', '\n '.join([str(v) for v in p_values.items()]))
        return len(n), p_evidence, p_queries, p_values


def extract_evidence(pl):
    engine = DefaultEngine()
    atoms = engine.query(pl, Term("evidence", None, None))
    atoms1 = engine.query(pl, Term("evidence", None))
    atoms2 = engine.query(pl, Term("observe", None))
    for atom in atoms1 + atoms2:
        atom = atom[0]
        if atom.is_negated():
            atoms.append((-atom, Term("false")))
        else:
            atoms.append((atom, Term("true")))
    result = []
    for at, vl in atoms:
        vlr = str2bool(vl)
        vlv = None
        if vlr is None:  # TODO: also check that atom is a continuous distribution
            vlr, vlv = str2num(vl)
        result.append((at, vlr, vlv))
    # return [(at, str2bool(vl)) for at, vl in atoms]
    return result


def read_examples(*filenames):
    for filename in filenames:
        engine = DefaultEngine()

        with open(filename) as f:
            example = ""
            for line in f:
                if line.strip().startswith("---"):
                    pl = PrologString(example)
                    atoms = extract_evidence(pl)
                    if len(atoms) > 0:
                        yield atoms
                    example = ""
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
        with open(output_model, "w") as f:
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

    parser = argparse.ArgumentParser(
        description="Learning from interpretations with ProbLog"
    )
    parser.add_argument("model")
    parser.add_argument("examples", nargs="+")
    parser.add_argument("-n", dest="max_iter", default=10000, type=int)
    parser.add_argument("-d", dest="min_improv", default=1e-10, type=float)
    parser.add_argument(
        "-O",
        "--output-model",
        type=str,
        default=None,
        help="write resulting model to given file",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="write output to file"
    )
    parser.add_argument(
        "-k",
        "--knowledge",
        dest="koption",
        choices=get_evaluatables(),
        default=None,
        help="knowledge compilation tool",
    )
    parser.add_argument(
        "-l",
        "--leak-probabilities",
        dest="leakprob",
        type=float,
        help="Add leak probabilities for evidence atoms.",
    )
    parser.add_argument(
        "--propagate-evidence",
        action="store_true",
        dest="propagate_evidence",
        default=True,
        help="Enable evidence propagation",
    )
    parser.add_argument(
        "--dont-propagate-evidence",
        action="store_false",
        dest="propagate_evidence",
        default=True,
        help="Disable evidence propagation",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-4,
        help="Smallest difference between continuous values (default 1e-4)",
    )
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize",
        action="store_true",
        dest="normalize",
        default=True,
        help="Normalize AD-weights (default).",
    )
    normalize_group.add_argument(
        "--nonormalize",
        action="store_false",
        dest="normalize",
        help="Do not normalize AD-weights.",
    )
    parser.add_argument(
        "--useparents",
        action="store_true",
        dest="use_parents",
        help="Use parents to compute expectation",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--web", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "-a",
        "--arg",
        dest="args",
        action="append",
        help="Pass additional arguments to the cmd_args builtin.",
    )

    return parser


def create_logger(name, verbose):
    levels = [logging.WARNING, logging.INFO, logging.DEBUG] + list(range(9, 0, -1))
    verbose = max(0, min(len(levels) - 1, verbose))
    logger = logging.getLogger(name)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
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
        outf = open(args.output, "w")

    create_logger("problog_lfi", args.verbose)
    create_logger("problog", args.verbose - 1)

    program = PrologFile(args.model)
    examples = list(read_examples(*args.examples))
    if len(examples) == 0:
        logging.getLogger("problog_lfi").warning("no examples specified")
    else:
        logging.getLogger("problog_lfi").info("Number of examples: %s" % len(examples))
    options = vars(args)
    del options["examples"]

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
        weights_print = []
        for weight in weights:
            if isinstance(weight, Term) and weight.functor in cdist_names:
                weights_print.append(weight)
            else:
                weights_print.append(round(float(weight), precision))
        print(score, weights, names, iterations, file=output)
        return 0
    else:
        print(process_error(d), file=output)
        return 1


def print_result_json(d, output, precision=8):
    import json

    success, d = d
    if success:
        score, weights, names, iterations, lfi = d
        results = {
            "SUCCESS": True,
            "score": score,
            "iterations": iterations,
            "weights": [
                [str(n), round(w, precision), n.loc[1], n.loc[2]]
                for n, w in zip(names, weights)
            ],
            "model": lfi.get_model(),
        }
        print(json.dumps(results), file=output)
    else:
        results = {"SUCCESS": False, "err": vars(d)}
        print(json.dumps(results), file=output)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
