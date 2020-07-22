#! /usr/bin/env python

"""
Learning from interpretations
-----------------------------
Parameter learning for ProbLog.
Given a probabilistic program with parameterized weights and a set of partial implementations,
learns appropriate values of the parameters.

Continuous distributions (Not-Implemented Yet)
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
from problog.util import init_logger
from logging import getLogger
from problog.engine import DefaultEngine, ground
from problog.evaluator import SemiringLogProbability, SemiringDensity, DensityValue
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
)
from problog.program import PrologString, PrologFile, LogicProgram
from problog.errors import InconsistentEvidenceError, process_error
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
        **extra
    ):
        """
        Learn parameters using LFI.
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
        :param extra: catch all for additional parameters (not used)
        """
        LogicProgram.__init__(self)
        self.source = source
        self._log = log
        self._eps = eps

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

            self._adatomc[indices[0]] = [
                -1 - indices[0]
            ]  # No AD, complement is negative variable

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
        index = int(index)
        if not args:
            # assert not isinstance(self._weights[index], dict)
            self._weights[index] = weight
        elif isinstance(self._weights[index], dict):
            if weight_changed and weight_changed[index]:
                # self._weights[index][Term(args.functor)] += weight
                self._weights[index][Term("t")] += weight
            else:
                # self._weights[index][Term(args.functor)] = weight
                self._weights[index][Term("t")] = weight
        else:
            # self._weights[index] = {Term(args.functor): weight}
            self._weights[index] = {Term("t"): weight}

    def _add_weight(self, weight):
        self._weights.append(weight)

    def _process_examples(self):
        """Process examples by grouping together examples with similar structure.
        :return: example groups based on evidence atoms
        :rtype: dict of atoms : values for examples
        """
        logger = getLogger("problog_lfi")
        # value can be True / False / None
        # ( atom ), ( ( value, ... ), ... )

        # Simple implementation: don't add neutral evidence.

        # ad_groups is a list of lists where each list contains an AD
        ad_groups = list()
        for ad in self._adatoms:
            # if it's an AD group
            if len(ad[1]) > 1:
                ad_list = []
                for var in ad[1]:
                    ad_list.append(Term(self.names[var].functor, *self.names[var].args))
                ad_groups.append(tuple(ad_list))
        logger.debug("AD Groups\t\t:" + str(ad_groups))

        def multiple_true(d):
            """
            This function recognizes inconsistent evidence s.t. more than one term is True in AD.
            :param d: dictionary of ADs in form {term: value}
                    value can be True, False, None, "Template"
            :return: whether more than one value is True
            """
            true_count = sum(v is True for v in d.values())
            return true_count > 1

        def all_false(d):
            """
            This function recognizes inconsistent evidence s.t. all values are False in AD.
            :param d: dictionary of ADs in form {term: value}
                    value can be True, False, None, "Template"
            :return: whether all values are False
            """
            # false_count should be the same as the length of d
            false_count = sum(v is False for v in d.values())
            return false_count == len(d)

        def all_false_except_one(d):
            """
            This function recognizes incomplete evidence s.t.
            the non-False value in AD needs to be set to True.
            :param d: dictionary of ADs in form {term: value}
                    value can be True, False, None, "Template"
            :return: whether all values except one are False
            """
            false_count = sum(v is False for v in d.values())
            the_left_is_none = bool(sum(v is None for v in d.values()))
            return (false_count == len(d) - 1) and the_left_is_none

        def getADtemplate(d, atom=None):
            """
            This function gets atom's complement AD template.
            This should only be used when the AD contains non-ground terms.
            :param d: dictionary of ADs in form {term: value}
                    value can be True, False, None, "Template"
            :param atom: an evidence
            :return: atom's complement AD template
            """
            if atom is not None:
                temp_dict = {
                    k: v
                    for k, v in d.items()
                    if v == "Template" and atom.signature != k.signature
                }
                return temp_dict
            else:
                temp_dict = {k: v for k, v in d.items() if v == "Template"}
                return temp_dict

        def add_to_ad_evidence(pair, l, ADtemplate):
            """
            :param pair: a new pair of (atom, value)
            :param l: a list of dictionaries, all dictionaries need to have the same format
            :return:
            """
            (k, v) = pair
            # if entry k exists, update the value with k
            for d in l:
                if k in d:
                    d[k] = v
                    return
            # if entry k does not exist, create a new dictionary from template
            # and instantiate it with k
            new_d = dict()
            for temp_k in ADtemplate.keys():
                new_key = Term(temp_k.functor, *k.args)
                new_d[new_key] = None
            # put v in there
            new_d[k] = v
            l.append(new_d)

        if self.propagate_evidence:
            result = ExampleSet()
            inconsistent = False
            # iterate over all examples given in .ev
            for index, example in enumerate(self.examples):
                ad_evidences = []
                non_ad_evidence = {}
                for ad_group in ad_groups:
                    # create a dictionary to memorize what evidence is given in AD
                    d = dict()
                    # TODO: what if the AD contains both ground and non-ground????
                    # e.g. t(_)::a; t(_)::b(X)
                    for var in ad_group:
                        if var.is_ground():
                            d[var] = None  # for ground unknown evidence
                        else:
                            d[var] = "Template"  # for unground unknown evidence
                    ad_evidences.append(d)

                # add all evidence in the example to ad_evidences
                for atom, value, cvalue in example:
                    # if atom has a tunable probability to learn
                    if any([atom.signature == name.signature for name in self.names]):
                        # Propositional evidence
                        if len(atom.args) == 0:
                            # insert evidence
                            for d in ad_evidences:
                                if atom in d:
                                    d[atom] = value
                            non_ad_evidence[
                                atom
                            ] = value  # TODO: what does this capture?
                        # First Order evidence
                        else:
                            # find the right AD dictionary : AD_dict
                            AD_dict = None
                            for d in ad_evidences:
                                if any([atom.signature == k.signature for k in d]):
                                    AD_dict = d
                            # if the instantiation is new, add it as a key to the dictionary
                            if AD_dict and AD_dict.get(atom) is None:
                                AD_dict[atom] = value
                                # also add other AD parts in the dictionary with value==None
                                other_ADs = getADtemplate(AD_dict, atom)
                                for otherAD in other_ADs.keys():
                                    new_key = Term(otherAD.functor, *atom.args)
                                    AD_dict[new_key] = AD_dict.get(new_key, None)
                            else:
                                non_ad_evidence[atom] = value
                    else:
                        non_ad_evidence[atom] = value

                # grounded_ad_evidences contains all usable evidence (gound, not template)
                grounded_ad_evidences = []
                for d in ad_evidences:
                    # for first order evidence dictionaries
                    if "Template" in d.values():
                        # new_ad_evidence is a list of dictionaries
                        # each dictionary is a group of the AD template instantiation
                        new_ad_evidence = list()
                        # get template AD evidence
                        ADtemplate = getADtemplate(d)
                        # group all pairs according to ADtemplate
                        for k, v in d.items():
                            if v is not "Template":
                                add_to_ad_evidence((k, v), new_ad_evidence, ADtemplate)
                        grounded_ad_evidences += new_ad_evidence
                    # for propositional evidence dictionaries
                    else:
                        # simply us them
                        grounded_ad_evidences.append(d)

                # # print(grounded_ad_evidences)

                inconsistent_example = False
                for i, d in enumerate(grounded_ad_evidences):
                    # inconsistent1 = multiple_true(d)
                    inconsistent2 = all_false(d)
                    add_compliment = all_false_except_one(d)

                    if inconsistent2:
                        inconsistent_example = True
                        continue
                    elif add_compliment:
                        for key, value in d.items():
                            if value is None:
                                grounded_ad_evidences[i][key] = True

                if not inconsistent_example and len(grounded_ad_evidences) > 0:
                    # There are (fully tunable) ADs in the program
                    evidence_set = set()
                    for d in grounded_ad_evidences:
                        for key, value in d.items():
                            if value is not None:
                                evidence_set.add((key, value, None))

                    for key, value in non_ad_evidence.items():
                        evidence_set.add((key, value, None))

                    atoms, values, cvalues = zip(*evidence_set)
                    result.add(index, atoms, values, cvalues)

                else:
                    # (No AD case) or (Inconsistent Evidence Case)
                    atoms, values, cvalues = zip(*example)
                    result.add(index, atoms, values, cvalues)
            # logger.debug(
            #     "\nProcessed Examples:\n\t"
            #     + "\n\t".join(
            #         [
            #             "Atoms: "
            #             + str(ex.atoms)
            #             + "\tValues: "
            #             + str(ex.values)
            #             + "\tContinuous Values: "
            #             + str(ex.n)
            #             for ex in result
            #         ]
            #     )
            # )
            return result
        else:
            # smarter: compile-once all examples with same atoms
            result = ExampleSet()
            for index, example in enumerate(self.examples):
                atoms, values, cvalues = zip(*example)
                result.add(index, atoms, values, cvalues)
            return result

    def _compile_examples(self):
        """Compile examples."""
        logger = getLogger("problog_lfi")
        baseprogram = DefaultEngine(**self.extra).prepare(self)
        logger.debug(
            "\nBase Program:\n\t" + baseprogram.to_prolog().replace("\n", "\n\t")
        )
        examples = self._process_examples()
        for i, example in enumerate(examples):
            logger.debug("\nCompiling example {}/{}".format(i + 1, len(examples)))
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

        random_weights = [random.random() for _ in range(0, num_random_weights + 1)]
        norm_factor = (1.0 - prior_probability - fixed_probability) / sum(
            random_weights
        )
        random_weights = [r * norm_factor for r in random_weights]

        # First argument is probability available for learnable weights in the AD.
        self.add_ad(1.0 - fixed_probability, [])  # TODO : this adds extra ad

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

        prob_args = []
        if isinstance(atom.probability, Term):
            for arg in atom.probability.args:
                if not isinstance(arg, Constant) and arg != Var("_"):
                    prob_args.append(arg)

        newcount = "_".join([str(self.count + count) for count in range(len(atoms))])

        vars = []
        for atom in atoms:
            q = list(atom.apply(ReplaceAnon()).args)
            for var in q:
                if var not in vars:
                    vars.append(var)

        # lfi_rule = Term("lfi_rule", Constant(newcount), Term("t", *prob_args, *vars))
        lfi_rule = Term("lfi_rule", Constant(newcount), *vars)
        if body is not None:
            extra_clauses.append(Clause(lfi_rule, body))

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

                atom1 = atom.apply(ReplaceAnon())

                # 1) Introduce a new LFI terms
                # lfi_fact = Term(
                #     "lfi_fact", Constant(self.count), Term("t", *prob_args, *atom1.args)
                # )
                # lfi_body = Term(
                #     "lfi_body", Constant(self.count), Term("t", *prob_args, *atom1.args)
                # )
                # lfi_par = Term(
                #     "lfi_par", Constant(self.count), Term("t", *prob_args, *atom1.args)
                # )
                # lfi_prob = Term("lfi_prob", Constant(self.count), Term("t"))
                lfi_fact = Term("lfi_fact", Constant(self.count), *atom1.args)
                lfi_body = Term("lfi_body", Constant(self.count), *atom1.args)
                lfi_par = Term("lfi_par", Constant(self.count), *atom1.args)
                lfi_prob = Term("lfi_prob", Constant(self.count), Term("t", *prob_args))

                # 2) Replacement atom
                replacement = lfi_fact.with_probability(lfi_prob)

                # 3) Create redirection clause
                extra_clauses.append(Clause(atom1.with_probability(), lfi_body))
                extra_clauses.append(Clause(lfi_body, lfi_par & lfi_fact))

                if body is None:
                    extra_clauses.append(Clause(lfi_par, Term("true")))
                else:
                    extra_clauses.append(Clause(lfi_par, lfi_rule))

                self.append_ad(len(self._weights))

                # 4) Set initial weight
                if start_value is None:
                    # Assign a random weight initially
                    start_value = random_weights.pop(-1)
                self._add_weight(start_value)

                # 5) Add name
                self.names.append(atom)
                self.bodies.append(lfi_body)
                self.parents.append(lfi_par)
                atoms_out.append(replacement)
            else:
                atoms_out.append(atom)

        self.verify_ad()

        if has_lfi_fact:
            if len(atoms) == 1:
                # Non AD
                return [atoms_out[0]] + extra_clauses
            else:
                # AD
                if body is None:
                    return [
                        AnnotatedDisjunction(atoms_out, Term("true"))
                    ] + extra_clauses
                else:
                    return [AnnotatedDisjunction(atoms_out, lfi_rule)] + extra_clauses
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

        atoms_fixed = []
        fixed_only = True
        for atom in atoms:
            if atom.probability and atom.probability.functor == "t":
                assert atom in self.names

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

        if self.output_mode is False:
            getLogger("problog_lfi").debug("\nProcessed Atoms:")
        for clause in self.source:
            if isinstance(clause, Clause):
                if clause.head.functor == "query" and clause.head.arity == 1:
                    continue
                extra_clauses = process_atom(clause.head, clause.body)
                for extra in extra_clauses:
                    if self.output_mode is False:
                        getLogger("problog_lfi").debug("\t" + str(extra))
                    yield extra
            elif isinstance(clause, AnnotatedDisjunction):
                extra_clauses = process_atom(Or.from_list(clause.heads), clause.body)
                for extra in extra_clauses:
                    if self.output_mode is False:
                        getLogger("problog_lfi").debug("\t" + str(extra))
                    yield extra
            else:
                if clause.functor == "query" and clause.arity == 1:
                    continue
                # Fact
                extra_clauses = process_atom(clause, None)
                for extra in extra_clauses:
                    if self.output_mode is False:
                        getLogger("problog_lfi").debug("\t" + str(extra))
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

        getLogger("problog_lfi").debug("Evaluating examples:")

        if self._log:
            evaluator = ExampleEvaluatorLog(self._weights, eps=self._eps)
        else:
            evaluator = ExampleEvaluator(self._weights, eps=self._eps)

        results = []
        for i, example in enumerate(self._compiled_examples):
            try:
                result = evaluator(example)
                results.append(result)
                getLogger("problog_lfi").debug(
                    "Example "
                    + str(i + 1)
                    + ":\tFrequency = "
                    + str(result[0][0])
                    + "\tp_evidence = "
                    + str(result[0][1])
                    + "\tp_queries = "
                    + str(result[0][2])
                )
            except InconsistentEvidenceError:
                # print("Ignoring example {}/{}".format(i + 1, len(self._compiled_examples)))
                getLogger("problog_lfi").warning(
                    "Ignoring example {}/{}".format(i + 1, len(self._compiled_examples))
                )

        return list(chain.from_iterable(results))

    def _update(self, results):
        """Update the current estimates based on the latest evaluation results."""
        logger = getLogger("problog_lfi")
        fact_marg = defaultdict(int)
        fact_body = defaultdict(int)
        fact_par = defaultdict(int)
        fact_count = defaultdict(int)

        score = 0.0
        for m, pEvidence, result in results:
            par_marg = dict()
            for fact, value in result.items():
                index = fact.args
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
                        if o_index >= 0 and len(index) == 1:
                            # Propositional AD
                            par_marg[(o_index,)] = value
                        elif o_index >= 0 and len(index) > 1:
                            # First Order AD
                            par_marg[(o_index, *index[1:])] = value
                fact_count[index] += m

            for index, value in par_marg.items():
                fact_par[index] += value * m
            try:
                if isinstance(pEvidence, DensityValue):
                    pEvidence = pEvidence.value()
                score += math.log(pEvidence)
            except ValueError:
                logger.debug("Pr(evidence) == 0.0")

        update_list = fact_body

        weight_changed = [False] * len(self.names)
        fact_par_grouped = dict()
        for key, value in fact_par.items():
            id = key[0]
            if id in fact_par_grouped:
                fact_par_grouped[id] += value
            else:
                fact_par_grouped[id] = value

        for index in update_list:
            if float(fact_body[index]) == 0.0:
                prob = 0.0
            else:
                prob = float(fact_body[index]) / float(fact_par_grouped[index[0]])
            logger.debug(
                "Update probabilistic fact {}: {} / {} = {}".format(
                    index, fact_body[index], fact_par_grouped[index[0]], prob
                )
            )
            self._set_weight(index[0], index[1:], prob, weight_changed=weight_changed)
            weight_changed[int(index[0])] = True

        if self._enable_normalize:
            self._normalize_weights()

        return score

    def _normalize_weights(self):
        # TODO: too late here, AD should be taken into account in _update
        # Derivation is sum(all values for var=k) / sum(all values for i sum(all values for var=i))

        for available_prob, idx in self._adatoms:
            if len(idx) == 1:
                # Not an AD; No need to normalize
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
        getLogger("problog_lfi").info("\nIteration " + str(self.iteration))
        results = self._evaluate_examples()
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

        getLogger("problog_lfi").info("Weights to learn: %s" % self.names)
        getLogger("problog_lfi").info("Bodies: %s" % self.bodies)
        getLogger("problog_lfi").info("Parents: %s" % self.parents)
        getLogger("problog_lfi").info("Initial weights: %s" % self._weights)
        delta = 1000
        prev_score = -1e10
        # TODO: isn't this comparing delta i logprob with min_improv in prob?
        while self.iteration < self.max_iter and (delta < 0 or delta > self.min_improv):
            score = self.step()
            getLogger("problog_lfi").info(
                "Weights after iteration %s: %s" % (self.iteration, self._weights)
            )
            getLogger("problog_lfi").info(
                "Score after iteration %s: %s" % (self.iteration, score)
            )
            delta = score - prev_score
            prev_score = score
        return prev_score


class ExampleSet(object):
    def __init__(self):
        self._examples = {}

    def add(self, index, atoms, values, cvalues):
        ex = self._examples.get((atoms, values))
        if ex is None:
            self._examples[(atoms, values)] = Example(index, atoms, values, cvalues)
        else:
            ex.add_index(index, cvalues)

    def __iter__(self):
        return iter(self._examples.values())

    def __len__(self):
        return len(self._examples)


class Example(object):
    def __init__(self, index, atoms, values, cvalues):
        """An example consists of a list of atoms and their corresponding values (True/False).
        Different continuous values are all mapped to True and stored in self.n.
        """
        self.atoms = tuple(atoms)
        self.values = tuple(values)
        self.compiled = []
        self.n = {tuple(cvalues): [index]}

    def __hash__(self):
        return hash((self.atoms, self.values))

    def __eq__(self, other):
        if other is None:
            return False
        return self.atoms == other.atoms and self.values == other.values

    def compile(self, lfi, baseprogram):
        logger = getLogger("problog_lfi")
        ground_program = None  # Let the grounder decide
        logger.debug("\tGrounded Atoms:\t" + str(self.atoms))
        logger.debug("\tEvidence:\t" + str(list(zip(self.atoms, self.values))))

        ground_program = ground(
            baseprogram,
            ground_program,
            evidence=list(zip(self.atoms, self.values)),
            propagate_evidence=lfi.propagate_evidence,
        )
        # logger.debug("\t" + "New ground_program:\n\t\t" + ground_program.to_prolog().replace("\n", "\n\t\t"))

        logger.debug(
            "\t"
            + "New ground_program:\n\t\t"
            + str(ground_program).replace("\n", "\n\t\t")
        )

        lfi_queries = []
        for i, node, t in ground_program:
            if (
                t == "atom"
                and isinstance(node.probability, Term)
                and node.probability.functor == "lfi_prob"
            ):
                factargs = ()
                if node.name.functor != "choice":
                    if node.name.functor == "lfi_fact":
                        factargs = node.name.args[1:]
                        # for arg in node.name.args:
                        #     if str(arg.functor) == "t":
                        #         factargs = arg.args
                    else:
                        factargs = node.name.args
                elif type(node.identifier) == tuple:
                    factargs = node.identifier[1]
                fact = Term("lfi_fact", node.probability.args[0], *factargs)
                # fact = Term("lfi_fact", node.probability.args[0], Term("t", *factargs))
                logger.debug(
                    "\tNode " + str(i) + ":\tAdding query for fact:\t" + str(fact)
                )
                ground_program.add_query(fact, i)

                tmp_body = Term("lfi_body", node.probability.args[0], *factargs)
                # tmp_body = Term(
                #     "lfi_body", node.probability.args[0], Term("t", *factargs)
                # )
                lfi_queries.append(tmp_body)
                logger.debug(
                    "\tNode "
                    + str(i)
                    + ":\tAdding query for body:\t"
                    + str(tmp_body)
                    + "\t"
                )
                tmp_par = Term("lfi_par", node.probability.args[0], *factargs)
                # tmp_par = Term(
                #     "lfi_par", node.probability.args[0], Term("t", *factargs)
                # )
                lfi_queries.append(tmp_par)
                logger.debug(
                    "\tNode "
                    + str(i)
                    + ":\tAdding query for par :\t"
                    + str(tmp_par)
                    + "\t"
                )

            elif t == "atom":
                pass

        ground_program = ground(
            baseprogram,
            ground_program,
            evidence=list(zip(self.atoms, self.values)),
            propagate_evidence=lfi.propagate_evidence,
            queries=lfi_queries,
        )
        logger.debug(
            "\t"
            + "New ground_program:\n\t\t"
            + str(ground_program).replace("\n", "\n\t\t")
        )

        self.compiled = lfi.knowledge.create_from(ground_program)
        try:
            logger.debug(
                "\tCompiled program:\n\t\t"
                + self.compiled.to_prolog().replace("\n", "\n\t\t")
            )
        except Exception:
            logger.debug(
                "\tCompiled program:\n\t\t" + str(self.compiled).replace("\n", "\n\t\t")
            )

    def add_index(self, index, cvalues):
        k = tuple(cvalues)
        if k in self.n:
            self.n[k].append(index)
        else:
            self.n[k] = [index]


class ExampleEvaluator(SemiringDensity):
    def __init__(self, weights, eps):
        SemiringDensity.__init__(self)
        self._weights = weights
        self._eps = eps

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
        if isinstance(a, Term) and a.functor == "lfi_prob":
            # index = int(a.args[0])
            return self._get_weight(*a.args)
        else:
            return float(a)

    def __call__(self, example):
        """Evaluate the model with its current estimates for all examples."""
        at = example.atoms
        val = example.values
        comp = example.compiled
        results = []
        for cval, n in example.n.items():
            results.append(self._call_internal(at, val, cval, comp, n))
        return results

    def _call_internal(self, at, val, cval, comp, n):

        evidence = {}

        for a, v, cv in zip(at, val, cval):
            if a in evidence:
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
                evidence[a] = v

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
            if name.functor not in ["lfi_body", "lfi_par"]:
                continue

            w = evaluator.evaluate(node)

            if w < 1e-6:
                p_queries[name] = 0.0
            else:
                p_queries[name] = w
        p_evidence = evaluator.evaluate_evidence()

        return len(n), p_evidence, p_queries


class ExampleEvaluatorLog(SemiringLogProbability):
    def __init__(self, weights, eps):
        SemiringLogProbability.__init__(self)
        self._weights = weights
        self._eps = eps

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

    def value(self, a):
        """Overrides from SemiringProbability.
        Replaces a weight of the form ``lfi(i, t(...))`` by its current estimated value.
        Other weights are passed through unchanged.
        :param a: term representing the weight
        :type a: Term
        :return: current weight
        :rtype: float
        """
        if isinstance(a, Term) and a.functor == "lfi_prob":
            rval = self._get_weight(*a.args)
        else:
            rval = math.log(float(a))
        return rval

    def __call__(self, example):
        """Evaluate the model with its current estimates for all examples."""
        at = example.atoms
        val = example.values
        comp = example.compiled
        results = []
        for cval, n in example.n.items():
            results.append(self._call_internal(at, val, cval, comp, n))
        return results

    def _call_internal(self, at, val, cval, comp, n):
        evidence = {}
        self._cevidence = {}
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
            p_queries[name] = w
        p_evidence = evaluator.evaluate_evidence()
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
        if vlr is None:
            vlr, vlv = str2num(vl)
        result.append((at, vlr, vlv))
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
    getLogger("problog_lfi").info("\nLearned Model:\t\n" + lfi.get_model())

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
        "-o",
        "--output",
        type=str,
        default=None,
        help="write resulting model to given file",
    )
    parser.add_argument("--logger", type=str, default=None, help="write log to file")
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
    logger = getLogger(name)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(levels[verbose])


def lfi_wrapper(plfile, evfiles, knowledge, options):
    program = PrologFile(plfile)
    examples = list(read_examples(*evfiles))
    return run_lfi(program, examples, knowledge=get_evaluatable(knowledge), **options)


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
        outf = None
    else:
        outf = open(args.output, "w")

    if args.logger is None:
        logf = None
    else:
        logf = open(args.logger, "w")

    logger = init_logger(verbose=args.verbose, name="problog_lfi", out=logf)
    create_logger("problog_lfi", args.verbose - 1)

    program = PrologFile(args.model)
    examples = list(read_examples(*args.examples))
    if len(examples) == 0:
        logger.warning("no examples specified")
    else:
        logger.info("Number of examples: %s" % len(examples))
    options = vars(args)
    del options["examples"]

    try:
        results = run_lfi(program, examples, knowledge=knowledge, **options)

        for n in results[2]:
            n.loc = program.lineno(n.location)
        retcode = result_handler((True, results), outf=outf)
    except Exception as err:
        trace = traceback.format_exc()
        getLogger("problog_lfi").error("\nError encountered:\t\n" + trace)
        err.trace = trace
        retcode = result_handler((False, err), outf=outf)

    if args.logger is not None:
        logf.close()

    if args.output is not None:
        outf.close()

    if retcode:
        sys.exit(retcode)


def print_result(d, outf, precision=8):
    success, d = d
    if success:
        score, weights, names, iterations, lfi = d
        weights = list(map(lambda x: round(x, precision), weights))
        print(score, weights, names, iterations, file=outf)
        return 0
    else:
        print(process_error(d), file=outf)
        return 1


def print_result_json(d, outf, precision=8):
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
        print(json.dumps(results), file=outf)
    else:
        results = {"SUCCESS": False, "err": vars(d)}
        print(json.dumps(results), file=outf)
    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
