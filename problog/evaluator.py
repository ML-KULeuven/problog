"""
problog.evaluator - Commone interface for evaluation
----------------------------------------------------

Provides common interface for evaluation of weighted logic formulas.

..
    Part of the ProbLog distribution.

    Copyright 2015 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
from __future__ import print_function

import math

from .core import ProbLogObject, transform_allow_subclass
from .errors import InconsistentEvidenceError, InvalidValue


class OperationNotSupported(Exception):

    def __init__(self):
        Exception.__init__(self, 'This operation is not supported by this semiring.')


class Semiring(object):
    """Interface for weight manipulation.

    A semiring is a set R equipped with two binary operations '+' and 'x'.

    The semiring can use different representations for internal values and external values.
    For example, the LogProbability semiring uses probabilities [0, 1] as external values and uses \
     the logarithm of these probabilities as internal values.

    Most methods take and return internal values. The execeptions are:

       - value, pos_value, neg_value: transform an external value to an internal value
       - result: transform an internal to an external value
       - result_zero, result_one: return an external value

    """

    def one(self):
        """Returns the identity element of the multiplication."""
        raise NotImplementedError()

    def is_one(self, value):
        """Tests whether the given value is the identity element of the multiplication."""
        return value == self.one

    def zero(self):
        """Returns the identity element of the addition."""
        raise NotImplementedError()

    def is_zero(self, value):
        """Tests whether the given value is the identity element of the addition."""
        return value == self.zero

    def plus(self, a, b):
        """Computes the addition of the given values."""
        raise NotImplementedError()

    def times(self, a, b):
        """Computes the multiplication of the given values."""
        raise NotImplementedError()

    def negate(self, a):
        """Returns the negation. This operation is optional.
        For example, for probabilities return 1-a.

        :raise OperationNotSupported: if the semiring does not support this operation
        """
        raise OperationNotSupported()

    def value(self, a):
        """Transform the given external value into an internal value."""
        return float(a)

    def result(self, a, formula=None):
        """Transform the given internal value into an external value."""
        return a

    def normalize(self, a, z):
        """Normalizes the given value with the given normalization constant.

        For example, for probabilities, returns a/z.

        :raise OperationNotSupported: if z is not one and the semiring does not support \
         this operation
        """
        if self.is_one(z):
            return a
        else:
            raise OperationNotSupported()

    def pos_value(self, a, key=None):
        """Extract the positive internal value for the given external value."""
        return self.value(a)

    def neg_value(self, a, key=None):
        """Extract the negative internal value for the given external value."""
        return self.negate(self.value(a))

    def result_zero(self):
        """Give the external representation of the identity element of the addition."""
        return self.result(self.zero())

    def result_one(self):
        """Give the external representation of the identity element of the multiplication."""
        return self.result(self.one())

    def is_dsp(self):
        """Indicates whether this semiring requires solving a disjoint sum problem."""
        return False

    def is_nsp(self):
        """Indicates whether this semiring requires solving a neutral sum problem."""
        return False

    def in_domain(self, a):
        """Checks whether the given (internal) value is valid."""
        return True

    def ad_complement(self, ws, key=None):
        s = self.zero()
        for w in ws:
            s = self.plus(s, w)
        return self.negate(s)

    def true(self, key=None):
        """Handle weight for deterministically true."""
        return self.one(), self.zero()

    def false(self, key=None):
        """Handle weight for deterministically false."""
        return self.zero(), self.one()


class SemiringProbability(Semiring):
    """Implementation of the semiring interface for probabilities."""

    def one(self):
        return 1.0

    def zero(self):
        return 0.0

    def is_one(self, value):
        return 1.0 - 1e-12 < value < 1.0 + 1e-12

    def is_zero(self, value):
        return -1e-12 < value < 1e-12

    def plus(self, a, b):
        return a + b

    def times(self, a, b):
        return a * b

    def negate(self, a):
        return 1.0 - a

    def normalize(self, a, z):
        return a / z

    def value(self, a):
        v = float(a)
        if 0.0 - 1e-9 <= v <= 1.0 + 1e-9:
            return v
        else:
            raise InvalidValue("Not a valid value for this semiring: '%s'" % a, location=a.location)

    def is_dsp(self):
        """Indicates whether this semiring requires solving a disjoint sum problem."""
        return True

    def in_domain(self, a):
        return 0.0 - 1e-9 <= a <= 1.0 + 1e-9


class SemiringLogProbability(SemiringProbability):
    """Implementation of the semiring interface for probabilities with logspace calculations."""

    inf, ninf = float("inf"), float("-inf")

    def one(self):
        return 0.0

    def zero(self):
        return self.ninf

    def is_zero(self, value):
        return value <= -1e100

    def is_one(self, value):
        return -1e-12 < value < 1e-12

    def plus(self, a, b):
        if a < b:
            if a == self.ninf:
                return b
            return b + math.log1p(math.exp(a - b))
        else:
            if b == self.ninf:
                return a
            return a + math.log1p(math.exp(b - a))

    def times(self, a, b):
        return a + b

    def negate(self, a):
        if a > -1e-10:
            return self.zero()
        return math.log1p(-math.exp(a))

    def value(self, a):
        v = float(a)
        if -1e-9 <= v < 1e-9:
            return self.zero()
        else:
            if 0.0 - 1e-9 <= v <= 1.0 + 1e-9:
                return math.log(v)
            else:
                raise InvalidValue("Not a valid value for this semiring: '%s'" % a, location=a.location)

    def result(self, a, formula=None):
        return math.exp(a)

    def normalize(self, a, z):
        # Assumes Z is in log
        return a - z

    def is_dsp(self):
        """Indicates whether this semiring requires solving a disjoint sum problem."""
        return True

    def in_domain(self, a):
        return a <= 1e-12


class SemiringSymbolic(Semiring):
    """Implementation of the semiring interface for probabilities using symbolic calculations."""

    def one(self):
        return "1"

    def zero(self):
        return "0"

    def plus(self, a, b):
        if a == "0":
            return b
        elif b == "0":
            return a
        else:
            return "(%s + %s)" % (a, b)

    def times(self, a, b):
        if a == "0" or b == "0":
            return "0"
        elif a == "1":
            return b
        elif b == "1":
            return a
        else:
            return "%s*%s" % (a, b)

    def negate(self, a):
        if a == "0":
            return "1"
        elif a == "1":
            return "0"
        else:
            return "(1-%s)" % a

    def value(self, a):
        return str(a)

    def normalize(self, a, z):
        if z == "1":
            return a
        else:
            return "%s / %s" % (a, z)

    def is_dsp(self):
        """Indicates whether this semiring requires solving a disjoint sum problem."""
        return True


class Evaluatable(ProbLogObject):

    def evidence_all(self):
        raise NotImplementedError()

    def _create_evaluator(self, semiring, weights, **kwargs):
        """Create a new evaluator.

        :param semiring: semiring to use
        :param weights: weights to use (replace weights defined in formula)
        :return: evaluator
        :rtype: Evaluator
        """
        raise NotImplementedError('Evaluatable._create_evaluator is an abstract method')

    def get_evaluator(self, semiring=None, evidence=None, weights=None, keep_evidence=False, **kwargs):
        """Get an evaluator for computing queries on this formula.
        It creates an new evaluator and initializes it with the given or predefined evidence.

        :param semiring: semiring to use
        :param evidence: evidence values (override values defined in formula)
        :type evidence: dict(Term, bool)
        :param weights: weights to use
        :return: evaluator for this formula
        """
        if semiring is None:
            semiring = SemiringLogProbability()

        evaluator = self._create_evaluator(semiring, weights, **kwargs)

        for ev_name, ev_index, ev_value in self.evidence_all():
            if ev_index == 0 and ev_value > 0:
                pass  # true evidence is deterministically true
            elif ev_index is None and ev_value < 0:
                pass  # false evidence is deterministically false
            elif ev_index == 0 and ev_value < 0:
                raise InconsistentEvidenceError(source='evidence('+str(ev_name)+',false)')  # true evidence is false
            elif ev_index is None and ev_value > 0:
                raise InconsistentEvidenceError(source='evidence('+str(ev_name)+',true)')  # false evidence is true
            elif evidence is None and ev_value != 0:
                evaluator.add_evidence(ev_value * ev_index)
            elif evidence is not None:
                try:
                    value = evidence[ev_name]
                    if value is None:
                        pass
                    elif value:
                        evaluator.add_evidence(ev_index)
                    else:
                        evaluator.add_evidence(-ev_index)
                except KeyError:
                    if keep_evidence:
                        evaluator.add_evidence(ev_value * ev_index)

        evaluator.propagate()
        return evaluator

    def evaluate(self, index=None, semiring=None, evidence=None, weights=None, **kwargs):
        """Evaluate a set of nodes.

        :param index: node to evaluate (default: all queries)
        :param semiring: use the given semiring
        :param evidence: use the given evidence values (overrides formula)
        :param weights: use the given weights (overrides formula)
        :return: The result of the evaluation expressed as an external value of the semiring. \
         If index is ``None`` (all queries) then the result is a dictionary of name to value.
        """
        evaluator = self.get_evaluator(semiring, evidence, weights, **kwargs)

        if index is None:
            result = {}
            # Probability of query given evidence

            # interrupted = False
            for name, node, label in evaluator.formula.labeled():
                w = evaluator.evaluate(node)
                result[name] = w
            return result
        else:
            return evaluator.evaluate(index)


@transform_allow_subclass
class EvaluatableDSP(Evaluatable):
    """Interface for evaluatable formulae."""

    def __init__(self):
        Evaluatable.__init__(self)


class Evaluator(object):
    """Generic evaluator."""

    # noinspection PyUnusedLocal
    def __init__(self, formula, semiring, weights, **kwargs):
        self.formula = formula
        self.weights = {}
        self.given_weights = weights

        self.__semiring = semiring

        self.__evidence = []

    @property
    def semiring(self):
        """Semiring used by this evaluator."""
        return self.__semiring

    def propagate(self):
        """Propagate changes in weight or evidence values."""
        raise NotImplementedError('Evaluator.propagate() is an abstract method.')

    def evaluate(self, index):
        """Compute the value of the given node."""
        raise NotImplementedError('abstract method')

    def evaluate_evidence(self):
        raise NotImplementedError('abstract method')

    def evaluate_fact(self, node):
        """Evaluate fact.

        :param node: fact to evaluate
        :return: weight of the fact (as semiring result value)
        """
        raise NotImplementedError('abstract method')

    def add_evidence(self, node):
        """Add evidence"""
        self.__evidence.append(node)

    def has_evidence(self):
        """Checks whether there is active evidence."""
        return self.__evidence != []

    def set_evidence(self, index, value):
        """Set value for evidence node.

        :param index: index of evidence node
        :param value: value of evidence
        """
        raise NotImplementedError('abstract method')

    def set_weight(self, index, pos, neg):
        """Set weight of a node.

        :param index: index of node
        :param pos: positive weight (as semiring internal value)
        :param neg: negative weight (as semiring internal value)
        """
        raise NotImplementedError('abstract method')

    def clear_evidence(self):
        """Clear all evidence."""
        self.__evidence = []

    def evidence(self):
        """Iterate over evidence."""
        return iter(self.__evidence)


class FormulaEvaluator(object):
    """Standard evaluator for boolean formula."""

    def __init__(self, formula, semiring, weights=None):
        self._computed_weights = {}
        self._semiring = semiring
        self._formula = formula
        self._fact_weights = {}
        if weights is not None:
            self.set_weights(weights)

    @property
    def semiring(self):
        return self._semiring

    @property
    def formula(self):
        return self._formula

    def set_weights(self, weights):
        """Set known weights.

        :param weights: dictionary of weights
        :return:
        """
        self._computed_weights.clear()
        self._fact_weights = weights

    def get_weight(self, index):
        """Get the weight of the node with the given index.

        :param index: integer or formula.TRUE or formula.FALSE
        :return: weight of the node
        """

        if index == self.formula.TRUE:
            return self.semiring.one()
        elif index == self.formula.FALSE:
            return self.semiring.zero()
        elif index < 0:
            weight = self._fact_weights.get(index)
            if weight is None:
                # This will only work if the semiring support negation!
                nw = self.get_weight(-index)
                return self.semiring.negate(nw)
            else:
                return weight[1]
        else:
            weight = self._fact_weights.get(index)
            if weight is None:
                weight = self._computed_weights.get(index)
                if weight is None:
                    weight = self.compute_weight(index)
                    self._computed_weights[index] = weight
                return weight
            else:
                return weight[0]

    def propagate(self):
        self._fact_weights = self.formula.extract_weights(self.semiring)

    def evaluate(self, index):
        return self.semiring.result(self.get_weight(index), self.formula)

    def compute_weight(self, index):
        """Compute the weight of the node with the given index.

        :param index: integer or formula.TRUE or formula.FALSE
        :return: weight of the node
        """

        if index == self.formula.TRUE:
            return self.semiring.one()
        elif index == self.formula.FALSE:
            return self.semiring.zero()
        else:
            node = self.formula.get_node(abs(index))
            ntype = type(node).__name__

            if ntype == 'atom':
                return self.semiring.one()
            else:
                childprobs = [self.get_weight(c) for c in node.children]
                if ntype == 'conj':
                    p = self.semiring.one()
                    for c in childprobs:
                        p = self.semiring.times(p, c)
                    return p
                elif ntype == 'disj':
                    p = self.semiring.zero()
                    for c in childprobs:
                        p = self.semiring.plus(p, c)
                    return p
                else:
                    raise TypeError("Unexpected node type: '%s'." % ntype)


class FormulaEvaluatorNSP(FormulaEvaluator):
    """Evaluator for boolean formula that addresses the Neutral Sum Problem."""

    def __init__(self, formula, semiring, weights=None):
        FormulaEvaluator.__init__(self, formula, semiring, weights)

    def get_weight(self, index):
        """Get the weight of the node with the given index.

        :param index: integer or formula.TRUE or formula.FALSE
        :return: weight of the node
        """

        if index == self.formula.TRUE:
            return self.semiring.one()
        elif index == self.formula.FALSE:
            return self.semiring.zero()
        elif index < 0:
            weight = self._fact_weights.get(-index)
            if weight is None:
                # This will only work if the semiring support negation!
                nw, nu = self.get_weight(-index)
                return self.semiring.negate(nw), nu
            else:
                return weight[1], {abs(index)}
        else:
            weight = self._fact_weights.get(index)
            if weight is None:
                weight = self._computed_weights.get(index)
                if weight is None:
                    weight = self.compute_weight(index)
                    self._computed_weights[index] = weight
                return weight
            else:
                return weight[0], {abs(index)}

    def evaluate(self, index):
        cp, cu = self.get_weight(index)
        all_used = set(self._fact_weights.keys())

        not_used = all_used - cu
        for nu in not_used:
            nu_p, a = self.get_weight(nu)
            nu_n, b = self.get_weight(-nu)
            cp = self.semiring.times(cp, self.semiring.plus(nu_p, nu_n))

        return self.semiring.result(cp, self.formula)

    def compute_weight(self, index):
        """Compute the weight of the node with the given index.

        :param index: integer or formula.TRUE or formula.FALSE
        :return: weight of the node
        """

        node = self.formula.get_node(index)
        ntype = type(node).__name__

        if ntype == 'atom':
            return self.semiring.one(), {index}
        else:
            childprobs = [self.get_weight(c) for c in node.children]
            if ntype == 'conj':
                p = self.semiring.one()
                all_used = set()
                for cp, cu in childprobs:
                    all_used |= cu
                    p = self.semiring.times(p, cp)
                return p, all_used
            elif ntype == 'disj':
                p = self.semiring.zero()
                all_used = set()
                for cp, cu in childprobs:
                    all_used |= cu

                for cp, cu in childprobs:
                    not_used = all_used - cu
                    for nu in not_used:
                        nu_p, u = self.get_weight(nu)
                        nu_n, u = self.get_weight(-nu)
                        cp = self.semiring.times(cp, self.semiring.plus(nu_p, nu_n))
                    p = self.semiring.plus(p, cp)
                return p, all_used
            else:
                raise TypeError("Unexpected node type: '%s'." % ntype)
