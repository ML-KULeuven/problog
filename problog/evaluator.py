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
from .errors import InconsistentEvidenceError, InvalidValue, ProbLogError, InstallError

try:
    from numpy import polynomial

    pn = polynomial.polynomial
except ImportError:
    pn = None


class OperationNotSupported(ProbLogError):
    def __init__(self):
        ProbLogError.__init__(self, "This operation is not supported by this semiring")


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
        return value == self.zero()

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

    def to_evidence(self, pos_weight, neg_weight, sign):
        """
        Converts the pos. and neg. weight (internal repr.) of a literal into the case where the literal is evidence.
        Note that the literal can be a negative atom regardless of the given sign.

        :param pos_weight: The current positive weight of the literal.
        :param neg_weight: The current negative weight of the literal.
        :param sign: Denotes whether the literal or its negation is evidence. sign > 0 denotes the literal is evidence,
            otherwise its negation is evidence. Note: The literal itself can also still be a negative atom.
        :returns: A tuple of the positive and negative weight as if the literal was evidence.
            For example, for probability, returns (self.one(), self.zero()) if sign else (self.zero(), self.one())
        """
        return (self.one(), self.zero()) if sign > 0 else (self.zero(), self.one())

    def ad_negate(self, pos_weight, neg_weight):
        """
        Negation in the context of an annotated disjunction. e.g. in a probabilistic context for 0.2::a ; 0.8::b,
        the negative label for both a and b is 1.0 such that model {a,-b} = 0.2 * 1.0 and {-a,b} = 1.0 * 0.8.
        For a, pos_weight would be 0.2 and neg_weight could be 0.8. The returned value is 1.0.
        :param pos_weight: The current positive weight of the literal (e.g. 0.2 or 0.8). Internal representation.
        :param neg_weight: The current negative weight of the literal (e.g. 0.8 or 0.2). Internal representation.
        :return: neg_weight corrected based on the given pos_weight, given the ad context (e.g. 1.0). Internal
        representation.
        """
        return self.one()


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
            raise InvalidValue(
                "Not a valid value for this semiring: '%s'" % a, location=a.location
            )

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
        if not self.in_domain(a):
            raise InvalidValue("Not a valid value for this semiring: '%s'" % a)
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
                raise InvalidValue(
                    "Not a valid value for this semiring: '%s'" % a, location=a.location
                )

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


class DensityValue(object):
    def __init__(self, coefficients=(0,)):
        raise Exception("xxx")
        if pn is None:
            raise InstallError("Density calculation require the NumPy package.")
        self.coefficients = coefficients

    def __add__(self, other):
        if not isinstance(other, DensityValue):
            other = DensityValue.wrap(other)
        result = DensityValue(pn.polyadd(self.coefficients, other.coefficients))
        print("__add__", self, other, result)
        return result

    def __radd__(self, other):
        other = DensityValue.wrap(other)
        result = DensityValue(pn.polyadd(self.coefficients, other.coefficients))
        print("__radd__", self, other, result)
        return result

    def __sub__(self, other):
        result = DensityValue(pn.polysub(self.coefficients, other.coefficients))
        print("__sub__", self, other, result)
        return result

    def __rsub__(self, other):
        other = DensityValue.wrap(other)
        rval = DensityValue(pn.polysub(other.coefficients, self.coefficients))
        print("__rsub__", self, other, rval)
        return rval

    def __mul__(self, other):
        if type(other) == DensityValue:
            rval = DensityValue(pn.polymul(self.coefficients, other.coefficients))
        else:
            rval = DensityValue(pn.polymul(self.coefficients, [other]))
        print("__mul__", self, other, rval)
        return rval

    def __rmul__(self, other):
        if type(other) == DensityValue:
            rval = DensityValue(pn.polymul(self.coefficients, other.coefficients))
        else:
            rval = DensityValue(pn.polymul(self.coefficients, [other]))
        print("__rmul__", self, other, rval)
        return rval

    def value(self, x=1e-5):
        return pn.polyval([x], self.coefficients)[0]

    def __float__(self):
        return self.value()

    def __truediv__(self, other):
        return self.__div__(other)

    def __div__(self, other):
        result = None
        if type(other) == DensityValue:
            if other.is_one():
                result = self
            else:
                r = []
                for ai, zi in zip(self.coefficients, other.coefficients):
                    # Given ai <= zi
                    # TODO: Is this assumption asserted somewhere?
                    if ai > 1e-12:
                        result = DensityValue([ai / zi])
                        # TODO: Why only the first coefficient?, is this an approximation?
                        break
                if result is None and not r:
                    result = DensityValue.zero()
        else:
            result = DensityValue(pn.polymul(self.coefficients, [1.0 / other]))
        print("__div__", self, other, "->", result)
        return result

    @classmethod
    def zero(cls):
        return cls(pn.polyzero)

    @classmethod
    def one(cls):
        return cls(pn.polyone)

    def is_one(self):
        return (
            len(self.coefficients) == 1 and 1 - 1e-12 < self.coefficients[0] < 1 + 1e-12
        )

    def is_zero(self):
        return len(self.coefficients) == 1 and -1e-12 < self.coefficients[0] < 1e-12

    def is_prob(self):
        return len(self.coefficients) == 1

    @classmethod
    def wrap(cls, value):
        """Automatically wrap a value in a DensityValue object."""
        if type(value) == DensityValue:
            return value
        else:
            return DensityValue([value])

    def __repr__(self):
        return str(self.coefficients)


class SemiringDensity(Semiring):
    """A semiring for computing with densities.

    Internally, weights are represented as polynomials in the variable 'dx'.
    A discrete probability is represented as [p], a density is represented as [0, d].
    All operations are defined in terms of polynomials, except for normalization.

    """

    def __init__(self):
        # if pn is None:
        #     raise InstallError("Density calculation require the NumPy package.")
        Semiring.__init__(self)

    def one(self):
        return 1.0

    def zero(self):
        return 0.0

    def plus(self, a, b):
        return a + b

    def times(self, a, b):
        return a * b

    def is_zero(self, value):
        if isinstance(value, DensityValue):
            return value.is_zero()
        else:
            return -1e-12 < value < 1e-12
        # return DensityValue.wrap(value).is_zero()

    def is_one(self, value):
        if isinstance(value, DensityValue):
            # return value.is_one()
            return DensityValue.wrap(value).is_one()
        else:
            return 1.0 - 1e-12 < value < 1.0 + 1e-12

    def negate(self, a):
        return self.one() - a

    def result(self, a, formula=None):
        return a

    def normalize(self, a, z):
        """Normalization computes a_i / z_i and returns the lowest rank non-zero coefficient.
        """
        if -1e-12 < a < 1e-12:
            return 0.0
        return a / z
        # return DensityValue.wrap(a) / z


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
        raise NotImplementedError("Evaluatable._create_evaluator is an abstract method")

    def get_evaluator(
        self, semiring=None, evidence=None, weights=None, keep_evidence=False, **kwargs
    ):
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
                raise InconsistentEvidenceError(
                    source="evidence(" + str(ev_name) + ",false)"
                )  # true evidence is false
            elif ev_index is None and ev_value > 0:
                raise InconsistentEvidenceError(
                    source="evidence(" + str(ev_name) + ",true)"
                )  # false evidence is true
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

    def evaluate(
        self, index=None, semiring=None, evidence=None, weights=None, **kwargs
    ):
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
        raise NotImplementedError("Evaluator.propagate() is an abstract method.")

    def evaluate(self, index):
        """Compute the value of the given node."""
        raise NotImplementedError("abstract method")

    def evaluate_evidence(self):
        raise NotImplementedError("abstract method")

    def evaluate_fact(self, node):
        """Evaluate fact.

        :param node: fact to evaluate
        :return: weight of the fact (as semiring result value)
        """
        raise NotImplementedError("abstract method")

    def add_evidence(self, node):
        """Add evidence"""
        self.__evidence.append(node)

    def has_evidence(self):
        """Checks whether there is active evidence."""
        return self.__evidence != []

    def set_evidence(self, index, value):
        """Set value for evidence node.

        :param index: index of evidence node
        :param value: value of evidence. True if the evidence is positive, False otherwise.
        """
        raise NotImplementedError("abstract method")

    def set_weight(self, index, pos, neg):
        """Set weight of a node.

        :param index: index of node
        :param pos: positive weight (as semiring internal value)
        :param neg: negative weight (as semiring internal value)
        """
        raise NotImplementedError("abstract method")

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
        self._computed_smooth = {}
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
        self._computed_smooth.clear()
        self._fact_weights = weights

    def update_weights(self, weights):
        """Update weights to given known weights.

        :param weights: dictionary of weights
        :return:
        """
        self._computed_weights.clear()
        self._computed_smooth.clear()
        self._fact_weights.update(weights)

    def get_weight(self, index, smooth=None):
        """Get the weight of the node with the given index.

        :param index: integer or formula.TRUE or formula.FALSE
        :return: weight of the node
        """

        if index == self.formula.TRUE:
            return self.semiring.one()
        elif index == self.formula.FALSE:
            return self.semiring.zero()
        elif index < 0:
            weight = self._fact_weights.get(abs(index))
            if weight is None:
                # This will only work if the semiring support negation!
                nw = self.get_weight(-index, smooth=smooth)
                return self.semiring.negate(nw)
            else:
                self._computed_smooth[index] = {abs(index)}
                return weight[1]
        else:
            weight = self._fact_weights.get(index)
            if weight is None:
                weight = self._computed_weights.get(index)
                if weight is None:
                    weight = self.compute_weight(index, smooth)
                    self._computed_weights[index] = weight
                return weight
            else:
                self._computed_smooth[index] = {index}
                return weight[0]

    def propagate(self):
        self._fact_weights = self.formula.extract_weights(
            self.semiring, self._fact_weights
        )

    def evaluate(self, index, smooth=None):
        return self.semiring.result(self.get_weight(index, smooth=smooth), self.formula)

    def compute_weight(self, index, smooth=None):
        """Compute the weight of the node with the given index.

        :param index: integer or formula.TRUE or formula.FALSE
        :return: weight of the node
        """
        # print('handle smooth', smooth)
        if smooth is None:
            raise Exception()

        if index == self.formula.TRUE:
            return self.semiring.one()
        elif index == self.formula.FALSE:
            return self.semiring.zero()
        else:
            node = self.formula.get_node(abs(index))
            ntype = type(node).__name__

            if ntype == "atom":
                self._computed_smooth[index] = {index}
                return self.semiring.one()
            else:
                childprobs = [self.get_weight(c, smooth=smooth) for c in node.children]
                all_vars = set()
                for c in node.children:
                    if c not in self._computed_smooth:
                        raise Exception(
                            "Smoothing expected node {} to be present".format(c)
                        )
                    all_vars.update(self._computed_smooth[c])
                # print('all_vars', all_vars)
                self._computed_smooth[index] = all_vars
                if ntype == "conj":
                    p = self.semiring.one()
                    for c in childprobs:
                        p = self.semiring.times(p, c)
                    return p
                elif ntype == "disj":
                    p = self.semiring.zero()
                    for c, ci in zip(childprobs, node.children):
                        if smooth:
                            diff_vars = all_vars.difference(self._computed_smooth[ci])
                            # print('diff_vars', diff_vars)
                            for diff_var in diff_vars:
                                vt = self.get_weight(diff_var)
                                vn = self.semiring.negate(vt)
                                vd = self.semiring.plus(vt, vn)
                                print("smoothing", vt, vn, vd)
                                c = self.semiring.times(c, vd)
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
        :return: weight of the node and the set of abs(literals) involved
        """

        if index == self.formula.TRUE:
            return self.semiring.one(), set()
        elif index == self.formula.FALSE:
            return self.semiring.zero(), set()
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

        if ntype == "atom":
            return self.semiring.one(), {index}
        else:
            childprobs = [self.get_weight(c) for c in node.children]
            if ntype == "conj":
                p = self.semiring.one()
                all_used = set()
                for cp, cu in childprobs:
                    all_used |= cu
                    p = self.semiring.times(p, cp)
                return p, all_used
            elif ntype == "disj":
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
