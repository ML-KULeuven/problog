"""
problog.constraint - Propositional constraints
----------------------------------------------

Data structures for specifying propositional constraints.

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

from .errors import InvalidValue
from .logic import Term, Constant


class Constraint(object):
    """A propositional constraint."""

    def get_nodes(self):
        """Get all nodes involved in this constraint."""
        return NotImplementedError('abstract method')

    def update_weights(self, weights, semiring):
        """Update the weights in the given dictionary according to the constraints.

        :param weights: dictionary of weights (see result of :func:`LogicFormula.extract_weights`)
        :param semiring: semiring to use for weight transformation
        """
        # Typically, constraints don't update weights
        pass

    def is_true(self):
        """Checks whether the constraint is trivially true."""
        return False

    def is_false(self):
        """Checks whether the constraint is trivially false."""
        return False

    def is_nontrivial(self):
        """Checks whether the constraint is non-trivial."""
        return not self.is_true() and not self.is_false()

    def as_clauses(self):
        """Represent the constraint as a list of clauses (CNF form).

        :return: list of clauses where each clause is represent as a list of node keys
        :rtype: list[list[int]]
        """
        return NotImplementedError('abstract method')

    def copy(self, rename=None):
        """Copy this constraint while applying the given node renaming.

        :param rename: node rename map (or None if no rename is required)
        :return: copy of the current constraint
        """
        return NotImplementedError('abstract method')


class ConstraintAD(Constraint):
    """Annotated disjunction constraint (mutually exclusive with weight update)."""

    def __init__(self, group):
        self.nodes = set()
        self.group = group
        self.extra_node = None

    def __str__(self):
        return 'annotated_disjunction(%s, %s)' % (list(self.nodes), self.extra_node)

    def get_nodes(self):
        if self.extra_node:
            return list(self.nodes) + [self.extra_node]
        else:
            return self.nodes

    def is_true(self):
        return len(self.nodes) <= 1

    def is_false(self):
        return False

    def add(self, node, formula):
        """Add a node to the constraint from the given formula.

        :param node: node to add
        :param formula: formula from which the node is taken
        :return: value of the node after constraint propagation
        """

        if node in self.nodes:
            return node

        is_extra = formula.get_node(node).probability == formula.WEIGHT_NEUTRAL

        if formula.has_evidence_values() and not is_extra:
            # Propagate constraint: if one of the other nodes is True: this one is false
            for n in self.nodes:
                if formula.get_evidence_value(n) == formula.TRUE:
                    return formula.FALSE
            if formula.get_evidence_value(node) == formula.FALSE:
                return node
            elif formula.get_evidence_value(node) == formula.TRUE:
                for n in self.nodes:
                    formula.set_evidence_value(n, formula.FALSE)

            if formula.semiring:
                sr = formula.semiring
                w = formula.get_weight(node, sr)
                for n in self.nodes:
                    w = sr.plus(w, formula.get_weight(n, sr))
                if sr.is_one(w):
                    unknown = None
                    if formula.get_evidence_value(node) != formula.FALSE:
                        unknown = node
                    for n in self.nodes:
                        if formula.get_evidence_value(n) != formula.FALSE:
                            if unknown is not None:
                                unknown = None
                                break
                            else:
                                unknown = n
                    if unknown is not None:
                        formula.set_evidence_value(unknown, formula.TRUE)

        if is_extra:
            self.extra_node = node
        else:
            self.nodes.add(node)

        if len(self.nodes) > 1 and self.extra_node is None:
            # If there are two or more choices -> add extra choice node
            self._update_logic(formula)
        return node

    def as_clauses(self):
        if self.is_nontrivial():
            nodes = list(self.nodes) + [self.extra_node]
            lines = []
            for i, n in enumerate(nodes):
                for m in nodes[i + 1:]:
                    lines.append((-n, -m))    # mutually exclusive
            lines.append(nodes)   # pick one
            return lines
        else:
            return []

    def _update_logic(self, formula):
        """Add extra information to the logic structure of the formula.

        :param formula: formula to update
        """
        if self.is_nontrivial():
            name = Term('choice', Constant(self.group[0]), Term('e'), Term('null'), *self.group[1])
            self.extra_node = formula.add_atom(('%s_extra' % (self.group,)), True, name=name, group=self.group)
            # formula.addConstraintOnNode(self, self.extra_node)

    def update_weights(self, weights, semiring):
        if self.is_nontrivial():
            s = semiring.zero()
            ws = []
            for n in self.nodes:
                pos, neg = weights.get(n, (semiring.one(), semiring.one()))
                weights[n] = (pos, semiring.one())
                ws.append(pos)
            if not semiring.in_domain(s):
                raise InvalidValue('Sum of annotated disjunction weigths exceed acceptable value')
                # TODO add location

            name = Term('choice', Constant(self.group[0]), Term('e'), Term('null'), *self.group[1])
            complement = semiring.ad_complement(ws, key=name)
            weights[self.extra_node] = (complement, semiring.one())

    def copy(self, rename=None):
        if rename is None:
            rename = {}
        result = ConstraintAD(self.group)
        result.nodes = set(rename.get(x, x) for x in self.nodes)
        result.extra_node = rename.get(self.extra_node, self.extra_node)
        return result

    def check(self, values):
        """Check the constraint

        :param values: dictionary of values for nodes
        :return: True if constraint succeeds, False otherwise
        """
        if self.is_true():
            return True
        elif self.is_false():
            return False
        else:
            actual = [values.get(i) for i in self.nodes if values.get(i) is not None]
            return sum(actual) == 1


class ClauseConstraint(Constraint):
    """A constraint specifying that a given clause should be true."""

    def __init__(self, nodes):
        self.nodes = nodes

    def as_clauses(self):
        return [self.nodes]

    def copy(self, rename=None):
        if rename is None:
            rename = {}
        return ClauseConstraint(map(lambda x: rename.get(x, x), self.nodes))

    def __str__(self):
        return '%s is true' % self.nodes


class TrueConstraint(Constraint):
    """A constraint specifying that a given node should be true."""

    def __init__(self, node):
        self.node = node

    def get_nodes(self):
        return [self.node]

    def as_clauses(self):
        return [[self.node]]

    def copy(self, rename=None):
        if rename is None:
            rename = {}
        return TrueConstraint(rename.get(self.node, self.node))

    def __str__(self):
        return '%s is true' % self.node
