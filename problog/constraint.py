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


class Constraint(object):
    """A propositional constraint."""

    def getNodes(self):
        """Get all nodes involved in this constraint."""
        return NotImplemented('Constraint.getNodes() is an abstract method.')

    def updateWeights(self, weights, semiring):
        # Typically, constraints don't update weights
        pass


class ConstraintAD(Constraint):
    """Annotated disjunction constraint (mutually exclusive with weight update)."""

    def __init__(self, group):
        self.nodes = set()
        self.group = group
        self.extra_node = None

    def __str__(self):
        return 'annotated_disjunction(%s, %s)' % (list(self.nodes), self.extra_node)

    def getNodes(self):
        if self.extra_node:
            return list(self.nodes) + [self.extra_node]
        else :
            return self.nodes

    def isTrue(self):
        return len(self.nodes) <= 1

    def isFalse(self):
        return False

    def isActive(self):
        return not self.isTrue() and not self.isFalse()

    def add(self, node, formula):
        if formula.has_evidence_values():
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

        self.nodes.add(node)
        if len(self.nodes) > 1 and self.extra_node is None:
            # If there are two or more choices -> add extra choice node
            self.updateLogic(formula)
        return node

    def encodeCNF(self):
        if self.isActive():
            nodes = list(self.nodes) + [self.extra_node]
            lines = []
            for i, n in enumerate(nodes):
                for m in nodes[i+1:]:
                    lines.append((-n, -m))    # mutually exclusive
            lines.append(nodes)   # pick one
            return lines
        else:
            return []

    def updateLogic(self, formula):
        """Add extra information to the logic structure of the formula."""

        if self.isActive():
            self.extra_node = formula.add_atom(('%s_extra' % (self.group,)), True, None)
            # formula.addConstraintOnNode(self, self.extra_node)

    def updateWeights(self, weights, semiring):
        """Update the weights of the logic formula accordingly."""
        if self.isActive():
            s = semiring.zero()
            for n in self.nodes:
                pos, neg = weights.get(n, (semiring.one(), semiring.one()))
                weights[n] = (pos, semiring.one())
                s = semiring.plus(s, pos)
            complement = semiring.negate(s)
            weights[self.extra_node] = (complement, semiring.one())

    def copy(self, rename={}):
        result = ConstraintAD(self.group)
        result.nodes = set(rename.get(x, x) for x in self.nodes)
        result.extra_node = rename.get(self.extra_node, self.extra_node)
        return result


class TrueConstraint(Constraint):
    """A constraint specifying that a given node should be true."""

    def __init__(self, node):
        self.node = node

    def isActive(self):
        return True

    def encodeCNF(self):
        return [[self.node]]

    def copy(self, rename={}):
        return TrueConstraint(rename.get(self.node, self.node))

    def __str__(self):
        return '%s is true' % self.node


class ClauseConstraint(Constraint):
    """A constraint specifying that a given clause should be true."""

    def __init__(self, nodes):
        self.nodes = nodes

    def isActive(self):
        return True

    def encodeCNF(self):
        return [self.nodes]

    def copy(self, rename={}):
        return ClauseConstraint(map(lambda x: rename.get(x, x), self.nodes))

    def __str__(self):
        return '%s is true' % self.nodes
