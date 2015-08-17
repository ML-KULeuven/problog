"""
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

from collections import defaultdict, OrderedDict

from .core import ProbLogObject

class BaseFormula(ProbLogObject):
    """Defines a basic logic formula consisting of nodes in some logical relation.

        Each node is represented by a key. This key has the following properties:
         - None indicates false
         - 0 indicates true
         - a number larger than 0 indicates a positive node
         - the key -a with a a number larger than 0 indicates the negation of a

        This data structure also support weights on nodes, names on nodes and constraints.
    """

    # Define special keys
    TRUE = 0
    FALSE = None

    WEIGHT_NEUTRAL = True
    WEIGHT_NO = None

    LABEL_QUERY = "query"
    LABEL_EVIDENCE_POS = "evidence+"
    LABEL_EVIDENCE_NEG = "evidence-"
    LABEL_EVIDENCE_MAYBE = "evidence?"
    LABEL_NAMED = "named"

    def __init__(self):
        self._weights = {}               # Node weights: dict(key: Term)

        self._constraints = []           # Constraints: list of Constraint

        self._names = defaultdict(OrderedDict)  # Node names: dict(label: dict(key, Term))

        self._atomcount = 0

    @property
    def atomcount(self):
        return self._atomcount

    # ====================================================================================== #
    # ==========                          NODE WEIGHTS                           =========== #
    # ====================================================================================== #

    def get_weights(self):
        return self._weights

    def set_weights(self, weights):
        self._weights = weights

    def get_weight(self, key, semiring):
        if self.is_false(key):
            return semiring.zero()
        elif self.is_true(key):
            return semiring.one()
        elif key < 0:
            return semiring.neg_value(self._weights[-key])
        else:
            return semiring.pos_value(self._weights[key])

    def set_weight(self, key, weight):
        self._weights[key] = weight

    def extract_weights(self, semiring, weights=None):
        """Extracts the positive and negative weights for all atoms in the data structure.

            :param semiring: semiring that determines the interpretation of the weights
            :param weights: dictionary of { node name : weight } that overrides the builtin weights
            :returns: dictionary { key: (positive weight, negative weight) }

            Atoms with weight set to neutral will get weight ``(semiring.one(), semiring.one())``.

            All constraints are applied to the weights.
        """

        if weights is None:
            weights = self.get_weights()
        else:
            weights = {self.get_node_by_name(n): v for n, v in weights.items()}

        result = {}

        for n, w in weights.items():
            if w == self.WEIGHT_NEUTRAL:
                result[n] = semiring.one(), semiring.one()
            else:
                result[n] = semiring.pos_value(w), semiring.neg_value(w)

        for c in self.constraints():
            c.updateWeights(result, semiring)

        return result

    # ====================================================================================== #
    # ==========                           NODE NAMES                            =========== #
    # ====================================================================================== #

    def add_name(self, name, key, label=None):
        if label is None:
            label = self.LABEL_NAMED
        self._names[label][name] = key

    def get_node_by_name(self, name):
        for names in self._names.values():
            res = names.get(name, '#NOTFOUND#')
            if res != '#NOTFOUND#':
                return res
        raise KeyError(name)

    def add_query(self, name, key):
        self.add_name(name, key, self.LABEL_QUERY)

    def add_evidence(self, name, key, value):
        if value is None:
            self.add_name(name, key, self.LABEL_EVIDENCE_MAYBE)
        elif value:
            self.add_name(name, key, self.LABEL_EVIDENCE_POS)
        else:
            self.add_name(name, key, self.LABEL_EVIDENCE_NEG)

    def get_names(self, label):
        return self._names.get(label, {}).items()

    def get_names_with_label(self):
        result = []
        for label in self._names:
            for name, key in self._names[label].items():
                result.append((name, key, label))
        return result

    def queries(self):
        return self.get_names(self.LABEL_QUERY)

    def evidence(self):
        evidence_true = self.get_names(self.LABEL_EVIDENCE_POS)
        evidence_false = self.get_names(self.LABEL_EVIDENCE_NEG)
        return list(evidence_true) + [(name, self.negate(node)) for name, node in evidence_false]

    # ====================================================================================== #
    # ==========                        KEY MANIPULATION                         =========== #
    # ====================================================================================== #

    def is_true(self, key):
        return key == self.TRUE

    def is_false(self, key):
        return key == self.FALSE

    def is_probabilistic(self, key):
        return not self.is_true(key) and not self.is_false(key)

    def negate(self, key):
        if key == self.TRUE:
            return self.FALSE
        elif key == self.FALSE:
            return self.TRUE
        else:
            return -key

    # ====================================================================================== #
    # ==========                          CONSTRAINTS                            =========== #
    # ====================================================================================== #

    def constraints(self):
        return self._constraints

    def add_constraint(self, constraint):
        self._constraints.append(constraint)
