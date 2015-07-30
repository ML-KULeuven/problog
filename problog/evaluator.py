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

from collections import defaultdict

import subprocess
import sys, os, tempfile
import math
from .core import InconsistentEvidenceError

class Semiring(object) :

    def one(self) :
        raise NotImplementedError()

    def zero(self) :
        raise NotImplementedError()

    def is_one(self, value):
        return value == self.one

    def is_zero(self, value):
        return value == self.zero

    def plus(self, a, b) :
        raise NotImplementedError()

    def times(self, a, b) :
        raise NotImplementedError()

    def negate(self, a) :
        raise NotImplementedError()

    def value(self, a) :
        raise NotImplementedError()

    def result(self, a) :
        raise NotImplementedError()

    def normalize(self, a, Z) :
        raise NotImplementedError()

    def isLogspace(self) :
        return False

    def pos_value(self, a) :
        return self.value(a)

    def neg_value(self, a) :
        return self.negate(self.value(a))

class SemiringProbability(Semiring) :

    def one(self) :
        return 1.0

    def zero(self) :
        return 0.0

    def is_one(self, value):
        return 1.0-1e-16 < value < 1.0+1e-16

    def is_zero(self, value):
        return -1e-16 < value < 1e-16

    def plus(self, a, b) :
        return a + b

    def times(self, a, b) :
        return a * b

    def negate(self, a) :
        return 1.0 - a

    def value(self, a) :
        return float(a)

    def result(self, a) :
        return a

    def normalize(self, a, Z) :
        return a/Z


class SemiringLogProbability(SemiringProbability) :
    inf, ninf = float("inf"), float("-inf")

    def one(self) :
        return 0.0

    def zero(self) :
        return self.ninf

    def is_zero(self, value):
        return value <= -1e100

    def is_one(self, value):
        return -1e-16 < value < 1e-16

    def plus(self, a, b) :
        if a < b:
            if a == self.ninf: return b
            return b + math.log1p(math.exp(a - b))
        else:
            if b == self.ninf: return a
            return a + math.log1p(math.exp(b - a))

    def times(self, a, b) :
        return a + b

    def negate(self, a) :
        if a > -1e-10: return self.zero()
        return math.log1p(-math.exp(a))

    def value(self, a) :
        if float(a) < 1e-10 :
            return self.zero()
        else :
            return math.log(float(a))

    def result(self, a) :
        return math.exp(a)

    def normalize(self, a, Z) :
        # Assumes Z is in log
        return a - Z

    def isLogspace(self) :
        return True


class SemiringSymbolic(Semiring) :

    def one(self) :
        return "1"

    def zero(self) :
        return "0"

    def plus(self, a, b) :
        if a == "0" :
            return b
        elif b == "0" :
            return a
        else :
            return "(%s + %s)" % (a,b)

    def times(self, a, b) :
        if a == "0" or b == "0" :
            return "0"
        elif a == "1" :
            return b
        elif b == "1" :
            return a
        else :
            return "%s*%s" % (a,b)

    def negate(self, a) :
        if a == "0" :
            return "1"
        elif a == "1" :
            return "0"
        else :
            return "(1-%s)" % a

    def value(self, a) :
        return str(a)

    def result(self, a) :
        return a

    def normalize(self, a, Z) :
        if Z == "1" :
            return a
        else :
            return "%s / %s" % (a,Z)

class Evaluatable(object) :

    def create_evaluator(self, semiring, weights):
        raise NotImplementedError('Evaluatable.create_evaluator is an abstract method')

    def get_evaluator(self, semiring=None, evidence=None, weights=None):
        if semiring is None :
            semiring = SemiringProbability()

        evaluator = self.create_evaluator(semiring, weights)

        for n_ev, node_ev in evaluator.get_names(self.LABEL_EVIDENCE_POS) :
            if node_ev == 0 :
                # Evidence is already deterministically true
                pass
            elif node_ev is None :
                # Evidence is deterministically true
                raise InconsistentEvidenceError()
            elif evidence is None :
                evaluator.add_evidence( node_ev )
            else :
                value = evidence.get( n_ev, None )
                if value == True :
                    evaluator.add_evidence( node_ev )
                elif value == False :
                    evaluator.add_evidence( -node_ev )

        for n_ev, node_ev in evaluator.get_names(self.LABEL_EVIDENCE_NEG) :
            if node_ev is None :
                # Evidence is already deterministically false
                pass
            elif node_ev == 0 :
                # Evidence is deterministically true
                # TODO raise correct error
                raise InconsistentEvidenceError()
            elif evidence is None :
                evaluator.add_evidence( -node_ev )
            else :
                value = evidence.get( n_ev, None )
                if value == True :
                    evaluator.add_evidence( node_ev )
                elif value == False :
                    evaluator.add_evidence( -node_ev )

        if evidence != None :
            for n_ev, node_ev in evaluator.get_names(self.LABEL_EVIDENCE_MAYBE) :
                value = evidence.get( n_ev, None )
                if value == True :
                    evaluator.add_evidence( node_ev )
                elif value == False :
                    evaluator.add_evidence( -node_ev )


        evaluator.propagate()
        return evaluator

    def evaluate(self, index=None, semiring=None, evidence=None, weights=None) :
        evaluator = self.get_evaluator(semiring, evidence, weights)

        if index is None :
            result = {}
            # Probability of query given evidence
            for name, node in evaluator.get_names(self.LABEL_QUERY):
                w = evaluator.evaluate(node)
                if not semiring is None:
                    w = semiring.result(w)
                result[name] = w
            return result
        else :
            return evaluator.evaluate(node)





class Evaluator(object) :

    def __init__(self, formula, semiring) :
        self.formula = formula
        self.__semiring = semiring

        self.__evidence = []

    @property
    def semiring(self):
        return self.__semiring

    def initialize(self):
        raise NotImplementedError('Evaluator.initialize() is an abstract method.')

    def propagate(self):
        raise NotImplementedError('Evaluator.propagate() is an abstract method.')

    def evaluate(self, index):
        """Compute the value of the given node."""
        raise NotImplementedError('Evaluator.evaluate() is an abstract method.')

    def evaluate_evidence(self):
        raise NotImplementedError('Evaluator.evaluate_evidence is an abstract method.')

    def get_z(self):
        """Get the normalization constant."""
        raise NotImplementedError('Evaluator.get_z() is an abstract method.')

    def add_evidence(self, node):
        """Add evidence"""
        self.__evidence.append(node)

    def has_evidence(self):
        return self.__evidence != []

    def clear_evidence(self):
        self.__evidence = []

    def evidence(self):
        return iter(self.__evidence)
