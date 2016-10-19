"""
problog.kbest - K-Best inference using MaxSat
---------------------------------------------

Anytime evaluation using best proofs.

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


from .core import transform
from .formula import LogicDAG
from .constraint import TrueConstraint, ClauseConstraint

from .cnf_formula import CNF, clarks_completion
from .maxsat import get_solver, UnsatisfiableError
from .evaluator import Evaluator, Evaluatable
from .logic import Term

from copy import deepcopy

import warnings
import logging

from functools import total_ordering


class KBestFormula(CNF, Evaluatable):

    transform_preference = 40

    def __init__(self, **kwargs):
        CNF.__init__(self)
        Evaluatable.__init__(self)

    def _create_evaluator(self, semiring, weights, **kwargs):
        return KBestEvaluator(self, semiring, weights, **kwargs)

    @classmethod
    def is_available(cls):
        """Checks whether the SDD library is available."""
        return True


transform(LogicDAG, KBestFormula, clarks_completion)


class KBestEvaluator(Evaluator):

    def __init__(self, formula, semiring, weights=None, lower_only=False,
                 verbose=None, convergence=1e-9, explain=None, **kwargs):
        Evaluator.__init__(self, formula, semiring, weights, **kwargs)

        self.sdd_manager = None
        # self.sdd_manager = SDDManager()
        # for l in range(1, formula.atomcount + 1):
        #     self.sdd_manager.add_variable(l)

        self._z = None
        self._weights = None
        self._given_weights = weights
        self._verbose = verbose
        self._lower_only = lower_only
        self._explain = explain
        if explain is not None:
            self._lower_only = True

        self._reverse_names = {index: name for name, index in self.formula.get_names()}

        self._convergence = convergence

    def initialize(self):
        raise NotImplementedError('Evaluator.initialize() is an abstract method.')

    def propagate(self):
        self._weights = self.formula.extract_weights(self.semiring, self._given_weights)
        self._z = self.semiring.one()
        # self._z = self.sdd_manager.wmc_true(self._weights, self.semiring)

    def evaluate(self, index):
        """Compute the value of the given node."""

        name = [n for n, i, l in self.formula.labeled() if i == index]
        if name:
            name = name[0]
        else:
            name = index

        logger = logging.getLogger('problog')
        logger.debug('evaluating query %s' % name)

        if index is None:
            if self._explain is not None:
                self._explain.append('%s :- fail.' % name)
            return 0.0
        elif index == 0:
            if self._explain is not None:
                self._explain.append('%s :- true.' % name)
            return 1.0
        else:
            lb = Border(self.formula, self.sdd_manager, self.semiring, index, 'lower')
            ub = Border(self.formula, self.sdd_manager, self.semiring, -index, 'upper')

            k = 0
            # Select the border with most improvement
            if self._lower_only:
                nborder = lb
            else:
                nborder = max(lb, ub)

            try:
                while not nborder.is_complete():
                    solution = nborder.update()
                    logger.debug('  update: %s %s < p < %s ' %
                                 (nborder.name, lb.value, 1.0 - ub.value))
                    if self._explain is not None and solution is not None:
                        solution_names = []

                        probability = nborder.improvement
                        for s in solution:
                            n = self._reverse_names.get(abs(s), Term('choice_%s' % (abs(s))))
                            if s < 0:
                                solution_names.append(-n)
                            else:
                                solution_names.append(n)
                        proof = ', '.join(map(str, solution_names))
                        self._explain.append('%s :- %s.  %% P=%.8g' % (name, proof, probability))

                    if solution is not None:
                        k += 1

                    if nborder.is_complete():
                        if nborder == lb:
                            if self._explain is not None:
                                if k == 0:
                                    self._explain.append('%s :- fail.' % name)
                                self._explain.append('')
                            return lb.value
                        else:
                            return 1.0 - ub.value

                    if ub.value + lb.value > 1.0 - self._convergence:
                        logger.debug('  convergence reached')
                        return lb.value, 1.0 - ub.value
                    if self._lower_only:
                        nborder = lb
                    else:
                        nborder = max(lb, ub)
            except KeyboardInterrupt:
                pass
            except SystemError:
                pass

            return lb.value, 1.0 - ub.value

    def evaluate_evidence(self):
        raise NotImplementedError('Evaluator.evaluate_evidence is an abstract method.')

    def add_evidence(self, node):
        """Add evidence"""
        warnings.warn('Evidence is not supported by this evaluation method and will be ignored.')

    def has_evidence(self):
        return self.__evidence != []

    def clear_evidence(self):
        self.__evidence = []

    def evidence(self):
        return iter(self.__evidence)


@total_ordering
class Border(object):

    def __init__(self, cnf, manager, semiring, query, name, smart_constraints=False):
        self.wcnf = deepcopy(cnf)
        self.wcnf.add_constraint(TrueConstraint(query), True)

        self.name = name

        self.manager = manager
        self.semiring = semiring

        self.weights = self.wcnf.extract_weights(self.semiring)
        # self.compiled = self.manager.false()

        self.value = 0.0
        self.improvement = 1.0

        self.smart_constraints = smart_constraints

    def update(self):
        solver = get_solver()

        try:
            solution = solver.evaluate(self.wcnf, partial=True, smart_constraints=True)
        except UnsatisfiableError:
            solution = None

        if solution is None:
            self.improvement = None
            return None
        else:
            # m = self.manager

            solution = self.wcnf.from_partial(solution)

            probability = self.semiring.one()
            for s in solution:
                wp, wn = self.weights[abs(s)]
                if s < 0:
                    probability = self.semiring.times(probability, wn)
                else:
                    probability = self.semiring.times(probability, wp)
            probability = self.semiring.result(probability)

            constraint = ClauseConstraint(list(map(lambda x: -x, solution)))
            self.wcnf.add_constraint(constraint, True)
            # literals = list(map(m.literal, solution))

            # proof_sdd = m.conjoin(*literals)
            # sdd_query_new = m.disjoin(self.compiled, proof_sdd)
            # m.deref(proof_sdd, self.compiled)
            # self.compiled = sdd_query_new
            #
            # pquery = m.wmc(self.compiled, self.weights, self.semiring)
            # ptrue = m.wmc_true(self.weights, self.semiring)
            # res = self.semiring.normalize(pquery, ptrue)

            # value = self.semiring.result(res)

            # self.improvement = value - self.value
            # self.value = value

            self.improvement = probability
            self.value = self.value + probability

            assert abs(self.improvement - probability) < 1e-8

            return solution

    def is_complete(self):
        return self.improvement is None

    def __lt__(self, other):
        if self.improvement is None:
            return True
        elif other.improvement is None:
            return False
        else:
            return self.improvement < other.improvement

    def __eq__(self, other):
        return self.improvement == other.improvement


