"""
Implementation of k-best approximation for ProbLog.
The task solved is inference.
"""

from __future__ import print_function


from .core import transform
from .program import PrologFile
from .formula import TrueConstraint, ClauseConstraint, LogicDAG
from .sdd_formula import SDDManager
from .cnf_formula import CNF, clarks_completion
from .maxsat import get_solver
from .util import init_logger
from .evaluator import Evaluator, Evaluatable

from copy import deepcopy

import argparse
import sys
import warnings
import logging

from functools import total_ordering


class KBestFormula(CNF, Evaluatable):

    def __init__(self, **kwargs):
        CNF.__init__(self)
        Evaluatable.__init__(self)

    def create_evaluator(self, semiring, weights):
        return KBestEvaluator(self, semiring, weights)

transform(LogicDAG, KBestFormula, clarks_completion)


class KBestEvaluator(Evaluator):

    def __init__(self, formula, semiring, weights=None, verbose=None, **kwargs):
        Evaluator.__init__(self, formula, semiring)

        self.sdd_manager = SDDManager()
        for l in range(1, formula.atomcount+1):
            self.sdd_manager.add_variable(l)

        self._z = None
        self._weights = None
        self._given_weights = weights
        self._verbose = verbose

    def initialize(self):
        raise NotImplementedError('Evaluator.initialize() is an abstract method.')

    def propagate(self):
        self._weights = self.formula.extract_weights(self.semiring, self._given_weights)
        self._z = self.sdd_manager.wmc_true(self._weights, self.semiring)

    def evaluate(self, index):
        """Compute the value of the given node."""

        name = [n for n, i in self.get_names() if i == index]
        if name:
            name = name[0]
        else:
            name = index

        logger = logging.getLogger('problog')
        logger.debug('evaluating query %s' % name)

        if index is None:
            return 0.0
        elif index == 0:
            return 1.0
        else:
            lb = Border(self.formula, self.sdd_manager, self.semiring, index)
            ub = Border(self.formula, self.sdd_manager, self.semiring, -index)

            try:
                # Select the border with most improvement
                nborder = max(lb, ub)
                while not nborder.is_complete():
                    solution = nborder.update()
                    if nborder.is_complete():
                        if nborder == lb:
                            return lb.value
                        else:
                            return 1.0-ub.value
                    logger.debug('  update: %s < p < %s' % (lb.value, 1-ub.value))
                    nborder = max(lb, ub)
            except KeyboardInterrupt:
                pass
            except SystemError:
                pass
            return lb.value, 1.0-ub.value

    def evaluate_evidence(self):
        raise NotImplementedError('Evaluator.evaluate_evidence is an abstract method.')

    def get_z(self):
        """Get the normalization constant."""
        return self._z

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

    def __init__(self, cnf, manager, semiring, query):
        self.wcnf = deepcopy(cnf)
        self.wcnf.add_constraint(TrueConstraint(query))

        self.manager = manager
        self.semiring = semiring

        self.weights = self.wcnf.extract_weights(self.semiring)
        self.compiled = self.manager.false()

        self.value = 0.0
        self.improvement = 1.0

    def update(self):
        solver = get_solver()
        solution = solver.evaluate(self.wcnf, partial=True)

        if solution is None:
            self.improvement = None
            return None
        else:
            m = self.manager

            solution = self.wcnf.from_partial(solution)
            self.wcnf.add_constraint(ClauseConstraint(list(map(lambda x: -x, solution))))
            literals = list(map(m.literal, solution))
            proof_sdd = m.conjoin(*literals)
            sdd_query_new = m.disjoin(self.compiled, proof_sdd)
            m.deref(proof_sdd, self.compiled)
            self.compiled = sdd_query_new

            pquery = m.wmc(self.compiled, self.weights, self.semiring)
            ptrue = m.wmc_true(self.weights, self.semiring)
            res = self.semiring.normalize(pquery, ptrue)

            value = self.semiring.result(res)
            self.improvement = value - self.value
            self.value = value
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


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-v', '--verbose', action='count')
    args = parser.parse_args(argv)

    init_logger(args.verbose)

    cnf = KBestFormula.createFrom(PrologFile(args.filename))

    results = cnf.evaluate()
    for k, v in results.items():
        print (k, v[0], v[1])


if __name__ == '__main__':
    main(sys.argv[1:])
