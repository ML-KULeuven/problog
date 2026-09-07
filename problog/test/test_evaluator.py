"""
Part of the ProbLog distribution.

Copyright 2019 KU Leuven, DTAI Research Group

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
import unittest

from problog import get_evaluatable
from problog.evaluator import (
    Evaluatable,
    Semiring,
    SemiringLogProbability,
    SemiringProbability,
)
from problog.formula import LogicFormula
from problog.logic import Term
from problog.program import PrologString

# noinspection PyBroadException
from problog.test.test_system import SemiringProbabilityNSPCopy

try:
    from pysdd import sdd

    has_sdd = True
except Exception as err:
    has_sdd = False

evaluatables = ["ddnnf"]

if has_sdd:
    evaluatables.append("sdd")
    evaluatables.append("sddx")
    evaluatables.append("fsdd")
else:
    print("No SDD support - The evaluator tests are not performed with SDDs.")


class TestEvaluator(unittest.TestCase):
    def test_evaluate_custom_weights(self):
        """
        Tests evaluate() with custom weights (not the ones from the ProbLog file)
        """
        for eval_name in evaluatables:
            with self.subTest(eval_name=eval_name):
                self.evaluate_custom_weights(eval_name)

    def evaluate_custom_weights(self, eval_name=None):
        class TestSemiringProbabilityNSP(SemiringProbability):
            def is_nsp(self):
                return True

            def pos_value(self, a, key=None):
                if isinstance(a, tuple):
                    return float(a[0])
                else:
                    return float(a)

            def neg_value(self, a, key=None):
                if isinstance(a, tuple):
                    return float(a[1])
                else:
                    return 1 - float(a)

        program = """
                    0.25::a.
                    query(a).
                """
        pl = PrologString(program)
        lf = LogicFormula.create_from(pl, label_all=True, avoid_name_clash=True)
        semiring = TestSemiringProbabilityNSP()
        kc_class = get_evaluatable(name=eval_name, semiring=semiring)
        kc = kc_class.create_from(lf)
        a = Term("a")

        # without custom weights
        results = kc.evaluate(semiring=semiring)
        self.assertEqual(0.25, results[a])

        # with custom weights
        weights = {a: 0.1}
        results = kc.evaluate(semiring=semiring, weights=weights)
        self.assertEqual(0.1, results[a])

        # with custom weights
        weights = {a: (0.1, 0.1)}
        results = kc.evaluate(semiring=semiring, weights=weights)
        self.assertEqual(0.5, results[a])

        # with custom weights based on index
        weights = {kc.get_node_by_name(a): 0.2}
        results = kc.evaluate(semiring=semiring, weights=weights)
        self.assertEqual(0.2, results[a])

        # Testing with weight on node 0 (True)
        weights = {0: 0.3, a: (0.1, 0.1)}
        results = kc.evaluate(semiring=semiring, weights=weights)
        self.assertEqual(0.5, results[a])

        # Testing query on node 0 (True)
        class TestSemiringProbabilityIgnoreNormalize(SemiringProbabilityNSPCopy):
            def normalize(self, a, z):
                return a

        weights = {0: (0.3, 0.7), a: (0.1, 0.1)}
        results = kc.evaluate(
            index=0, semiring=TestSemiringProbabilityIgnoreNormalize(), weights=weights
        )
        self.assertEqual(0.06, results)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluator)
    unittest.TextTestRunner(verbosity=2).run(suite)


class TestResultDomain(unittest.TestCase):
    """A compiler that returns a formula not representing the program must not
    produce a silently wrong probability (issue #113)."""

    def test_probability_semirings_reject_impossible_results(self):
        for semiring in (SemiringProbability(), SemiringLogProbability()):
            # result() of the log semiring exponentiates, so both report
            # probabilities as their external value.
            self.assertTrue(semiring.result_in_domain(0.0))
            self.assertTrue(semiring.result_in_domain(0.5))
            self.assertTrue(semiring.result_in_domain(1.0))
            self.assertFalse(semiring.result_in_domain(156.1304671248511))
            self.assertFalse(semiring.result_in_domain(-0.5))

    def test_custom_semirings_are_unconstrained(self):
        # aProbLog semirings compute over arbitrary values and must not be
        # restricted to [0, 1].
        class CustomSemiring(Semiring):
            def one(self):
                return 1

            def zero(self):
                return 0

            def plus(self, a, b):
                return a + b

            def times(self, a, b):
                return a * b

        self.assertTrue(CustomSemiring().result_in_domain(156.13))

    def test_evaluate_rejects_an_out_of_domain_result(self):
        from problog.errors import CompilationError

        class FakeFormula(object):
            def labeled(self):
                return [(Term("q"), 1, None)]

        class FakeEvaluator(object):
            def __init__(self):
                self.formula = FakeFormula()
                self.semiring = SemiringProbability()

            def evaluate(self, index):
                return 156.1304671248511

        class FakeEvaluatable(Evaluatable):
            def _create_evaluator(self, semiring, weights, **kwargs):
                return FakeEvaluator()

            def get_evaluator(self, semiring=None, evidence=None, weights=None, **kwargs):
                return FakeEvaluator()

        with self.assertRaises(CompilationError):
            FakeEvaluatable().evaluate()
        with self.assertRaises(CompilationError):
            FakeEvaluatable().evaluate(index=1)
