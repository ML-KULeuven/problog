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

from problog.program import PrologString
from problog.formula import LogicFormula
from problog import get_evaluatable
from problog.evaluator import SemiringProbability
from problog.logic import Term

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
