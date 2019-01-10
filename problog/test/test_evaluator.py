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

import glob, os, traceback, sys

from problog import root_path
from problog import get_evaluatable

from problog.setup import install
from problog.program import PrologFile, DefaultPrologParser, ExtendedPrologFactory, PrologString
from problog.formula import LogicFormula
from problog.ddnnf_formula import DDNNF
from problog.sdd_formula import SDD
from problog import get_evaluatable
from problog.evaluator import SemiringProbability, SemiringLogProbability
from problog.logic import Term
from parameterizedtestcase import ParameterizedTestCase


# noinspection PyBroadException
try:
    from pysdd import sdd
    has_sdd = True
except Exception as err:
    has_sdd = False

evaluatables = ["ddnnf"]

if has_sdd:
    evaluatables.append("sdd")
else:
    print("No SDD support - The evaluator tests are not performed with SDDs.")


class TestEvaluator(ParameterizedTestCase):

    def setUp(self):
        try:
            self.assertSequenceEqual = self.assertItemsEqual
        except AttributeError:
            self.assertSequenceEqual = self.assertCountEqual

    @ParameterizedTestCase.parameterize(("evaluatable_name",), ((t,) for t in evaluatables), func_name_format='{func_name}_{case_num}')
    def test_evaluate_custom_weights(self, evaluatable_name=None):
        """
        Tests evaluate() with custom weights (not the ones from file)
        """
        class TestSemiringProbabilityNSP(SemiringProbability):
            def is_nsp(self):
                return True

        program = """
                    0.25::a.
                    query(a).
                """
        pl = PrologString(program)
        lf = LogicFormula.create_from(pl, label_all=True, avoid_name_clash=True)
        semiring = TestSemiringProbabilityNSP()
        kc_class = get_evaluatable(name=evaluatable_name, semiring=semiring)
        kc = kc_class.create_from(lf)
        a = Term('a')

        # without custom weights
        results = kc.evaluate(semiring=semiring)
        self.assertEqual(0.25, results[a])

        # with custom weights
        weights = {a: 0.1}
        results = kc.evaluate(semiring=semiring, weights=weights)
        self.assertEqual(0.1, results[a])

if __name__ == '__main__' :
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluator)
    unittest.TextTestRunner(verbosity=2).run(suite)
