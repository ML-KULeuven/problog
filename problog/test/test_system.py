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
import glob
import os
import sys
import unittest

from problog.forward import _ForwardSDD

if __name__ == "__main__":
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    )

from problog import root_path

from problog.program import PrologFile, DefaultPrologParser, ExtendedPrologFactory
from problog import get_evaluatable
from problog.evaluator import SemiringProbability, SemiringLogProbability, Semiring

# noinspection PyBroadException
try:
    from pysdd import sdd

    has_sdd = True
except Exception as err:
    print("SDD library not available due to error: ", err, file=sys.stderr)
    has_sdd = False


class TestDummy(unittest.TestCase):
    def test_dummy(self):
        pass


class TestSystemGeneric(unittest.TestCase):
    def setUp(self):
        try:
            self.assertSequenceEqual = self.assertItemsEqual
        except AttributeError:
            self.assertSequenceEqual = self.assertCountEqual


def read_result(filename):
    results = {}
    with open(filename) as f:
        reading = False
        for l in f:
            l = l.strip()
            if l.startswith("%Expected outcome:"):
                reading = True
            elif reading:
                if l.lower().startswith("% error"):
                    return l[len("% error") :].strip()
                elif l.startswith("% "):
                    query, prob = l[2:].rsplit(None, 1)
                    results[query.strip()] = float(prob.strip())
                else:
                    reading = False
            if l.startswith("query(") and l.find("% outcome:") >= 0:
                pos = l.find("% outcome:")
                query = l[6:pos].strip().rstrip(".").rstrip()[:-1]
                prob = l[pos + 10 :]
                results[query.strip()] = float(prob.strip())
    return results


def createSystemTestGeneric(filename, logspace=False):

    correct = read_result(filename)

    def test(self):
        semirings = {
            "Default": None,
            "Custom": SemiringProbabilityCopy(),
            "CustomNSP": SemiringProbabilityNSPCopy(),
        }
        for eval_name in evaluatables:
            for semiring in semirings:
                with self.subTest(evaluatable_name=eval_name, semiring=semiring):
                    evaluate(
                        self,
                        evaluatable_name=eval_name,
                        custom_semiring=semirings[semiring],
                    )

        # explicit encoding from ForwardSDD
        if has_sdd:
            for semiring in semirings:
                with self.subTest(semiring=semiring):
                    evaluate_explicit_from_fsdd(
                        self, custom_semiring=semirings[semiring]
                    )

    def evaluate(self, evaluatable_name=None, custom_semiring=None):
        try:
            parser = DefaultPrologParser(ExtendedPrologFactory())
            kc = get_evaluatable(name=evaluatable_name).create_from(
                PrologFile(filename, parser=parser)
            )

            if custom_semiring is not None:
                semiring = custom_semiring  # forces the custom semiring code.
            elif logspace:
                semiring = SemiringLogProbability()
            else:
                semiring = SemiringProbability()

            computed = kc.evaluate(semiring=semiring)
            computed = {str(k): v for k, v in computed.items()}
        except Exception as err:
            # print("exception %s" % err)
            e = err
            computed = None

        if computed is None:
            self.assertEqual(correct, type(e).__name__)
        else:
            self.assertIsInstance(correct, dict)
            self.assertSequenceEqual(correct, computed)

            for query in correct:
                self.assertAlmostEqual(correct[query], computed[query], msg=query)

    def evaluate_explicit_from_fsdd(self, custom_semiring=None):
        try:
            parser = DefaultPrologParser(ExtendedPrologFactory())
            lf = PrologFile(filename, parser=parser)
            kc = _ForwardSDD.create_from(lf)  # type: _ForwardSDD
            kc = kc.to_explicit_encoding()

            if custom_semiring is not None:
                semiring = custom_semiring  # forces the custom semiring code.
            elif logspace:
                semiring = SemiringLogProbability()
            else:
                semiring = SemiringProbability()

            computed = kc.evaluate(semiring=semiring)
            computed = {str(k): v for k, v in computed.items()}
        except Exception as err:
            # print("exception %s" % err)
            e = err
            computed = None

        if computed is None:
            self.assertEqual(correct, type(e).__name__)
        else:
            self.assertIsInstance(correct, dict)
            self.assertSequenceEqual(correct, computed)

            for query in correct:
                self.assertAlmostEqual(correct[query], computed[query], msg=query)

    return test


class SemiringProbabilityCopy(Semiring):
    """Mocking SemiringProbability to force the 'custom semiring' code -> Must not extend SemiringProbability."""

    def __init__(self):
        self._semiring = SemiringProbability()

    def one(self):
        return self._semiring.one()

    def zero(self):
        return self._semiring.zero()

    def is_one(self, value):
        return self._semiring.is_one(value)

    def is_zero(self, value):
        return self._semiring.is_zero(value)

    def plus(self, a, b):
        return self._semiring.plus(a, b)

    def times(self, a, b):
        return self._semiring.times(a, b)

    def negate(self, a):
        return self._semiring.negate(a)

    def normalize(self, a, z):
        return self._semiring.normalize(a, z)

    def value(self, a):
        return self._semiring.value(a)

    def is_dsp(self):
        return self._semiring.is_dsp()

    def in_domain(self, a):
        return self._semiring.in_domain(a)


class SemiringProbabilityNSPCopy(SemiringProbabilityCopy):
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


if __name__ == "__main__":
    filenames = sys.argv[1:]
else:
    filenames = glob.glob(root_path("test", "*.pl"))


evaluatables = ["ddnnf"]

if has_sdd:
    evaluatables.append("sdd")
    evaluatables.append("sddx")
    evaluatables.append("fsdd")
else:
    print("No SDD support - The system tests are not performed with SDDs.")


for testfile in filenames:
    testname = "test_system_" + os.path.splitext(os.path.basename(testfile))[0]
    setattr(TestSystemGeneric, testname, createSystemTestGeneric(testfile, True))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemGeneric)
    unittest.TextTestRunner(verbosity=2).run(suite)
