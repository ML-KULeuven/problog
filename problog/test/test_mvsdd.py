"""
Part of the ProbLog distribution.

Copyright 2026 KU Leuven, DTAI Research Group

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

from problog.evaluator import SemiringLogProbability
from problog.logic import Term
from problog.mvsdd_formula import MVSDD, ISSUE_URL
from problog.program import PrologString

# Grounding keeps only the heads the queries need: c(1,b) and c(2,r) are left out, and the
# choice of none of the heads absorbs their probability.
COLOURS = """
0.2::c(X,r); 0.3::c(X,g); 0.4::c(X,b) :- between(1,2,X).
0.5::f.
q :- c(1,r), \\+ c(2,b).
q :- f, c(2,g).
query(q).
query(c(1,r)).
query(c(2,g)).
"""

COLOURS_RESULT = {
    "q": 0.2 * 0.6 + 0.5 * 0.3 - 0.2 * 0.5 * 0.3,
    "c(1,r)": 0.2,
    "c(2,g)": 0.3,
}

# Given c(1) is not g, c(1) is r with probability 0.2 / 0.7.
COLOURS_EVIDENCE = COLOURS + "evidence(\\+ c(1,g)).\n"

COLOURS_EVIDENCE_RESULT = {
    "q": 2 / 7 * 0.6 + 0.5 * 0.3 - 2 / 7 * 0.5 * 0.3,
    "c(1,r)": 2 / 7,
    "c(2,g)": 0.3,
}


def choice_program(heads):
    """An annotated disjunction over pick(1..heads), with queries that need all of its heads
    and its exclusivity."""
    p = 1.0 / (heads + 1)
    ad = "; ".join("%s::pick(%d)" % (p, i) for i in range(1, heads + 1))
    return (
        "%s.\n"
        "both :- pick(1), pick(2).\n"
        "some :- pick(_).\n"
        "query(both).\n"
        "query(some).\n" % ad
    )


@unittest.skipUnless(MVSDD.is_available(), "mv-sdd is not installed")
class TestMVSDD(unittest.TestCase):
    def assertResults(self, expected, computed):
        computed = {str(k): v for k, v in computed.items()}
        self.assertEqual(set(expected), set(computed))
        for query, value in expected.items():
            self.assertAlmostEqual(value, computed[query], msg=query)

    def test_annotated_disjunction_is_one_variable(self):
        kc = MVSDD.create_from(PrologString(COLOURS))
        # c(2) has heads g and b and the choice of none of them.  c(1) needs only its head r,
        # which is then an ordinary probabilistic fact, as is f.
        sizes = sorted(v.domain_size for v in kc.get_variables())
        self.assertEqual([2, 2, 3], sizes)
        self.assertTrue(kc.get_manager().is_true(kc.get_constraint_inode()))
        self.assertResults(COLOURS_RESULT, kc.evaluate())

    def test_evidence(self):
        kc = MVSDD.create_from(PrologString(COLOURS_EVIDENCE))
        self.assertResults(COLOURS_EVIDENCE_RESULT, kc.evaluate())

    def test_logspace(self):
        kc = MVSDD.create_from(PrologString(COLOURS_EVIDENCE))
        computed = kc.evaluate(semiring=SemiringLogProbability())
        self.assertResults(COLOURS_EVIDENCE_RESULT, computed)

    def test_custom_weights(self):
        kc = MVSDD.create_from(PrologString(COLOURS))
        computed = kc.evaluate(weights={Term("f"): 1.0})
        self.assertAlmostEqual(0.2 * 0.6 + 0.3 - 0.2 * 0.3, computed[Term("q")])

    def test_largest_native_annotated_disjunction(self):
        # 63 heads and the choice of none of them fill the 64 values of a variable.
        with self.assertNoLogs("problog", level="WARNING"):
            kc = MVSDD.create_from(PrologString(choice_program(63)))
            computed = kc.evaluate()
        self.assertEqual([64], [v.domain_size for v in kc.get_variables()])
        self.assertResults({"both": 0.0, "some": 63 / 64}, computed)

    def test_large_annotated_disjunction_falls_back_to_booleans(self):
        with self.assertLogs("problog", level="WARNING") as logs:
            kc = MVSDD.create_from(PrologString(choice_program(64)))
            computed = kc.evaluate()
        self.assertEqual(1, len(logs.output))
        self.assertIn("pick(1)", logs.output[0])
        self.assertIn(ISSUE_URL, logs.output[0])
        self.assertEqual({2}, set(v.domain_size for v in kc.get_variables()))
        self.assertResults({"both": 0.0, "some": 64 / 65}, computed)

    def test_to_formula(self):
        kc = MVSDD.create_from(PrologString(COLOURS))
        formula = kc.to_formula()
        self.assertResults(COLOURS_RESULT, MVSDD.create_from(formula).evaluate())

    def test_internal_dot(self):
        kc = MVSDD.create_from(PrologString(COLOURS))
        dot = kc.to_dot(use_internal=True)
        self.assertTrue(dot.startswith("digraph"))
        self.assertIn("c(1,r)", dot)

    def test_no_probabilistic_atoms(self):
        kc = MVSDD.create_from(PrologString("a.\nb :- fail.\nquery(a).\nquery(b).\n"))
        self.assertResults({"a": 1.0, "b": 0.0}, kc.evaluate())


if __name__ == "__main__":
    unittest.main()
