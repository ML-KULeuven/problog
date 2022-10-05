"""
Part of the ProbLog distribution.

Copyright 2022 KU Leuven, DTAI Research Group

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

import pytest

import problog
import problog.evaluator
from problog.engine import GenericEngine
from problog.formula import LogicFormula
from problog.logic import Term, Constant
from problog.program import PrologString, LogicProgram

# noinspection PyBroadException
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

_MOCK_SEMIRING_CONSTRUCTED_COUNT = 0


class MockSemiring(problog.evaluator.SemiringProbability):
    def __init__(self):
        """Constructor to log that this was the semi ring that was actually constructed"""
        super().__init__()
        global _MOCK_SEMIRING_CONSTRUCTED_COUNT
        _MOCK_SEMIRING_CONSTRUCTED_COUNT += 1

    @classmethod
    def create(cls, *, engine, database, **kwargs):
        assert isinstance(engine, GenericEngine)
        assert isinstance(database, LogicProgram)
        return cls()


problog.register_semiring("mock", MockSemiring)

program = """
0.5::a.
b(P) :- subquery(a, P, [], "{semiring}", "{evaluator}").
query(b(_)).
"""


@pytest.mark.parametrize("eval_name", evaluatables)
def test_subquery(eval_name):
    """Test that the correct semiring gets called, the correct number of times."""
    old_semiring_count = _MOCK_SEMIRING_CONSTRUCTED_COUNT

    # Construct & ground program
    pl = PrologString(program.format(semiring="mock", evaluator=eval_name))
    lf = LogicFormula.create_from(pl, label_all=True, avoid_name_clash=True)
    semiring = problog.get_semiring("logprob")()
    kc_class = problog.get_evaluatable(name=eval_name, semiring=semiring)
    kc = kc_class.create_from(lf)

    assert _MOCK_SEMIRING_CONSTRUCTED_COUNT == old_semiring_count + 1

    b = Term("b", Constant(0.5))
    results = kc.evaluate(semiring=semiring)
    assert 1.0 == results[b]

    # Subqueries are evaluated during grounding, instantiation count should not change here.
    assert _MOCK_SEMIRING_CONSTRUCTED_COUNT == old_semiring_count + 1


@pytest.mark.parametrize(
    ("eval_name", "expected"),
    zip(
        evaluatables,
        [
            # ddnnf gives a different result for whatever reason
            Term("b", Constant("0.5")),
            Term("b", Constant("0.5 / (0.5 + (1-0.5))")),
            Term("b", Constant("0.5 / (0.5 + (1-0.5))")),
            # fsdd doesn't work: TypeError: must be real number, not str
            Term("b", Constant("0.5 / (0.5 + (1-0.5))")),
        ],
    ),
)
def test_subquery_symbolic(eval_name, expected):
    """
    Test the symbolic semiring in subqueries. This has observably different results
    (unlike prob vs logprob).
    """
    if eval_name == "fsdd":
        pytest.xfail(
            "fsdd + symbolic semiring results in TypeError: must be real number, not str"
        )
    pl = PrologString(program.format(semiring="symbolic", evaluator=eval_name))
    lf = LogicFormula.create_from(pl, label_all=True, avoid_name_clash=True)
    # Outer subquery is not symbolic.
    semiring = problog.get_semiring("logprob")()
    kc_class = problog.get_evaluatable(name=eval_name, semiring=semiring)
    kc = kc_class.create_from(lf)

    results = kc.evaluate(semiring=semiring)
    assert expected in results
    assert 1.0 == results[expected]
