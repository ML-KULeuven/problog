from __future__ import print_function

import unittest

from problog.program import PrologString
from problog.formula import LogicFormula
from problog import get_evaluatable
from problog.evaluator import SemiringProbability

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
    print("No SDD support - The transformation tests are not performed with SDDs.")


class TestTransformation(unittest.TestCase):
    def setUp(self):
        try:
            self.assertCollectionEqual = self.assertItemsEqual
        except AttributeError:
            self.assertCollectionEqual = self.assertCountEqual

    def test_ad_atom_duplicate(self):
        """
        Run ad_atom_duplicate for each evaluatable.
        This test must pickup the case where during the transformation, additional _extra atoms are created because
        add_atom(..., cr_extra=True) is used instead of cr_extra=False.
        """
        for eval_name in evaluatables:
            with self.subTest(eval_name=eval_name):
                self.ad_atom_duplicate(eval_name)

    def ad_atom_duplicate(self, eval_name=None):
        """
        This test must pickup the case where during the transformation, additional _extra atoms are created because
        add_atom(..., cr_extra=True) is used instead of cr_extra=False.
        """
        program = """
                    0.2::a ; 0.8::b.
                    query(a).
                    query(b).
                """
        pl = PrologString(program)
        lf = LogicFormula.create_from(pl, label_all=True, avoid_name_clash=True)
        semiring = SemiringProbability()
        kc_class = get_evaluatable(name=eval_name, semiring=semiring)
        kc = kc_class.create_from(lf)  # type: LogicFormula
        self.assertEqual(3, kc.atomcount)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTransformation)
    unittest.TextTestRunner(verbosity=2).run(suite)
