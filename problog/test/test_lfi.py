"""
test_lfi.py - Test class for LFI problems
"""
import random

from problog import root_path
import unittest
import os
import sys
import glob
from problog.learning.lfi import lfi_wrapper, LFIProblem

try:
    from pysdd import sdd

    has_sdd = True
except Exception as err:
    print("SDD library not available due to error: ", err, file=sys.stderr)
    has_sdd = False


class TestLFI(unittest.TestCase):
    def setUp(self):
        try:
            self.assertSequenceEqual = self.assertItemsEqual
        except AttributeError:
            self.assertSequenceEqual = self.assertCountEqual


def read_result(filename):
    results = []
    with open(filename) as f:
        reading = False
        for l in f:
            l = l.strip()
            if l.startswith("%Expected outcome:"):
                reading = True
            elif reading:
                if l.lower().startswith("% error: "):
                    return l[len("% error: "):].strip()
                elif l.startswith("% "):
                    res = l[2:]
                    results.append(res)
                else:
                    reading = False
    return results


def createTestLFI(filename, evaluatables):
    def test(self):
        for eval_name in evaluatables:
            for logspace in [False, True]:
                with self.subTest(evaluatable=eval_name, logspace=logspace):
                    test_func(self, evaluatable=eval_name, logspace=logspace)

    def test_func(self, evaluatable, logspace):
        model = filename
        examples = filename.replace(".pl", ".ev")
        expectedlines = read_result(model)

        if not os.path.exists(examples):
            raise Exception("Evidence file is missing: {}".format(examples))

        try:
            d = {
                "max_iter": 10000,
                "min_improv": 1e-10,
                "leakprob": None,
                "propagate_evidence": False,
                "logspace": logspace,
                "normalize": True,
                "web": False,
                "args": None,
                "infer_AD_values": True
            }
            random.seed(a=1000)
            score, weights, names, iterations, lfi = lfi_wrapper(
                model, [examples], evaluatable, d
            )
            outlines = lfi.get_model()
        except Exception as _:
            assert expectedlines == "NonGroundProbabilisticClause"
            return

        outlines = outlines.split("\n")[:-1]
        assert len(expectedlines) == len(outlines)

        # We need to sort outlines and expectedlines, because apparently the order of the rules can
        # depend on the target language (-k flag).
        outlines.sort(key=lambda x: x if "::" not in x else x.split("::")[1])
        expectedlines.sort(key=lambda x: x if "::" not in x else x.split("::")[1])

        # Compare expected program and learned program line by line
        for expectedline, outline in zip(expectedlines, outlines):
            # When there are probabilities
            if "::" in outline:
                # Break the lines into components where each component has exactly one probability
                expectedline_comps = expectedline.split(";")
                outline_comps = outline.split(";")
                new_expectedline_comps = []
                new_outline_comps = []
                assert len(outline_comps) == len(expectedline_comps)
                # Compare one expected probability and one learned probability at a time
                for expectedline_comp, outline_comp in zip(
                        expectedline_comps, outline_comps
                ):
                    outline_comp = outline_comp.strip()
                    expectedline_comp = expectedline_comp.strip()
                    # When the learned prob in outline_component does not matter,
                    # discard the learned probability
                    if "<RAND>" in expectedline_comp:
                        outline_comp = "<RAND>::" + outline_comp.split("::")[1]
                    else:
                        # Round the expected and learned probabilities
                        rounded_outline_comp_prob = "{:.6f}".format(
                            float(outline_comp.split("::")[0])
                        )
                        rounded_expectedline_comp_prob = "{:.6f}".format(
                            float(expectedline_comp.split("::")[0])
                        )
                        # Update the expected component probability
                        expectedline_comp = (
                                rounded_expectedline_comp_prob
                                + "::"
                                + expectedline_comp.split("::")[1]
                        )
                        # If the learned probability is close enough to the expected probability
                        if (
                                abs(
                                    float(rounded_outline_comp_prob)
                                    - float(rounded_expectedline_comp_prob)
                                )
                                < 0.00001
                        ):
                            # Make the two lines identical
                            outline_comp = (
                                    rounded_expectedline_comp_prob
                                    + "::"
                                    + outline_comp.split("::")[1]
                            )
                    new_outline_comps.append(outline_comp)
                    new_expectedline_comps.append(expectedline_comp)
                new_outline = "; ".join(new_outline_comps)
                new_expectedline = "; ".join(new_expectedline_comps)
                expectedline = new_expectedline
                outline = new_outline
            self.assertEqual(outline, expectedline)

    return test


def main():
    AD_filenames = glob.glob(root_path("test", "lfi", "ad", "*.pl"))
    simple_filenames = glob.glob(root_path("test", "lfi", "simple", "*.pl"))
    misc_filenames = glob.glob(root_path("test", "lfi", "misc", "*.pl"))
    vars_in_T_filenames = glob.glob(root_path("test", "lfi", "vars_in_tunable", "*.pl"))

    evaluatables = ["ddnnf"]

    if has_sdd:
        evaluatables.append("sdd")
        evaluatables.append("sddx")

    else:
        print("No SDD support - The system tests are not performed with SDDs.")

    # tests for ADs
    for testfile in AD_filenames:
        testname = "test_lfi_ad_" + os.path.splitext(os.path.basename(testfile))[0]
        setattr(TestLFI, testname, createTestLFI(testfile, evaluatables))

    # tests for simple unit tests
    for testfile in simple_filenames:
        testname = "test_lfi_simple_" + os.path.splitext(os.path.basename(testfile))[0]
        setattr(TestLFI, testname, createTestLFI(testfile, evaluatables))

    # tests for Variables in t()
    for testfile in vars_in_T_filenames:
        testname = "test_lfi_vars_inT_" + os.path.splitext(os.path.basename(testfile))[0]
        setattr(TestLFI, testname, createTestLFI(testfile, evaluatables))

    # tests for Miscellaneous files
    for testfile in misc_filenames:
        testname = "test_lfi_misc_" + os.path.splitext(os.path.basename(testfile))[0]
        setattr(TestLFI, testname, createTestLFI(testfile, evaluatables))


main()

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLFI)
    unittest.TextTestRunner(verbosity=2).run(suite)
