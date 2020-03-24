"""
Module name
"""

from __future__ import print_function

from problog import root_path
from problog.util import subprocess_call, subprocess_check_output
import unittest
import os
import sys
import glob
import subprocess, traceback

if __name__ == "__main__":
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    )

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
                    return l[len("% error: ") :].strip()
                elif l.startswith("% "):
                    res = l[2:]
                    results.append(res)
                else:
                    reading = False
    return results


def createTestLFI(filename, useparents=False):
    def test(self):
        for eval_name in evaluatables:
            with self.subTest(evaluatable=eval_name):
                test_func(self, evaluatable=eval_name)

    def test_func(self, evaluatable="ddnnf"):
        problogcli = root_path("problog-cli.py")

        model = filename
        examples = filename.replace(".pl", ".ev")
        out_model = filename.replace(".pl", ".l_pl")

        expected = read_result(model)

        if not os.path.exists(examples):
            raise Exception("Evidence file is missing: {}".format(examples))
        if useparents:
            try:
                out = subprocess_check_output(
                    [
                        sys.executable,
                        problogcli,
                        "lfi",
                        "-k",
                        evaluatable,
                        "-n",
                        "500",
                        "-o",
                        model.replace(".pl", ".l_pl"),
                        model,
                        examples,
                    ]
                )
            except Exception as err:
                print(expected)
                print(err)
                assert expected == "NonGroundProbabilisticClause"
                return
                # # This test is specifically for test/lfi/AD/relatedAD_1 and test/lfi/AD/relatedAD_2
                # # print(type(err))
                # if isinstance(err, subprocess.CalledProcessError):
                # # print(type(err))
                #     print(err.output)
                #     tb = traceback.format_exc()
                #     print(tb)

        with open(out_model, "r") as f:
            outlines = f.readlines()

        outlines = [line.strip() for line in outlines]

        assert len(expected) == len(outlines)
        for expectedline, line in zip(expected, outlines):
            line = line.strip()
            if "::" in line:
                if ";" not in line:
                    if "<RAND>" in expectedline and ";" not in expectedline:
                        line = "<RAND>::" + line.split("::")[1]
                    else:
                        prob = float(line.split("::")[0])
                        expectedline_prob = float(expectedline.split("::")[0])
                        rounded_prob = "{:.6f}".format(prob)
                        rounded_expectedline_prob = "{:.6f}".format(expectedline_prob)
                        if (
                            abs(float(rounded_prob) - float(rounded_expectedline_prob))
                            < 0.00001
                        ):
                            line = (
                                rounded_expectedline_prob + "::" + line.split("::")[1]
                            )
                        else:
                            line = rounded_prob + "::" + line.split("::")[1]
                        expectedline = (
                            str(rounded_expectedline_prob)
                            + "::"
                            + expectedline.split("::")[1]
                        )

                else:
                    if ";" in expectedline:
                        ad_expectedlines = expectedline.split(";")
                        ad_lines = line.split(";")
                        rounded_ad_lines = []
                        rounded_expected_ad_lines = []
                        # TODO assert len(ad_expectedlines) == len(ad_lines)
                        for ad_expectedline, ad_line in zip(ad_expectedlines, ad_lines):
                            ad_line = ad_line.strip()
                            ad_expectedline = ad_expectedline.strip()
                            if "<RAND>" in ad_expectedline:
                                ad_line = "<RAND>::" + ad_line.split("::")[1]
                            else:
                                ad_prob = float(ad_line.split("::")[0])
                                ad_expectedline_prob = float(
                                    ad_expectedline.split("::")[0]
                                )
                                rounded_ad_prob = "{:.6f}".format(ad_prob)
                                rounded_ad_expectedline_prob = "{:.6f}".format(
                                    ad_expectedline_prob
                                )
                                if (
                                    abs(
                                        float(rounded_ad_prob)
                                        - float(rounded_ad_expectedline_prob)
                                    )
                                    < 0.00001
                                ):
                                    ad_line = (
                                        rounded_ad_expectedline_prob
                                        + "::"
                                        + ad_line.split("::")[1]
                                    )
                                else:
                                    ad_line = (
                                        rounded_ad_prob + "::" + ad_line.split("::")[1]
                                    )
                                ad_expectedline = (
                                    rounded_ad_expectedline_prob
                                    + "::"
                                    + ad_expectedline.split("::")[1]
                                )
                            rounded_ad_lines.append(ad_line)
                            rounded_expected_ad_lines.append(ad_expectedline)
                        line = "; ".join(rounded_ad_lines)
                        expectedline = "; ".join(rounded_expected_ad_lines)
                    else:
                        raise AssertionError
            # rounded_outlines.append(line)
            assert expectedline == line

        print(expected)
        print(outlines)

    return test


def ignore_previous_output(path):
    # dir_name = "../../test/lfi/unit_tests/"
    test = os.listdir(path)
    for item in test:
        if item.endswith(".l_pl"):
            os.remove(os.path.join(path, item))


if __name__ == "__main__":
    filenames = sys.argv[1:]
else:
    ignore_previous_output("../../test/lfi/AD/")
    ADfilenames = glob.glob(root_path("test", "lfi", "AD", "*.pl"))
    ignore_previous_output("../../test/lfi/simple/")
    simple_filenames = glob.glob(root_path("test", "lfi", "simple", "*.pl"))
    ignore_previous_output("../../test/lfi/useParents/")
    useParents_filenames = glob.glob(root_path("test", "lfi", "useParents", "*.pl"))
    ignore_previous_output("../../test/lfi/unit_tests/")
    unit_test_filenames = glob.glob(root_path("test", "lfi", "unit_tests", "test_*.pl"))
    ignore_previous_output("../../test/lfi/test_interface/")
    test_interface_filenames = glob.glob(
        root_path("test", "lfi", "test_interface", "test1_model.pl")
    )
    # ignore_previous_output("../../test/lfi/todo/")
    # test_todo_filenames = glob.glob(root_path("test", "lfi", "todo", "*.pl"))

evaluatables = ["ddnnf"]

if has_sdd:
    evaluatables.append("sdd")
    evaluatables.append("sddx")
else:
    print("No SDD support - The system tests are not performed with SDDs.")

# tests for simple cases (non-ADs)
for testfile in simple_filenames:
    testname = "test_lfi_simple_" + os.path.splitext(os.path.basename(testfile))[0]
    setattr(TestLFI, testname, createTestLFI(testfile, True))

# tests for ADs
for testfile in ADfilenames:
    testname = "test_lfi_AD_" + os.path.splitext(os.path.basename(testfile))[0]
    setattr(TestLFI, testname, createTestLFI(testfile, True))

# tests for useParents
for testfile in useParents_filenames:
    testname = "test_lfi_parents_" + os.path.splitext(os.path.basename(testfile))[0]
    setattr(TestLFI, testname, createTestLFI(testfile, True))

# tests for unit tests
for testfile in unit_test_filenames:
    testname = "test_lfi_unit_test_" + os.path.splitext(os.path.basename(testfile))[0]
    setattr(TestLFI, testname, createTestLFI(testfile, True))

# tests for test_interface
for testfile in test_interface_filenames:
    testname = (
        "test_lfi_test_interface_" + os.path.splitext(os.path.basename(testfile))[0]
    )
    setattr(TestLFI, testname, createTestLFI(testfile, useparents=True))
# for testfile in test_todo_filenames:
#     testname = (
#         "test_lfi_test_todo_" + os.path.splitext(os.path.basename(testfile))[0]
#     )
#     setattr(TestLFI, testname, createTestLFI(testfile, useparents=True))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLFI)
    unittest.TextTestRunner(verbosity=2).run(suite)
