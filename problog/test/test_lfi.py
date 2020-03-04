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

if __name__ == "__main__":
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    )


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
                if l.lower().startswith("% error"):
                    return l[len("% error") :].strip()
                elif l.startswith("% "):
                    res = l[2:]
                    results.append(res)
                else:
                    reading = False
    return results


def createTestLFI(filename, useparents=False):
    def test(self):
        problogcli = root_path("problog-cli.py")

        model = filename
        examples = filename.replace(".pl", ".ev")
        out_model = filename.replace(".pl", ".l_pl")

        expected = read_result(model)

        if not os.path.exists(examples):
            raise Exception("Evidence file is missing: {}".format(examples))
        if useparents:
            out = subprocess_check_output(
                [
                    sys.executable,
                    problogcli,
                    "lfi",
                    # "-k",
                    # "ddnnf",
                    "-n",
                    "500",
                    "-O",
                    model.replace(".pl", ".l_pl"),
                    model,
                    examples,
                    "--useparents",
                ]
            )

        with open(out_model, "r") as f:
            outlines = f.readlines()
        # outlines = [line.strip() for line in outlines]
        rounded_outlines = []
        for line in outlines:
            line = line.strip()
            if "::" in line:
                if ";" not in line:
                    prob = float(line.split("::")[0])
                    rounded_prob = round(prob, 3)
                    if abs(prob - rounded_prob) < 0.000001:
                        line = str(rounded_prob) + "::" + line.split("::")[1]
                    else:
                        line = str(prob) + "::" + line.split("::")[1]
                else:
                    ad_lines = line.split(";")
                    rounded_ad_lines = []
                    for ad_line in ad_lines:
                        ad_line = ad_line.strip()
                        ad_prob = float(ad_line.split("::")[0])
                        rounded_ad_prob = round(ad_prob, 3)
                        if abs(ad_prob - rounded_ad_prob) < 0.000001:
                            ad_line = (
                                str(rounded_ad_prob) + "::" + ad_line.split("::")[1]
                            )
                        else:
                            ad_line = str(ad_prob) + "::" + ad_line.split("::")[1]
                        rounded_ad_lines.append(ad_line)
                    line = "; ".join(rounded_ad_lines)
            rounded_outlines.append(line)
        print(expected)
        print(outlines)
        print(rounded_outlines)
        assert expected == rounded_outlines

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
        root_path("test", "lfi", "test_interface", "*.pl")
    )


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
    setattr(TestLFI, testname, createTestLFI(testfile, True))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLFI)
    unittest.TextTestRunner(verbosity=2).run(suite)
