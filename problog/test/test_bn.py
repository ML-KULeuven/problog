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
import unittest

import glob
import os
import sys

from problog.formula import LogicDAG
from problog.tasks.bayesnet import formula_to_bn
from problog.program import PrologFile, DefaultPrologParser, ExtendedPrologFactory

if __name__ == "__main__":
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    )

from problog import root_path


class TestBNGeneric(unittest.TestCase):
    pass


def read_expected_result(filename):
    """
    Get the expected result from the given file. The expected outcome is the % message following after
    %Expected outcome:
    :param filename: The name of the file
    :return: The expected result.strip().
    :rtype: str
    """
    result = ""
    with open(filename) as f:
        reading = False
        for l in f:
            l = l.strip()
            if l.startswith("%Expected outcome:"):
                reading = True
            elif reading:
                if l.lower().startswith("% error"):
                    return l[len("% error") :].strip()
                elif l.startswith("%"):
                    result = result + "\n" + l[1:]
                else:
                    reading = False
    return result.strip()


def createBNTestGeneric(filename, logspace=False):

    correct = read_expected_result(filename)

    def test(self):
        try:
            target = LogicDAG  # Break cycles
            gp = target.createFrom(
                PrologFile(
                    filename, parser=DefaultPrologParser(ExtendedPrologFactory())
                ),
                label_all=True,
                avoid_name_clash=False,
                keep_order=True,  # Necessary for to prolog
                keep_duplicates=False,
            )
            bn = formula_to_bn(gp)
            computed = str(bn).strip()
        except Exception as err:
            e = err
            computed = None

        if computed is None:
            self.assertEqual(correct, type(e).__name__)
        else:
            self.assertIsInstance(computed, str)
            self.assertEqual(correct, computed)

    return test


if __name__ == "__main__":
    filenames = sys.argv[1:]
else:
    filenames = glob.glob(root_path("test/bn", "*.pl"))


for testfile in filenames:
    testname = "test_bn_" + os.path.splitext(os.path.basename(testfile))[0]
    setattr(TestBNGeneric, testname, createBNTestGeneric(testfile, True))


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBNGeneric)
    unittest.TextTestRunner(verbosity=2).run(suite)
