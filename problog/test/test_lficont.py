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


class TestLFICont(unittest.TestCase):
    def setUp(self):
        try:
            self.assertSequenceEqual = self.assertItemsEqual
        except AttributeError:
            self.assertSequenceEqual = self.assertCountEqual


def createTestLFICont(filename):
    def test(self):
        problogcli = root_path("problog-cli.py")

        model = filename
        examples = filename.replace(".pl", ".ev")
        if not os.path.exists(examples):
            raise Exception("Evidence file is missing: {}".format(examples))

        out = subprocess_check_output(
            [
                sys.executable,
                problogcli,
                "lfi",
                "-n",
                "10",
                "-O",
                model.replace(".pl", ".l_pl"),
                model,
                examples,
            ]
        )
        outline = out.strip().split()
        print(outline)

    return test


# if __name__ == "__main__":
#     filenames = sys.argv[1:]
# else:
#     filenames = glob.glob(root_path("test", "lficont", "*.pl"))
#
#
# for testfile in filenames:
#     testname = "test_lficont_" + os.path.splitext(os.path.basename(testfile))[0]
#     setattr(TestLFICont, testname, createTestLFICont(testfile))
#
#
# if __name__ == "__main__":
#     suite = unittest.TestLoader().loadTestsFromTestCase(TestLFICont)
#     unittest.TextTestRunner(verbosity=2).run(suite)
