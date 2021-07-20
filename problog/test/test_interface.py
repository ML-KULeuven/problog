"""
Module name
"""

import os
import sys
import unittest

from problog import root_path
from problog.util import subprocess_call, subprocess_check_output


class TestInterfaces(unittest.TestCase):
    def _test_cmd(self, task, testfile=None):
        problogcli = root_path("problog-cli.py")
        if testfile is None:
            testfile = [root_path("test", "7_probabilistic_graph.pl")]

        with open(os.devnull, "w") as out:
            if task is None:
                self.assertEqual(
                    subprocess_call(
                        [sys.executable, problogcli] + testfile, stdout=out
                    ),
                    0,
                )
            else:
                self.assertEqual(
                    subprocess_call(
                        [sys.executable, problogcli, task] + testfile, stdout=out
                    ),
                    0,
                )

    def test_cli_mpe(self):
        return self._test_cmd("mpe")

    def test_cli_prob(self):
        return self._test_cmd("prob")

    # def test_cli_explain(self):
    #     return self._test_cmd('explain')

    def test_cli_ground(self):
        return self._test_cmd("ground")

    def test_cli_sample(self):
        return self._test_cmd("sample")

    def test_cli_default(self):
        return self._test_cmd(None)

    def test_cli_learn(self):
        problogcli = root_path("problog-cli.py")

        model = root_path("test", "lfi", "ad", "ADtest_8_1.pl")
        examples = root_path("test", "lfi", "ad", "ADtest_8_1.ev")

        out = subprocess_check_output(
            [sys.executable, problogcli, "lfi", model, examples, ]
        )
        assert "[0.4, 0.2, 0.4] [t(_)::a(X), t(_)::b(Y), t(_)::c(Z)]" in out
