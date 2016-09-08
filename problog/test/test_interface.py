"""
Module name
"""

from __future__ import print_function

from problog import root_path
from problog.util import subprocess_call, subprocess_check_output
import unittest
import os
import sys


class TestInterfaces(unittest.TestCase):

    def _test_cmd(self, task, testfile=None):
        problogcli = root_path('problog-cli.py')
        if testfile is None:
            testfile = [root_path('test', '7_probabilistic_graph.pl')]

        with open(os.devnull, 'w') as out:
            if task is None:
                self.assertEqual(
                    subprocess_call([sys.executable, problogcli] + testfile, stdout=out), 0)
            else:
                self.assertEqual(
                    subprocess_call([sys.executable, problogcli, task] + testfile, stdout=out), 0)

    def test_cli_mpe(self):
        return self._test_cmd('mpe')

    def test_cli_prob(self):
        return self._test_cmd('prob')

    # def test_cli_explain(self):
    #     return self._test_cmd('explain')

    def test_cli_ground(self):
        return self._test_cmd('ground')

    def test_cli_sample(self):
        return self._test_cmd('sample')

    def test_cli_default(self):
        return self._test_cmd(None)

    def test_cli_learn(self):
        problogcli = root_path('problog-cli.py')

        model = root_path('problog', 'learning', 'test1_model.pl')
        examples = root_path('problog', 'learning', 'test1_examples.pl')

        out = subprocess_check_output([sys.executable, problogcli, 'lfi', model, examples])
        outline = out.strip().split()

        self.assertGreater(int(outline[-1]), 2)

        weights = [float(outline[i].strip('[],')) for i in (1, 2, 3, 4)]

        self.assertAlmostEqual(weights[0], 1.0 / 3)
        self.assertAlmostEqual(weights[2], 1.0)
        self.assertAlmostEqual(weights[3], 0.0)
