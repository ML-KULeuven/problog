import os
import unittest
import pytest
from pathlib import Path

from problog.logic import Constant, Term
from problog.tasks import map, constraint, explain, time1, shell, bayesnet

dirname = os.path.dirname(__file__)
test_folder = Path(dirname, "./../../test/")


class TestTasks(unittest.TestCase):
    def test_map(self):
        file_name = test_folder / "tasks" / "map_probabilistic_graph.pl"
        result = map.main([str(file_name)])
        success = result[0]
        if not success:
            self.fail("Failed executing MAP on " + str(file_name))
        else:
            choices, score, stats = result[1]
            self.assertEqual(
                {
                    Term("edge", Constant(1), Constant(2)): 1,
                    Term("edge", Constant(1), Constant(3)): 0,
                },
                choices,
            )
            self.assertAlmostEqual(1.878144, score, places=5)

    def test_explain_trivial(self):
        file_name = test_folder / "tasks" / "map_probabilistic_graph.pl"
        result = explain.main([str(file_name)])

        self.assertTrue(result["SUCCESS"])
        results = result["results"]
        self.assertAlmostEqual(
            0.6, results[Term("edge", Constant(1), Constant(2))], delta=1e6
        )
        self.assertAlmostEqual(
            0.1, results[Term("edge", Constant(1), Constant(3))], delta=1e6
        )

    def test_explain_some_heads(self):
        file_name = test_folder / "tasks" / "some_heads.pl"
        result = explain.main([str(file_name)])

        # Check if successful
        self.assertTrue(result["SUCCESS"])

        # Test proofs
        proofs = result["proofs"]
        self.assertEqual("someHeads :- heads2.  % P=0.6", proofs[0])
        self.assertEqual("someHeads :- heads1, \\+heads2.  % P=0.2", proofs[1])

        # Test result
        results = result["results"]
        self.assertAlmostEqual(0.8, results[Term("someHeads")], delta=1e6)

    def test_time(self):
        file_name = test_folder / "tasks" / "some_heads.pl"
        result = time1.main([str(file_name)])
        self.assertEquals(6, len(result))

