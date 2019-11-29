import os
import unittest
import pytest
from pathlib import Path

from problog.logic import Constant, Term, Not
from problog.tasks import map, constraint, explain, time1, shell, bayesnet, mpe

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

    # def test_mpe_semiring_some_heads(self):
    #     file_name = test_folder / "tasks" / "some_heads.pl"
    #     result = mpe.main([str(file_name), "--use-semiring"])
    #     self.assertTrue(result[0])
    #     self.assertEqual(0.3, result[1][0])
    #     self.assertEqual([Term("someHeads")], result[1][1])

    def test_mpe_maxsat_some_heads(self):
        file_name = test_folder / "tasks" / "some_heads.pl"
        result = mpe.main([str(file_name), "--use-maxsat"])
        self.assertTrue(result[0])
        self.assertAlmostEqual(0.3, result[1][0], places=6)
        self.assertEqual([Term("someHeads")], result[1][1])

    def test_mpe_maxsat_pgraph(self):
        file_name = test_folder / "tasks" / "pgraph.pl"
        result = mpe.main([str(file_name), "--use-maxsat"])
        self.assertTrue(result[0])
        print("result", result)
        self.assertAlmostEqual(0.3, result[1][0], delta=1e6)
        self.assertEqual(
            {
                Term("edge", Constant(1), Constant(2)),
                Term("edge", Constant(2), Constant(5)),
                Not("\\+", Term("edge", Constant(1), Constant(3))),
                Not("\\+", Term("edge", Constant(3), Constant(4))),
                Term("edge", Constant(4), Constant(5)),
                Term("edge", Constant(2), Constant(6)),
                Not("\\+", Term("edge", Constant(5), Constant(6))),
            },
            set(result[1][1]),
        )

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

    def test_bn(self):
        file_name = test_folder / "tasks" / "some_heads.pl"
        result = bayesnet.main([str(file_name)])
        print("result", result)
        self.assertTrue(result[0])
        self.assertEqual(
            "Factor (c0) = 0, 1\n(): [0.5, 0.5]"
            "\n\nOrCPT heads1 [0,1] -- c0"
            "\n('c0', 1)"
            "\n\nFactor (c1) = 0, 1"
            "\n(): [0.4, 0.6]"
            "\n\nOrCPT heads2 [0,1] -- c1"
            "\n('c1', 1)"
            "\n\nFactor (c2 | heads1) = 0, 1"
            "\n(False,): [1.0, 0.0]\n(True,): [0.0, 1.0]"
            "\n\nOrCPT someHeads [0,1] -- c2,c3\n('c2', 1)\n('c3', 1)"
            "\n\nFactor (c3 | heads2) = 0, 1"
            "\n(False,): [1.0, 0.0]\n(True,): [0.0, 1.0]\n",
            result[1],
        )
