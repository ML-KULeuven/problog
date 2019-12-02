import os
import unittest
from pathlib import Path

from problog.logic import Constant, Term, Not
from problog.tasks import map, explain, time1, bayesnet, mpe

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

    # BN
    def check_bn(self, file_name, expected):
        result = bayesnet.main([str(file_name)])
        print("result", result)
        self.assertTrue(result[0])
        self.equal_bn_result(expected, result[1])

    @staticmethod
    def normalise_bn_output(bn_string):
        return bn_string.strip().replace("\n\n", "\n").replace(" ", "")

    def equal_bn_result(self, expected, actual):
        return self.assertEqual(
            self.normalise_bn_output(expected), self.normalise_bn_output(actual)
        )

    def test_bn(self):
        file_name = test_folder / "bn" / "single_atom_ad.pl"
        self.check_bn(
            file_name,
            "Factor (c0) =0, 1"
            "\n(): [0.4, 0.6]"
            "\n\nOrCPT someHeads [0,1] -- c0"
            "\n('c0', 1)",
        )

    def test_bn_some_heads(self):
        file_name = test_folder / "tasks" / "some_heads.pl"
        self.check_bn(
            file_name,
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
        )

    def test_bn_some_heads_or(self):
        file_name = test_folder / "tasks" / "some_heads_or.pl"
        self.check_bn(
            file_name,
            "Factor (c0) = 0, 1\n(): [0.5, 0.5]"
            "\n\nOrCPT heads1 [0,1] -- c0\n('c0', 1)"
            "\n\nFactor (c1) = 0, 1\n(): [0.4, 0.6]"
            "\n\nOrCPT heads2 [0,1] -- c1\n('c1', 1)"
            "\n\nFactor (c2 | heads2) = 0\n(False,): [1.0]\n(True,): [1.0]"
            "\n\nFactor (c3 | \\+heads1, \\+heads2) = 0\n(False,): [1.0]\n(True,): [1.0]"
            "\n\nFactor (c4 | heads1) = 0\n(False,): [1.0]\n(True,): [1.0]"
            "\n\nFactor (c5 | heads2) = 0\n(False,): [1.0]\n(True,): [1.0]"
            "\n\nFactor (c6 | heads1) = 0\n(False,): [1.0]\n(True,): [1.0]"
            "\n\nFactor (c7 | heads1, heads2) = 0, 1\n(False, False): [1.0, 0.0]"
            "\n(False, True): [1.0, 0.0]\n(True, False): [1.0, 0.0]\n(True, True): [0.0, 1.0]"
            "\n\nOrCPT someHeads [0,1] -- c7\n('c7', 1)\n",
        )

    def test_bn_pgraph(self):
        file_name = test_folder / "tasks" / "pgraph.pl"
        self.check_bn(
            file_name,
            "Factor (c0) = 0, 1"
            "\n(): [0.4, 0.6]"
            "\n\nOrCPT edge(1,2) [0,1] -- c0"
            "\n('c0', 1)"
            "\n\nFactor (c1) = 0, 1"
            "\n(): [0.6, 0.4]"
            "\n\nOrCPT edge(2,5) [0,1] -- c1"
            "\n('c1', 1)"
            "\n\nFactor (c2) = 0, 1"
            "\n(): [0.9, 0.1]"
            "\n\nOrCPT edge(1,3) [0,1] -- c2"
            "\n('c2', 1)\n\nFactor (c3) = 0, 1"
            "\n(): [0.7, 0.3]"
            "\n\nOrCPT edge(3,4) [0,1] -- c3\n('c3', 1)"
            "\n\nFactor (c4) = 0, 1\n(): [0.19999999999999996, 0.8]"
            "\n\nOrCPT edge(4,5) [0,1] -- c4\n('c4', 1)"
            "\n\nFactor (c5 | edge(3,4), edge(4,5)) = 0, 1\n(False, False): [1.0, 0.0]"
            "\n(False, True): [1.0, 0.0]\n(True, False): [1.0, 0.0]\n(True, True): [0.0, 1.0]"
            "\n\nOrCPT path(3,5) [0,1] -- c5\n('c5', 1)\n\nFactor (c6 | edge(1,2), edge(2,5)) = 0, 1"
            "\n(False, False): [1.0, 0.0]\n(False, True): [1.0, 0.0]\n(True, False): [1.0, 0.0]"
            "\n(True, True): [0.0, 1.0]\n\nOrCPT path(1,5) [0,1] -- c6,c7\n('c6', 1)\n('c7', 1)"
            "\n\nFactor (c7 | edge(1,3), path(3,5)) = 0, 1\n(False, False): [1.0, 0.0]\n(False, True): [1.0, 0.0]"
            "\n(True, False): [1.0, 0.0]\n(True, True): [0.0, 1.0]\n\nFactor (c8) = 0, 1\n(): [0.7, 0.3]"
            "\n\nOrCPT edge(2,6) [0,1] -- c8\n('c8', 1)\n\nFactor (c9) = 0, 1\n(): [0.8, 0.2]"
            "\n\nOrCPT edge(5,6) [0,1] -- c9\n('c9', 1)"
            "\n\nFactor (c10 | edge(2,6)) = 0, 1\n(False,): [1.0, 0.0]\n(True,): [0.0, 1.0]"
            "\n\nOrCPT path(2,6) [0,1] -- c10,c11\n('c10', 1)\n('c11', 1)"
            "\n\nFactor (c11 | edge(2,5), edge(5,6)) = 0, 1\n(False, False): [1.0, 0.0]"
            "\n(False, True): [1.0, 0.0]\n(True, False): [1.0, 0.0]\n(True, True): [0.0, 1.0]"
            "\n\nFactor (c12 | edge(4,5), edge(5,6)) = 0, 1"
            "\n(False, False): [1.0, 0.0]\n(False, True): [1.0, 0.0]"
            "\n(True, False): [1.0, 0.0]\n(True, True): [0.0, 1.0]"
            "\n\nOrCPT path(4,6) [0,1] -- c12\n('c12', 1)"
            "\n\nFactor (c13 | edge(3,4), path(4,6)) = 0, 1\n(False, False): [1.0, 0.0]"
            "\n(False, True): [1.0, 0.0]\n(True, False): [1.0, 0.0]\n(True, True): [0.0, 1.0]"
            "\n\nOrCPT path(3,6) [0,1] -- c13\n('c13', 1)"
            "\n\nFactor (c14 | edge(1,2), path(2,6)) = 0, 1"
            "\n(False, False): [1.0, 0.0]\n(False, True): [1.0, 0.0]"
            "\n(True, False): [1.0, 0.0]\n(True, True): [0.0, 1.0]"
            "\n\nOrCPT path(1,6) [0,1] -- c14,c15\n('c14', 1)\n('c15', 1)"
            "\n\nFactor (c15 | edge(1,3), path(3,6)) = 0, 1\n(False, False): [1.0, 0.0]"
            "\n(False, True): [1.0, 0.0]\n(True, False): [1.0, 0.0]\n(True, True): [0.0, 1.0]\n",
        )
