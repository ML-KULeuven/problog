import os
import unittest
from pathlib import Path

from problog.logic import Constant, Term
from problog.tasks import map

dirname = os.path.dirname(__file__)
test_folder = Path(dirname, "./../../test/")


class TestTasks(unittest.TestCase):
    def test_map(self):
        file_name = test_folder / "map_probabilistic_graph.pl"
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
