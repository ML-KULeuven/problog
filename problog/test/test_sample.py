import os
import random
import unittest
from pathlib import Path

from problog.program import PrologString

from problog.tasks import sample

dirname = os.path.dirname(__file__)
test_folder = Path(dirname, "./../../test/sample")


class TestSampleTask(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(123)

    def get_samples(self, file_name, num_samples=1):
        file_name = test_folder / file_name
        model = PrologString(file_name.read_text())
        sample_generator = sample.sample(model, n=num_samples)
        return list(sample_generator)

    def test_some_heads_through_task_main(self):
        file_name = test_folder / "some_heads.pl"
        result = sample.main([str(file_name), "-n", str(1)])
        success, data = result
        if not success:
            self.fail("Could not successfully sample" + str(file_name))

    def test_some_heads(self):
        samples = self.get_samples("some_heads.pl", num_samples=100)
        print("Samples", samples)

        # success = result[0]
        # if not success:
        #     self.fail("Failed executing MAP on " + str(file_name))
        # else:
        #     choices, score, stats = result[1]
        #     self.assertEqual(
        #         {
        #             Term("edge", Constant(1), Constant(2)): 1,
        #             Term("edge", Constant(1), Constant(3)): 0,
        #         },
        #         choices,
        #     )
        #     self.assertAlmostEqual(1.878144, score, places=5)
