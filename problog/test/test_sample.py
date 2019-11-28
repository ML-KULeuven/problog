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
        random.seed(12345)

    def get_samples(self, file_name, num_samples=1):
        file_name = test_folder / file_name
        model = PrologString(file_name.read_text())
        sample_generator = sample.sample(model, n=num_samples)
        result_list = list(sample_generator)
        self.assertEqual(
            num_samples,
            len(result_list),
            "Was not able to sample the right number of samples from " + str(file_name),
        )
        return result_list

    @staticmethod
    def count_number_of_atoms(sample_list, atom_name):
        return len([s for s in sample_list if (atom_name + ".") in s])

    def test_some_heads_through_task_main(self):
        file_name = test_folder / "some_heads.pl"
        result = sample.main([str(file_name), "-n", str(1)])
        success, data = result
        if not success:
            self.fail("Could not successfully sample" + str(file_name))

    def test_some_heads(self):
        samples = self.get_samples("some_heads.pl", num_samples=1000)
        self.assertAlmostEqual(
            800, self.count_number_of_atoms(samples, "someHeads"), delta=50
        )
