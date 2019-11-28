import os
import random
import unittest
from pathlib import Path

from problog.logic import Term

from problog.program import PrologString
from problog.tasks import sample

dirname = os.path.dirname(__file__)
test_folder = Path(dirname, "./../../test/sample")


class TestSampleTask(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(12345)

    def get_samples(self, file_name, num_samples=1, **kwargs):
        file_name = test_folder / file_name
        model = PrologString(file_name.read_text())
        sample_generator = sample.sample(model, n=num_samples, format="dict", **kwargs)
        result_list = list(sample_generator)
        self.assertEqual(
            num_samples,
            len(result_list),
            "Was not able to sample the right number of samples from " + str(file_name),
        )
        return result_list

    @staticmethod
    def count_number_of_atoms(sample_list, term):
        return len(
            [
                sampled_dict
                for sampled_dict in sample_list
                if term in sampled_dict and sampled_dict[term] is True
            ]
        )

    def test_task_some_heads(self):
        file_name = test_folder / "some_heads.pl"
        result = sample.main([str(file_name), "-n", str(1)])
        success, data = result
        if not success:
            self.fail("Could not successfully sample" + str(file_name))

    def test_some_heads_distribution(self):
        samples = self.get_samples("some_heads.pl", num_samples=1000)
        self.assertAlmostEqual(
            800, self.count_number_of_atoms(samples, Term("someHeads")), delta=50
        )

    def test_some_heads_evidence_distribution(self):
        samples = self.get_samples("some_heads_evidence.pl", num_samples=1000)
        number_of_some_heads = self.count_number_of_atoms(samples, Term("someHeads"))
        self.assertAlmostEqual(600, number_of_some_heads, delta=50)
