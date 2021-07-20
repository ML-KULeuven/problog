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

    def get_samples(self, file_name, num_samples=1, propagate_evidence=False):
        file_name = test_folder / file_name
        model = PrologString(file_name.read_text())
        sample_generator = sample.sample(
            model, n=num_samples, format="dict", propagate_evidence=propagate_evidence
        )
        result_list = list(sample_generator)
        self.assertEqual(
            num_samples,
            len(result_list),
            "Was not able to sample the right number of samples from " + str(file_name),
        )
        return result_list

    @staticmethod
    def estimate(file_name, num_samples=1, propagate_evidence=False):
        file_name = test_folder / file_name
        model = PrologString(file_name.read_text())
        return sample.estimate(
            model, n=num_samples, format="dict", propagate_evidence=propagate_evidence
        )

    @staticmethod
    def count_number_of_atoms(sample_list, term):
        return len(
            [
                sampled_dict
                for sampled_dict in sample_list
                if term in sampled_dict and sampled_dict[term] is True
            ]
        )

    def test_some_heads_distribution(self):
        samples = self.get_samples("some_heads.pl", num_samples=1000)
        self.assertAlmostEqual(
            800, self.count_number_of_atoms(samples, Term("someHeads")), delta=50
        )

    def test_some_heads_evidence(self):
        samples = self.get_samples("some_heads_evidence.pl", num_samples=100)
        number_of_some_heads = self.count_number_of_atoms(samples, Term("someHeads"))
        self.assertAlmostEqual(60, number_of_some_heads, delta=10)

    def test_some_heads_propagate_evidence(self):
        samples = self.get_samples(
            "some_heads_evidence.pl", num_samples=1000, propagate_evidence=True
        )
        number_of_some_heads = self.count_number_of_atoms(samples, Term("someHeads"))
        self.assertAlmostEqual(600, number_of_some_heads, delta=50)

    def test_some_heads_estimate(self):
        with_propagation = self.estimate(
            "some_heads.pl", num_samples=1000, propagate_evidence=False
        )
        self.assertAlmostEqual(0.8, with_propagation[Term("someHeads")], delta=0.05)
        without_propagation = self.estimate(
            "some_heads.pl", num_samples=1000, propagate_evidence=True
        )
        self.assertAlmostEqual(0.8, without_propagation[Term("someHeads")], delta=0.05)

    def test_some_heads_previous(self):
        samples = self.get_samples(
            "some_heads_previous.pl", num_samples=100, propagate_evidence=True
        )
        previous = False
        for s in samples:
            self.assertEqual(
                s[Term("previous", Term("someHeads"), Term("false"))], previous
            )
            previous = s[Term("someHeads")]

    def test_maze_generator(self):
        samples = self.get_samples("maze.pl", num_samples=1)
        self.assertEqual(1, len(samples))
        connected_predicates = [
            p for p in samples[0].keys() if "connected" in p.functor
        ]
        self.assertTrue(25 < len(connected_predicates))

    def test_heights(self):
        samples = self.get_samples("heights.pl", num_samples=10)
        self.assertEqual(10, len(samples))
        for s in samples:
            self.assertAlmostEqual(
                360, float(s[Term("twice_height", Term("harry"))]), delta=60
            )
        print(samples)

    # Tasks

    def test_some_heads_task(self):
        file_name = test_folder / "some_heads.pl"
        result = sample.main([str(file_name), "-n", str(1)])
        success, data = result
        if not success:
            self.fail("Could not successfully sample" + str(file_name))

    def test_some_heads_task_estimate(self):
        file_name = test_folder / "some_heads.pl"
        sample.main([str(file_name), "-n", str(100), "--estimate"])

    def test_some_heads_task_propagate_evidence(self):
        file_name = test_folder / "some_heads_evidence.pl"
        result = sample.main([str(file_name), "-n", str(10), "--propagate-evidence"])
        success, data = result
        if not success:
            self.fail("Could not successfully sample" + str(file_name))
