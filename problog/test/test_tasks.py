import unittest
from collections import OrderedDict

from problog.logic import Term
from problog.tasks import dtproblog


class TestTasks(unittest.TestCase):
    def dt_problog_check_if_output_equals(self, dt_file, expected_choices, expected_score, expected_stats):
        real_file_name = "./../../test/dtproblog/" + dt_file
        result = dtproblog.main([real_file_name])
        print("RESULT", dt_file, result)
        choices, score, stats = result[1]
        self.assertEqual(OrderedDict(expected_choices), OrderedDict(choices))
        self.assertAlmostEqual(expected_score, score, delta=1e-6)
        self.assertEqual(expected_stats, stats)

    def test_simple_dt_problog(self):
        self.dt_problog_check_if_output_equals(
            "ex1.pl", {Term("b"): 1}, 1.8, {"eval": 2}
        )
        self.dt_problog_check_if_output_equals(
            "ex2.pl", {Term("c"): 1}, 1.8, {"eval": 2}
        )
        self.dt_problog_check_if_output_equals(
            "ex3.pl", {Term("a"): 1}, 1.8, {"eval": 2}
        )
        self.dt_problog_check_if_output_equals(
            "ex4.pl", {Term("b"): 1, Term("c"): 1}, 3, {"eval": 4}
        )
        self.dt_problog_check_if_output_equals(
            "ex5.pl", {Term("a"): 1, Term("c"): 1}, 1.8, {"eval": 4}
        )


    if __name__ == "__main__":
        unittest.main()
