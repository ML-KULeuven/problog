import unittest
from collections import OrderedDict

from problog.logic import Term, Constant
from problog.tasks import dtproblog


class TestTasks(unittest.TestCase):
    def dt_problog_check_if_output_equals(self, dt_file, expected_choices, expected_score):
        real_file_name = "./../../test/dtproblog/" + dt_file
        result = dtproblog.main([real_file_name])
        print("RESULT", dt_file, result)
        choices, score, stats = result[1]
        self.assertEqual(expected_choices, choices)
        self.assertAlmostEqual(expected_score, score, delta=1e-6)

    def test_simple_dt_problog(self):
        self.dt_problog_check_if_output_equals(
            "ex1.pl", {Term("b"): 1}, 1.8
        )
        self.dt_problog_check_if_output_equals(
            "ex2.pl", {Term("c"): 1}, 1.8
        )
        self.dt_problog_check_if_output_equals(
            "ex3.pl", {Term("a"): 1}, 1.8
        )
        self.dt_problog_check_if_output_equals(
            "ex4.pl", {Term("b"): 1, Term("c"): 1}, 3
        )
        self.dt_problog_check_if_output_equals(
            "ex5.pl", {Term("a"): 1, Term("c"): 1}, 1.8
        )

    def test_harder_dtproblog(self):
        self.dt_problog_check_if_output_equals(
            "mut_exl.pl",
            {
                Term("play", Constant(1)): 0,
                Term("play", Constant(2)): 1,
                Term("bet", Constant(1), Term("heads")): 0,
                Term("bet", Constant(2), Term("heads")): 0,
                Term("bet", Constant(1), Term("tails")): 1,
                Term("bet", Constant(2), Term("tails")): 1,
            },
            1.5
        )
        self.dt_problog_check_if_output_equals(
            "mut_exl2.pl",
            {
                Term("play"): 0,
                Term("bet", Term("heads")): 0,
                Term("bet", Term("tails")): 1
            },
            0
        )
        self.dt_problog_check_if_output_equals(
            "umbrella.pl",
            {
                Term("umbrella"): 1,
                Term("raincoat"): 0,
            },
            43
        )
        self.dt_problog_check_if_output_equals(
            "umbrella_poole.pl",
            {
                Term("umbrella", Term("sunny")): 0,
                Term("umbrella", Term("cloudy")): 0,
                Term("umbrella", Term("rainy")): 1
            },
            77
        )

    if __name__ == "__main__":
        unittest.main()
