import os.path
import sys
import unittest
from pathlib import Path

from problog.logic import Term, Constant
from problog.tasks import dtproblog

if __name__ == "__main__":
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    )

dt_problog_test_folder = Path("./test/dtproblog/")


class TestTasks(unittest.TestCase):
    def dt_problog_check_if_output_equals(
        self, dt_file, expected_choices, expected_score
    ):
        real_file_name = dt_problog_test_folder / dt_file
        file_exists = real_file_name.exists()
        self.assertTrue(
            file_exists,
            msg="File "
            + str(real_file_name)
            + " was not found. Maybe this is a pathing issue?",
        )
        if file_exists:
            result = dtproblog.main([str(real_file_name)])
            if result[0]:
                choices, score, stats = result[1]
                self.assertEqual(expected_choices, choices)
                self.assertAlmostEqual(expected_score, score, delta=1e-6)
            else:
                self.fail(
                    "An error occured was returned when executing "
                    + dt_file
                    + "\nError "
                    + str(result[1])
                )

    def test_simple_dt_problog(self):
        self.dt_problog_check_if_output_equals("ex1.pl", {Term("b"): 1}, 1.8)
        self.dt_problog_check_if_output_equals("ex2.pl", {Term("c"): 1}, 1.8)
        self.dt_problog_check_if_output_equals("ex3.pl", {Term("a"): 1}, 1.8)
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
            1.5,
        )
        self.dt_problog_check_if_output_equals(
            "mut_exl2.pl",
            {
                Term("play"): 0,
                Term("bet", Term("heads")): 0,
                Term("bet", Term("tails")): 1,
            },
            0,
        )
        self.dt_problog_check_if_output_equals(
            "umbrella.pl", {Term("umbrella"): 1, Term("raincoat"): 0}, 43
        )
        self.dt_problog_check_if_output_equals(
            "umbrella_poole.pl",
            {
                Term("umbrella", Term("sunny")): 0,
                Term("umbrella", Term("cloudy")): 0,
                Term("umbrella", Term("rainy")): 1,
            },
            77,
        )
        self.dt_problog_check_if_output_equals(
            "var_util.pl",
            {
                Term("bet", Term("w1"), Constant(5)): 1,
                Term("bet", Term("w2"), Constant(7)): 0,
                Term("bet", Term("w3"), Constant(10)): 0,
            },
            4,
        )

    if __name__ == "__main__":
        unittest.main()
