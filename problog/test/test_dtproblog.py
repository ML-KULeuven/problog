import os
import unittest
from pathlib import Path

from problog.logic import Term, Constant
from problog.tasks import dtproblog

dirname = os.path.dirname(__file__)
dt_problog_test_folder = Path(dirname, "./../../test/dtproblog/")


class TestDTProblog(unittest.TestCase):
    def dt_problog_check_if_output_equals(
        self, dt_file, expected_choices, expected_score, local=False
    ):
        real_file_name = dt_problog_test_folder / dt_file
        file_exists = real_file_name.exists()
        if not file_exists:
            self.fail(
                "File "
                + str(real_file_name)
                + " was not found. Maybe this is a pathing issue?"
            )
        else:
            arguments = [str(real_file_name)]
            if local:
                arguments.append("-s")
                arguments.append("local")

            result = dtproblog.main(arguments)
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

    def test_dt_problog_simple(self):
        self.dt_problog_check_if_output_equals("ex1.pl", {Term("b"): 1}, 1.8)
        self.dt_problog_check_if_output_equals("ex2.pl", {Term("c"): 1}, 1.8)
        self.dt_problog_check_if_output_equals("ex3.pl", {Term("a"): 1}, 1.8)
        self.dt_problog_check_if_output_equals(
            "ex4.pl", {Term("b"): 1, Term("c"): 1}, 3
        )
        self.dt_problog_check_if_output_equals(
            "ex5.pl", {Term("a"): 1, Term("c"): 1}, 1.8
        )

    def test_dt_problog_local(self):
        self.dt_problog_check_if_output_equals(
            "ex5.pl", {Term("a"): 0, Term("c"): 0}, 0, local=True
        )
        # Takes too long:
        # self.dt_problog_check_if_output_equals(
        #     "umbrella_poole.pl",
        #     {
        #         Term("umbrella", Term("sunny")): 0,
        #         Term("umbrella", Term("cloudy")): 0,
        #         Term("umbrella", Term("rainy")): 1,
        #     },
        #     77, local=True
        # )

    def test_dtproblog_mutex(self):
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

    def test_dtproblog_umbrella(self):
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

    # Takes too long:

    # def test_dtproblog_long_test(self):
    #     self.dt_problog_check_if_output_equals(
    #         "viralmarketing.pl",
    #         {
    #             Term("marketed", Term("angelika")): 0,
    #             Term("marketed", Term("bernd")): 0,
    #             Term("marketed", Term("guy")): 1,
    #             Term("marketed", Term("ingo")): 1,
    #             Term("marketed", Term("kurt")): 0,
    #             Term("marketed", Term("laura")): 0,
    #             Term("marketed", Term("martijn")): 1,
    #             Term("marketed", Term("theo")): 1,
    #         },
    #         3.210966333135799,
    #         local=True
    #     )

    def test_dtproblog_other(self):
        self.dt_problog_check_if_output_equals(
            "var_util.pl",
            {
                Term("bet", Term("w1"), Constant(5)): 1,
                Term("bet", Term("w2"), Constant(7)): 0,
                Term("bet", Term("w3"), Constant(10)): 0,
            },
            4,
        )
        self.dt_problog_check_if_output_equals(
            "weather_poole_alt.pl",
            {
                Term("decide_u", Term("rainy")): 1,
                Term("decide_u", Term("cloudy")): 0,
                Term("decide_u", Term("sunny")): 0,
            },
            77,
        )
        # self.dt_problog_check_if_output_equals("winning.pl", {}, 0)
        self.dt_problog_check_if_output_equals(
            "winning_undecided.pl", {Term("play1"): 1, Term("play2"): 1}, 34.5
        )

    if __name__ == "__main__":
        unittest.main()
