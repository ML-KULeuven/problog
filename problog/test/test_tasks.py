import unittest

from problog.logic import Term
from problog.tasks import dtproblog


class TestTasks(unittest.TestCase):
    def check_if_output_equals(self, dt_file, expected_output):
        real_file_name = "./../../test/dtproblog/" + dt_file
        result = dtproblog.main([real_file_name])
        print("RESULT", dt_file, result)
        self.assertTrue(result[0])
        self.assertEqual(expected_output, result[1])

    def test_dt_problog(self):
        self.check_if_output_equals("ex1.pl", ({Term("b"): 1}, 1.7999999999999998, {"eval": 2}))
        self.check_if_output_equals("ex2.pl", ({Term("c"): 1}, 1.7999999999999998, {"eval": 2}))
        self.check_if_output_equals("ex3.pl", ({Term("a"): 1}, 1.7999999999999998, {"eval": 2}))
        self.check_if_output_equals("ex4.pl", ({Term("b"): 1, Term("c"): 1}, 3, {"eval": 4}))
        self.check_if_output_equals("ex5.pl", ({Term("a"): 1, Term("c"): 1}, 1.7999999999999998, {"eval": 4}))


if __name__ == '__main__':
    unittest.main()
