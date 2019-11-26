import unittest

from problog.tasks import dtproblog


class TestTasks(unittest.TestCase):
    def check_if_output_equals(self, dt_file, check_output):
        real_file_name = "./../../test/dtproblog/" + dt_file

        def result_handler(result_tuple, ignore):
            if len(result_tuple) > 0:
                self.assertEqual(True, result_tuple[0])
            if len(result_tuple) > 1 and check_output:
                check_output(result_tuple[1])

        dtproblog.main([real_file_name], result_handler=result_handler)

    def test_dt_problog(self):
        self.check_if_output_equals("ex1.pl", lambda result: self.assertTrue(result[1], 1.7999999999999998))


if __name__ == '__main__':
    unittest.main()
