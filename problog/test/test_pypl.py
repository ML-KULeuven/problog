import unittest

from problog.logic import Term
from problog.pypl import py2pl, pl2py


class TestPypl(unittest.TestCase):
    def test_list_double_conversion(self):
        test_list = [1, 2, 3, "a", (4, 5, "c"), 7.0, [8, "d"], Term("h")]
        res_list = pl2py(py2pl(test_list))
        self.assertEqual(test_list, res_list)

    def test_tuple_double_conversion(self):
        test_tuple = (1, 2, 3, "a", [4, 5, "c"], 7.0, [8, "d"])
        res_tuple = pl2py(py2pl(test_tuple))
        self.assertEqual(test_tuple, res_tuple)

    def test_empty_list_double_conversion(self):
        test_list = []
        res_list = pl2py(py2pl(test_list))
        self.assertEqual(test_list, res_list)

    def test_empty_tuple_double_conversion(self):
        test_tuple = ()
        res_tuple = pl2py(py2pl(test_tuple))
        self.assertEqual(test_tuple, res_tuple)

    def test_unsupported_types(self):
        test_dict = {"a": 1}
        with self.assertRaises(ValueError):
            py2pl(test_dict)

    def test_list_at_end_of_comma_double_conversion(self):
        test_value = (1, [])
        res_value = pl2py(py2pl(test_value))
        self.assertEqual(test_value, res_value)
