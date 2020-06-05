import unittest

import problog
from problog.logic import Clause, Term


class TestEquality(unittest.TestCase):
    def test_clause_equality(self):
        c1 = Clause(Term("a"), Term("b"))
        c2 = Clause(Term("c"), Term("b"))
        c3 = Clause(Term("c"), Term("b"))
        self.assertTrue(c2 == c3)
        self.assertFalse(c1 == c2)
        self.assertFalse(c1 == c3)
