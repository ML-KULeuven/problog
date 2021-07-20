import unittest

from problog.logic import And, AnnotatedDisjunction, Or, Clause, Not, Term, Var


class TestLogic(unittest.TestCase):
    def test_and(self):
        c1 = And(Term("a"), Term("b"))
        c2 = Term("a") & Term("b")
        c3 = Term("b") & Term("c")
        c4 = Term("a") & Term("b", None)
        c5 = Term("a") & Term("b") & Term("c")
        c6 = And(Term("a"), And(Term("b"), Term("c")))
        c7 = And(And(Term("a"), Term("b")), Term("c"))
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 == c3)
        self.assertFalse(c1 == c4)
        self.assertFalse(c2 == c4)
        self.assertTrue(c5 == c6)
        self.assertFalse(c5 == c7)
        self.assertFalse(c6 == c7)

    def test_or(self):
        c1 = Or(Term("a"), Term("b"))
        c2 = Term("a") | Term("b")
        c3 = Term("b") | Term("c")
        c4 = Term("a") | Term("b", None)
        c5 = Term("a") | Term("b") | Term("c")
        c6 = Or(Term("a"), Or(Term("b"), Term("c")))
        c7 = Or(Or(Term("a"), Term("b")), Term("c"))
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 == c3)
        self.assertFalse(c1 == c4)
        self.assertFalse(c2 == c4)
        self.assertTrue(c5 == c6)
        self.assertFalse(c5 == c7)
        self.assertFalse(c6 == c7)

    def test_and_or(self):
        c1 = Term("a") & Term("b") | Term("c")
        c2 = Term.from_string("(a,b);c.")
        c3 = Term("a") & (Term("b") | Term("c"))
        c4 = (Term("b") | Term("c")) & Term("a")
        self.assertTrue(c1 == c2)
        self.assertFalse(c2 == c3)
        self.assertFalse(c1 == c3)
        self.assertFalse(c3 == c4)  # Different because of unification

    def test_not(self):
        c1 = Not("\\+", Term("a"))
        c2 = ~Term("a")
        c3 = Not("not", Term("a"))
        self.assertTrue(c1 == c2)
        self.assertTrue(c1 == c3)

        c4 = And(Term("a"), Not("not", Term("b")))
        c5 = And(Term("a"), Not("\\+", Term("b")))
        self.assertTrue(c4 == c5)

    def test_lshift(self):
        c1 = Clause(Term("a"), Term("b"))
        c2 = Term("a") << Term("b")
        c3 = Term("b") << Term("c")
        c4 = Term("a") << Term("b", None)
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 == c3)
        self.assertFalse(c1 == c4)
        self.assertFalse(c2 == c4)

    def test_clause_equality(self):
        c1 = Clause(Term("a", None), Term("b", None))
        c2 = Clause(Term("a", None), Term("b", None))
        c3 = Clause(Term("a", Term("c", None)), Term("b", None))
        c4 = Clause(Term("a", Term("c", None)), Term("b", None))
        c5 = Clause(Term("a", None), Term("b", None))
        c6 = Clause(Term("c", None), Term("b", None))
        c7 = Clause(Term("a", Var("_")), Term("b", None))
        c8 = Clause(Term("a", Var("X1")), Term("b", Var("X1")))
        c9 = Clause(Term("a", Var("X2")), Term("b", Var("X2")))
        c10 = Clause(Term("a", -1), Term("b", -1))
        c11 = Clause(Term("a", -2), Term("b", -2))
        c12 = Clause(Term("a", -1), Term("b", -2))
        self.assertTrue(c1 == c2)
        self.assertTrue(c3 == c4)
        self.assertFalse(c5 == c6)
        self.assertFalse(c7 == c1)
        self.assertFalse(c8 == c9)
        self.assertFalse(c1 == c10)
        self.assertFalse(c8 == c10)
        self.assertFalse(c1 == c11)
        self.assertFalse(c1 == c12)
        self.assertFalse(c10 == c11)
        self.assertFalse(c10 == c12)

    def test_from_string(self):
        c1 = Term.from_string("a.")
        c2 = Term("a")
        c3 = Term.from_string("a:-b.")
        c4 = Clause(Term("a"), Term("b"))
        c5 = Term.from_string("a")
        self.assertTrue(c1 == c2)
        self.assertTrue(c3 == c4)
        self.assertTrue(c1 == c5)
        self.assertTrue(c2 == c5)

    def test_from_list(self):
        c1 = Term("a") | Term("b")
        c2 = Or.from_list([Term("a"), Term("b")])
        c3 = Term("a") | (Term("b") | Term("c"))
        c4 = Or.from_list([Term("a"), Term("b"), Term("c")])
        c5 = Or.from_list([])
        c6 = Term("fail")
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 == c4)
        self.assertTrue(c3 == c4)
        self.assertTrue(c5 == c6)

        c1 = Term("a") & Term("b")
        c2 = And.from_list([Term("a"), Term("b")])
        c3 = Term("a") & (Term("b") & Term("c"))
        c4 = And.from_list([Term("a"), Term("b"), Term("c")])
        c5 = And.from_list([])
        c6 = Term("true")
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 == c4)
        self.assertTrue(c3 == c4)
        self.assertTrue(c5 == c6)

    def test_to_list(self):
        term_list = [Term("a"), Term("b"), Term("c")]
        c1 = And.from_list(term_list)
        self.assertTrue(term_list == c1.to_list())
        self.assertTrue([Term("a"), Term("b"), Term("c")] == c1.to_list())

        c2 = Or.from_list(term_list)
        self.assertTrue(term_list == c2.to_list())
        self.assertTrue([Term("a"), Term("b"), Term("c")] == c2.to_list())

    def test_annotated_disjunction(self):
        c1 = AnnotatedDisjunction([Term("a")], None)
        c2 = AnnotatedDisjunction([Term("a")], None)
        c3 = AnnotatedDisjunction([Term("b")], None)
        c4 = AnnotatedDisjunction([Term("a"), Term("b")], Term("c"))
        c5 = AnnotatedDisjunction([Term("a"), Term("b")], Term("c"))
        c6 = AnnotatedDisjunction([Term("a"), Term("b")], Term("d"))
        c7 = AnnotatedDisjunction([Term("a"), Term("b")], None)
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 == c3)
        self.assertTrue(c4 == c5)
        self.assertFalse(c4 == c6)
        self.assertFalse(c4 == c7)
