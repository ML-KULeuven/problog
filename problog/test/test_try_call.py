import unittest

from problog import get_evaluatable
from problog.program import PrologString


class TestTryCall(unittest.TestCase):
    def test_try_call_existing_fact(self):
        p = PrologString(
            """
        a(1).
        res :- try_call(a(1)).
        query(res).
        """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values())[0], 1)

    def test_try_call_existing_fact_non_ground(self):
        p = PrologString(
            """
        a(1).
        res :- try_call(a(X)).
        query(res).
        """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values())[0], 1)

    def test_try_call_existing_fact_non_ground2(self):
        p = PrologString(
            """
        a(1).
        b :- a(X), X > 0.
        res :- try_call(b).
        query(res).
        """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values())[0], 1)

    def test_try_call_non_existing_fact_non_ground(self):
        p = PrologString(
            """
        a(1).
        b :- a(X), X > 2.
        res :- try_call(b).
        query(res).
        """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values())[0], 0)

    def test_try_call_non_existing_fact_non_ground2(self):
        p = PrologString(
            """
        a(1).
        res :- try_call(b(X)).
        query(res).
        """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values())[0], 0)

    def test_try_call_wrong_comp(self):
        p = PrologString(
            """
        res :- try_call(1 > 2).
        query(res).
        """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values())[0], 0)

    def test_try_call_right_comp(self):
        p = PrologString(
            """
        res :- try_call(1 < 2).
        query(res).
        """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values())[0], 1)

    def test_try_call_non_existing_comp(self):
        p = PrologString(
            """
        res :- try_call(X > 2).
        query(res).
        """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values())[0], 0)

    def test_try_call_existing_clause(self):
        p = PrologString(
            """
        a:(c:-d).
        res :- try_call(a:(c:-d)).
        query(res).
        """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values())[0], 1)

    def test_try_call_non_existing_clause(self):
        p = PrologString(
            """
        a:(c:-d).
        res :- try_call(a:(c:-e)).
        query(res).
        """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values())[0], 0)
