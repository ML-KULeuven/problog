from __future__ import print_function

import unittest

import problog



a = problog.logic.Term('a')
b = problog.logic.Term('b')
c = problog.logic.Term('c')

def unify(t1, t2):
    sv = problog.engine_unify._VarTranslateWrapper({}, min(t2.variables()))
    tv = {}
    problog.engine_unify.unify_value_dc(t1, t2, sv, tv)

    sv = {k: tv.get(v, v) for k, v in sv.items()}
    sv = {k: problog.engine_unify.substitute_all([v], tv)[0] for k, v in sv.items()}
    return sv


class TestUnify(unittest.TestCase):

    def setUp(self):
        try:
            self.assertCollectionEqual = self.assertItemsEqual
        except AttributeError:
            self.assertCollectionEqual = self.assertCountEqual


tests = [
    (a(-1, a, -2, -2), a(-1, -3, -1, -2), {-1: -1, -2: -1}),
    (a(-1, -1, -2, -2), a(-1, -2, -2, a), {-1: a, -2: a}),
    (a(-1, -1), a(-1, b(-1)), 'OccursCheck'),
    (a(-1, b(-2), c), a(a, -2, -2), 'UnifyError'),
    (a(-2, -2, -1, -1), a(-1, -2, -2, a), {-1: a, -2: a}),
    (a(-2, b(-2, -1), -1), a(-1, b(-2, -2), a), {-1: a, -2: a}),
    (a(-2, -3, -1, -1), a(-1, b(-2, -2), a, -2), {-1: a, -2: -1, -3: b(a, a)}),
    (a(-1, -1, -2, -2, -3, -3), a(-2, -3, -1, a, -1, -2), {-1: a, -2: a, -3: a}),
    (a(-1, -1, -2, -2, -3, -3), a(-2, -3, -1, -4, -1, -2), {-1: -1, -2: -1, -3: -1}),
    (a(-1, -1, -3, -3, -2, -2), a(-2, -3, -1, -4, -1, -2), {-1: -1, -2: -1, -3: -1}),
]


def create_test(t1, t2, r):

    def f(self):
        if r == 'OccursCheck':
            self.assertRaises(problog.engine_unify.OccursCheck, unify, t1, t2)
        elif r == 'UnifyError':
            self.assertRaises(problog.engine_unify.UnifyError, unify, t1, t2)
        else:
            self.assertCollectionEqual(unify(t1, t2).items(), r.items())
    return f

for i, t in enumerate(tests):
    t1, t2, r = t

    f = create_test(t1, t2, r)

    t2s = problog.logic.term2str
    if type(r) == dict:
        r = ', '.join(['%s=%s' % (t2s(x), t2s(y)) for x, y in r.items()])

    f.__doc__ = "%s = %s -> %s" % (t2s(t1), t2s(t2), r)

    setattr(TestUnify, 'test_unify_%03d' % i, f)


if __name__ == '__main__':
    unittest.main()
