import unittest
from .basic import *
from .unification import *


class TestLogicUnify(unittest.TestCase) :
    
    def setUp(self) :
        self.A = Var('A')
        self.B = Var('B')
        self.C = Var('C')
        self.D = Var('D')
        
        self.f = Term('f')
        self.g = Term('g')
        self.h = Term('h')
        
        self.a = Term('a')
        self.b = Term('b')
        
        self.c_0 = Constant(0)
        self.c_1 = Constant(1)
        
        self.c_a = Constant('a')
        self.c_b = Constant('b')
    
    def test_var_var(self) :
        tdb = TermDB()
        tdb.unify(self.A,self.B)
        self.assertEqual(tdb[self.A], tdb[self.B])

    def test_var_atom(self) :
        tdb = TermDB()
        tdb.unify(self.A,self.a)
        
        self.assertEqual(tdb[self.A], self.a)
        
    def test_var_const(self) :
        tdb = TermDB()
        tdb.unify(self.A,self.c_a)
        
        self.assertEqual(tdb[self.A], self.c_a)
        
    def test_var_cycle(self) :
        tdb = TermDB()
        with self.assertRaises(UnifyError) :
            tdb.unify(self.A,self.f(self.A))
    
    def test_unify_compound(self) :
        tdb = TermDB()
        self.assertTrue(tdb.unify(self.A, self.f(self.B)))
    
            
    def test_undo_unify(self) :
        
        tdb = TermDB()

        f1 = tdb.add(self.f(self.a,self.B))
        f2 = tdb.add(self.f(self.A,self.b))
        
        with tdb :            
            self.assertTrue(tdb.unify(f1,f2))
            
            self.assertEqual(tdb[self.a],self.a)
            self.assertEqual(tdb[self.b],self.b)        
            self.assertEqual(tdb[self.A],self.a)        
            self.assertEqual(tdb[self.B],self.b)
            self.assertEqual(tdb[f1],tdb[f2])        

        self.assertEqual(tdb[self.a],self.a)
        self.assertEqual(tdb[self.b],self.b)        
        self.assertNotEqual(tdb[self.A],self.a)        
        self.assertNotEqual(tdb[self.B],self.b)
        self.assertEqual(tdb[self.A],self.A)        
        self.assertEqual(tdb[self.B],self.B)
        self.assertNotEqual(tdb[f1],tdb[f2])  
        
    def test_same(self) :
        a1 = Term('a')
        a2 = Term('a')
        
        tdb = TermDB()
        f1 = tdb.add(self.f(a1))
        f2 = tdb.add(self.f(a2))
        self.assertEqual(f1,f2)
        
    def test_copyTo_two_vars_diff(self) :
        tdb1 = TermDB()
        tdb2 = TermDB()
        
        f = self.f
        X = self.A
        Y = self.B
        
        t1 = tdb1.add(X)
        T1, = tdb1.copyTo(tdb2, t1)
        T2, = tdb1.copyTo(tdb2, t1)
        self.assertNotEqual(tdb2[T1], tdb2[T2])

    def test_copyTo_two_vars_eq(self) :
        tdb1 = TermDB()
        tdb2 = TermDB()
        
        f = self.f
        X = self.A
        Y = self.B
        
        t1 = tdb1.add(X)
        
        T1, T2 = tdb1.copyTo(tdb2, t1, t1)
        self.assertEqual(tdb2[T1], tdb2[T2])

    def test_copyTo_func_two_vars(self) :
        tdb1 = TermDB()
        tdb2 = TermDB()
        
        f = self.f
        X = self.A
        Y = self.B
        
        t1 = tdb1.add(f(X,X))
        
        T1, = tdb1.copyTo(tdb2, t1)
        
        term = tdb1.getTerm(T1)
        self.assertEqual( *term.args )
        
        
    def test_add_unify_find(self) :
        tdb = TermDB()
        
        f = self.f
        A = self.A
        B = self.B
        
        tdb.add(A)
        tdb.add(B)
        t = tdb.add(self.f(B))
        tdb.unify(A,B)
        
        self.assertEqual(tdb.find(f(B)),t)
        
        
if __name__ == '__main__' :
    unittest.main(verbosity=2)        