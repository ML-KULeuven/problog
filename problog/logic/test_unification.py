import unittest
from .basic import *
from .unification import *


class TestLogicUnify(unittest.TestCase) :
    
    def setUp(self) :
        self.A = Var('A')
        self.B = Var('B')
        self.C = Var('C')
        self.D = Var('D')
        
        self.f = Term.create('f')
        self.g = Term.create('g')
        self.h = Term.create('h')
        
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
        
        
              
        
        
if __name__ == '__main__' :
    unittest.main(verbosity=2)        