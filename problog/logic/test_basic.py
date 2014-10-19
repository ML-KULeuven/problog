import unittest
from .basic import *


class TestLogicBasic(unittest.TestCase) :
    
    def test_term_eq_term(self) :
        a = Term('a')
        b = Term('a')
        self.assertEqual(a,b)
        
    def test_var_eq_var(self) :
        a = Var('A')
        b = Var('A')
        self.assertEqual(a,b)
        
    def test_const_eq_const(self) :
        a = Constant(1)
        b = Constant(1)
        self.assertEqual(a,b)
        
    def test_constInt_eq_constFloat(self) :
        a = Constant(1)
        b = Constant(1.0)
        self.assertEqual(a,b)
    
    def test_constant_variable_clash_str(self) :
        a = Var('A')
        b = Constant('A')
        self.assertNotEqual(str(a),str(b))
        
    def test_term_variable_clash_str(self) :
        a = Term("'A'")
        b = Var('A')
        self.assertNotEqual(str(a),str(b))
        
if __name__ == '__main__' :
    unittest.main(verbosity=2)
        