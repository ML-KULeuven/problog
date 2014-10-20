import unittest
from .basic import *
from .program import *

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
        
class TestLogicProgram(unittest.TestCase) :

    def test_lp_create_other(self) :
        lp = SimpleProgram()
        lp_copy = PrologFile.createFrom( lp )
        self.assertEqual( type(lp_copy), PrologFile )
        
        lp_copy = ClauseDB.createFrom( lp )
        self.assertEqual( type(lp_copy), ClauseDB )
    
    def test_lp_create_self(self) :
        lp = SimpleProgram()
        lp_copy = SimpleProgram.createFrom( lp )
        self.assertEqual(id(lp), id(lp_copy))

    def test_lp_create_copy(self) :
        lp = SimpleProgram()
        lp_copy = SimpleProgram.createFrom( lp, force_copy=True )
        self.assertNotEqual(id(lp), id(lp_copy))
        
if __name__ == '__main__' :
    unittest.main(verbosity=2)
        