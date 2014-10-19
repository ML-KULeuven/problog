import unittest
from .basic import *
from .prolog import *

class TestLogicProlog(unittest.TestCase) :
    
    def test_term_eq_term(self) :
        
        f = Term('+', Constant(2), Constant(3))
        
        self.assertEqual(compute(f), Constant(5) )

        
if __name__ == '__main__' :
    unittest.main(verbosity=2)
        