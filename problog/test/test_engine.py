import unittest

from problog.program import PrologString
from problog.engine import DefaultEngine
from problog.logic import Term, Constant

import glob, os

class TestDummy(unittest.TestCase):
    
    def setUp(self) :

        try :
            self.assertCollectionEqual = self.assertItemsEqual
        except AttributeError :
            self.assertCollectionEqual = self.assertCountEqual
    
    
    def test_nonground_query_ad(self) :

        program = """
            0.1::p(a); 0.2::p(b).
            query(p(_)).
        """
    
        engine = DefaultEngine()
        db = engine.prepare( PrologString(program) )
        
        result=None
        for query in engine.query(db, Term('query', None)) :
            result = engine.ground(db, query[0], result, label='query')
        
        found = [ str(x) for x, y in result.queries() ]
        
        self.assertCollectionEqual( found, [ 'p(a)', 'p(b)' ])
        

    def test_compare(self) :
        
        program = """
            morning(Hour) :- Hour >= 6, Hour =< 10.
        """
        
        engine = DefaultEngine()
        db = engine.prepare( PrologString(program) )
                
        self.assertEqual( engine.query(db, Term('morning', Constant(8) )), [[8]])
        
    def test_anonymous_variable(self) :
        
        program = """
            p(_,_).
        """
        
        engine = DefaultEngine()
        db = engine.prepare( PrologString(program) )
                
        self.assertEqual( engine.query(db, Term('p', Constant(1), Constant(2) )), [[1,2]])
    