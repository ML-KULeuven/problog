import unittest

from problog.program import PrologString
from problog.engine import DefaultEngine
from problog.logic import Term

import glob, os

class TestDummy(unittest.TestCase):
    
    def setUp(self) :

        try :
            self.assertSequenceEqual = self.assertItemsEqual
        except AttributeError :
            self.assertSequenceEqual = self.assertCountEqual
    
    
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
        
        self.assertSequenceEqual( found, [ 'p(a)', 'p(b)' ])
        
