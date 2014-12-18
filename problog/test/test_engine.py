import unittest

from problog.program import PrologString
from problog.engine import DefaultEngine
from problog.logic import Term, Constant

import glob, os

class TestEngine(unittest.TestCase):
    
    def setUp(self) :

        try :
            self.assertCollectionEqual = self.assertItemsEqual
        except AttributeError :
            self.assertCollectionEqual = self.assertCountEqual
    
    
    def test_nonground_query_ad(self) :
        """Non-ground call to annotated disjunction"""

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
        """Comparison operator"""
        
        program = """
            morning(Hour) :- Hour >= 6, Hour =< 10.
        """
        
        engine = DefaultEngine()
        db = engine.prepare( PrologString(program) )
                
        self.assertEqual( engine.query(db, Term('morning', Constant(8) )), [[8]])
        
    def test_anonymous_variable(self) :
        """Anonymous variables are distinct"""
        
        program = """
            p(_,X,_) :- X = 3.
            
            q(1,2,3).
            q(1,2,4).
            q(2,3,5).
            r(Y) :- q(_,Y,_). 
            
        """
        
        engine = DefaultEngine()
        db = engine.prepare( PrologString(program) )
        self.assertEqual( engine.query(db, Term('p', Constant(1), Constant(3), Constant(2) )), [[Constant(1),Constant(3),Constant(2)]])
    
        self.assertEqual(engine.query(db, Term('r', None )), [[2], [3]])
        
    def test_functors(self) :
        """Calls with functors"""

        program = """
            p(_,f(A,B),C) :- A=y, B=g(C).    
            a(X,Y,Z) :- p(X,f(Y,Z),c).
        """
        pl = PrologString(program)

        r1 = DefaultEngine().query(pl, Term('a',Term('x'),None,Term('g',Term('c'))))
        r1 = [ list(map(str,sol)) for sol in r1  ]
        self.assertCollectionEqual( r1, [['x', 'y', 'g(c)']])

        r2 = DefaultEngine().query(pl, Term('a',Term('x'),None,Term('h',Term('c'))))
        self.assertCollectionEqual( r2, [])

        r3 = DefaultEngine().query(pl, Term('a',Term('x'),None,Term('g',Term('z'))))
        self.assertCollectionEqual( r3, [])
    
        if __name__ == '__main__' :
            test1()