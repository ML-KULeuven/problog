import unittest

from .program import PrologString
from .eb_engine import EventBasedEngine
from .basic import Term

class TestLogicEngine(unittest.TestCase) :

    def __init__(self, *args) :
        unittest.TestCase.__init__(self, *args)
        try :
            self.assertElements = self.assertCountEqual
        except AttributeError :
            self.assertElements = self.assertItemsEqual
    
    def setUp(self) :
        self.engine = EventBasedEngine()
        
    
    def test_query_facts(self) :
        pl = PrologString("""
            parent(ma,er).
            parent(er,ka).
            parent(er,an).
            parent(ka,li).
        """)
        
        expected_result = ['(ma, er)', '(er, ka)', '(er, an)', '(ka, li)']
        obtained_result = map(str,map(tuple,self.engine.query( pl, Term('parent', None, None) )))
        
        self.assertElements( expected_result, obtained_result )

    def test_query_single(self) :
        pl = PrologString("""
            parent(ma,er).
            parent(er,ka).
            parent(er,an).
            parent(ka,li).
            
            grandparent(X,Y) :- parent(X,Z), parent(Z,Y).
        """)
        
        expected_result = ['(ma, ka)', '(ma, an)', '(er, li)']
        obtained_result = map(str,map(tuple,self.engine.query( pl, Term('grandparent', None, None) )))
        
        self.assertElements( expected_result, obtained_result )

    def test_query_multi(self) :
        pl = PrologString("""
            parent(ma,er).
            parent(er,ka).
            parent(er,an).
            parent(ka,li).
            
            married(er,fr).
            married(ma,ja).
            
            grandparent(X,Y) :- parent(X,Z), parent(Z,Y).
            grandparent(X,Y) :- married(X,Z), grandparent(Z,Y).
            grandparent(X,Y) :- married(Z,X), grandparent(Z,Y).
        """)
        
        expected_result = ['(ma, ka)', '(ma, an)', '(er, li)', '(fr, li)', '(ja, an)', '(ja, ka)']
        obtained_result = map(str,map(tuple,self.engine.query( pl, Term('grandparent', None, None) )))
        
        self.assertElements( expected_result, obtained_result )

    def test_query_fact_clause(self) :
        pl = PrologString("""
            parent(ma,er).
            parent(er,ka).
            parent(er,an).
            parent(ka,li).
            
            married(er,fr).
            married(ma,ja).
            
            parent(X,Y) :- married(X,Z), parent(Z,Y).
            parent(X,Y) :- married(Z,X), parent(Z,Y).
        """)
        
        expected_result = ['(ma, er)', '(ja, er)', '(er, ka)', '(fr, ka)', '(er, an)', '(fr, an)', '(ka, li)']
        obtained_result = map(str,map(tuple,self.engine.query( pl, Term('parent', None, None) )))
        
        self.assertElements( expected_result, obtained_result )
        
        
    def test_query_recurse_nocyle(self) :
        pl = PrologString("""
            parent(ma,er).
            parent(er,ka).
            parent(er,an).
            parent(ka,li).
            
            married(er,fr).
            married(ma,ja).
            
            ancestor(X,Y) :- parent(X,Y).
            ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y).
        """)
    
    
        expected_result = ['(ma, ka)', '(ma, an)', '(er, li)', '(ma, li)', '(er, ka)', '(er, an)', '(ka, li)', '(ma, er)']
        obtained_result = map(str,map(tuple,self.engine.query( pl, Term('ancestor', None, None) )))
        
        self.assertElements( expected_result, obtained_result )

    def test_query_recurse_cyle(self) :
        pl = PrologString("""
            parent(ma,er).
            parent(er,ka).
            parent(er,an).
            parent(ka,li).
            
            married(er,fr).
            married(ma,ja).
            
            ancestor(X,Y) :- ancestor(Z,Y), parent(X,Z).
            ancestor(X,Y) :- parent(X,Y).
        """)
    
    
        expected_result = ['(ma, ka)', '(ma, an)', '(er, li)', '(ma, li)', '(er, ka)', '(er, an)', '(ka, li)', '(ma, er)']
        obtained_result = map(str,map(tuple,self.engine.query( pl, Term('ancestor', None, None) )))
        
        self.assertElements( expected_result, obtained_result )

    def test_query_recurse_doublecyle(self) :
        pl = PrologString("""
            parent(ma,er).
            parent(er,ka).
            parent(er,an).
            parent(ka,li).
            parent(X,Y) :- married(X,Z), parent(Z,Y).
            
            married(er,fr).
            married(ma,ja).
            married(X,Y) :- married(Y,X).
            
            ancestor(X,Y) :- ancestor(Z,Y), parent(X,Z).
            ancestor(X,Y) :- parent(X,Y).
        """)
    
    
        expected_result = ['(ma, ka)', '(ma, an)', '(er, li)', '(ma, li)', '(er, ka)', '(er, an)', '(ka, li)', '(ma, er)']
        expected_result += ['(ja, ka)', '(ja, an)', '(fr, li)', '(ja, li)', '(fr, ka)', '(fr, an)', '(ja, er)']
        obtained_result = map(str,map(tuple,self.engine.query( pl, Term('ancestor', None, None) )))
        
        self.assertElements( expected_result, obtained_result )

