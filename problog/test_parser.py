import unittest

from .logic.program import PrologFactory
from .parser import PrologParser

class TestParser(unittest.TestCase) :

    def setUp(self) :
        self.parser = PrologParser(PrologFactory())
        
    def test_clause(self) :
        c = self.parser.parseString("alarm :- burglary, earthquake, p_alarm3.")
        self.assertEqual( str(list(c)[0]), "alarm :- burglary, earthquake, p_alarm3" )

    def test_clause_negation(self) :
        c = self.parser.parseString("alarm :- burglary, \+earthquake, p_alarm3.")
        self.assertEqual( str(list(c)[0]), "alarm :- burglary, \+(earthquake), p_alarm3" )

    def test_clause_negation_first(self) :
        c = self.parser.parseString("alarm :- \+burglary, earthquake, p_alarm3.")
        self.assertEqual( str(list(c)[0]), "alarm :- \+(burglary), earthquake, p_alarm3" )

