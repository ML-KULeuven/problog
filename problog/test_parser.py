import unittest

from .logic.program import PrologFactory
from .parser import PrologParser
from .logic import Term

class TestParser(unittest.TestCase) :

    def setUp(self) :
        self.parser = PrologParser(PrologFactory())
        
    def test_clause(self) :
        c = self.parser.parseString("alarm :- burglary, earthquake, p_alarm3.")
        self.assertEqual( str(list(c)[0]), "alarm :- burglary, earthquake, p_alarm3" )

    def test_clause_negation(self) :
        c = self.parser.parseString("alarm :- burglary, \+earthquake, p_alarm3.")
        self.assertEqual( str(list(c)[0]), "alarm :- burglary, \+(earthquake), p_alarm3" )

    def test_directive(self) :
        c = self.parser.parseString(":- consult(file).")
        c = list(c)[0]
        self.assertEqual( c , Term("':-'", Term('consult', Term('file' ) ) )  )
        
    def test_clause_negation_first(self) :
        c = self.parser.parseString("alarm :- \+burglary, earthquake, p_alarm3.")
        self.assertEqual( str(list(c)[0]), "alarm :- \+(burglary), earthquake, p_alarm3" )


