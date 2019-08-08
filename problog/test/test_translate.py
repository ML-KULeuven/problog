import unittest
from problog.program import PrologString
from problog.sdd_formula import SDD
from problog.prolog_engine.engine_prolog import EngineProlog
from problog.logic import Term, Var
from problog.formula import LogicFormula

file = '''
a(0) :- b(X,2), c(X,2).
a(1) :- b(X,2).
0.5::b(X,2);0.5::c(X,2) :- d(X,Y), e(a).
d(0,3).
d(1,3).
e(a).
query(a(X)).
        '''

class Test(unittest.TestCase):
    def test_engine(self):
        program = PrologString(file)
        engine = EngineProlog()
        db = engine.prepare(program)
        ground = engine.ground_all(db, target=LogicFormula(keep_all=True))
        ac = SDD.create_from(ground)
        print(ac.evaluate())

if __name__ == '__main__':
    unittest.main()
