import unittest
from problog.program import PrologString, PrologFile
from problog.sdd_formula import SDD
from problog.prolog_engine.engine_prolog import EngineProlog
from problog.logic import Term, Var
from problog.formula import LogicFormula

file = '''
0.5::heads1.
0.6::heads2.

twoHeads :- heads1, heads2.

query(heads1).
query(heads2).
query(twoHeads).
        '''

class Test(unittest.TestCase):
    def test_engine(self):
        # program = PrologString(file)
        program = PrologFile('/home/robinm/phd/problog/test/00_trivial_not_and.pl')
        engine = EngineProlog()
        db = engine.prepare(program)
        ground = engine.ground_all(db, target=LogicFormula(keep_all=True))
        ac = SDD.create_from(ground)
        print(ac.evaluate())

if __name__ == '__main__':
    unittest.main()
