import unittest
from problog.program import PrologString, PrologFile
from problog.sdd_formula import SDD
from problog.prolog_engine.engine_prolog import EngineProlog
from problog.logic import Term, Var
from problog.formula import LogicFormula

file = '''
0.6::edge(1,2).
0.1::edge(1,3).
0.4::edge(2,5).
0.3::edge(2,6).
0.3::edge(3,4).
0.8::edge(4,5).
0.2::edge(5,6).

path(X,Y) :- edge(X,Y).
path(X,Y) :- edge(X,Z),
             Y \== Z,
         path(Z,Y).


query(path(1,5)).
query(path(1,6)).

        '''

class Test(unittest.TestCase):
    def test_engine(self):
        program = PrologString(file)
        engine = EngineProlog()
        db = engine.prepare(program)
        ground = engine.ground_all(db, target=LogicFormula(keep_all=False))
        ac = SDD.create_from(ground)
        print(ac.evaluate())

# if __name__ == '__main__':
#     unittest.main()

test = Test()
test.test_engine()