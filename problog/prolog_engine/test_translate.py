import unittest
from problog.program import PrologString, PrologFile
from problog.sdd_formula import SDD
from problog.prolog_engine.engine_prolog import EngineProlog
from problog.logic import Term, Var
from problog.formula import LogicFormula
from problog.prolog_engine.swi_program import SWIProgram
program='''
0.25::stress(1).
0.35::stress(2).

0.2::influences(1,2).
0.2::influences(2,1).

smokes(X) :- stress(X).
smokes(X) :- influences(Y, X), smokes(Y).

query(smokes(1)).
query(smokes(2)).'''

program = PrologString(program)
engine = EngineProlog()
db = engine.prepare(program)


print(SWIProgram(db))

ground = engine.ground_all(db, target=LogicFormula(keep_all=False), k=3)
ac = SDD.create_from(ground)
print(ac.evaluate())
