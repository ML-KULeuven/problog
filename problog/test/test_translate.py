import unittest
from problog.program import PrologString
from problog.prolog_engine.translate import TranslatedProgram, translate_clasusedb
from problog.engine import DefaultEngine
from problog.sdd_formula import SDD

class Test(unittest.TestCase):
    def test_translate(self):
        file = '''
a(Y) :-\+b(X), c(Y).
0.5::b(0).
c(X) :- d(X).
d(0).
        '''
        program = PrologString(file)
        engine = DefaultEngine()
        db = engine.prepare(program)
        translate_program = translate_clasusedb(db)
        print(translate_program)
        print(translate_program.get_proofs('a(X)'))
        formula2 = translate_program.to_logic_formula('a(X)')
        sdd = SDD(formula2)
        print(sdd.evaluate())


if __name__ == '__main__':
    unittest.main()
