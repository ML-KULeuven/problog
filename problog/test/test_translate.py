import unittest
from problog.program import PrologString
from problog.prolog_engine.translate import TranslatedProgram, translate_clasusedb
from problog.engine import DefaultEngine

class Test(unittest.TestCase):
    def test_translate(self):
        file = '''
a(Y) :-\+b(X), c(Y).
0.5::b(0).
c(X) :- d(X).
d(X) :- X is 2-1.
        '''
        program = PrologString(file)
        engine = DefaultEngine()
        db = engine.prepare(program)
        translate_program = translate_clasusedb(db)
        print(translate_program)


if __name__ == '__main__':
    unittest.main()
