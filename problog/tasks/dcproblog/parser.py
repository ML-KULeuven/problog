from problog.parser import PrologParser, Token
from problog.program import ExtendedPrologFactory


class DCParser(PrologParser):
    def __init__(self):
        PrologParser.__init__(self, ExtendedPrologFactory())

    def _token_tilde(self, s, pos):
        return Token('~', pos, binop=(700, 'xfx', self.factory.build_binop),
        functor=self._next_paren_open(s, pos)), pos + 1
