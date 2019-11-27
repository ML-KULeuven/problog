from problog.parser import PrologParser, Token
from problog.program import ExtendedPrologFactory

from .program import DCPrologFactory


SPECIAL_BRACE_OPEN = 14
SPECIAL_BRACE_CLOSE = 15


class DCParser(PrologParser):
    def __init__(self):
        PrologParser.__init__(self, DCPrologFactory())


    def _token_brace_open(self, s, pos):
        return Token('{', pos, atom=False, special=SPECIAL_BRACE_OPEN), pos + 1

    def _token_brace_close(self, s, pos):
        return Token('}', pos, atom=False, special=SPECIAL_BRACE_CLOSE), pos + 1

    def _token_tilde(self, s, pos):
        return Token('~', pos, binop=(700, 'xfx', self.factory.build_binop),
        functor=self._next_paren_open(s, pos)), pos + 1
