from problog.prolog_engine.pylo.XSBProlog import XSBProlog
from problog.prolog_engine.pylo.GnuProlog import GNUProlog
from problog.prolog_engine.pylo.language import global_context
from problog.logic import Term, Constant, Var

class PyloProlog:
    def __init__(self):
        self.pl = XSBProlog("/home/yann/xsb-git/XSB")

    def term2pylo(self, term):
        if isinstance(term, Constant):
            return global_context.get_constant(term.value)
        if isinstance(term, Var):
            return global_context.get_variable(term.name)
        if isinstance(term, Term):
            t = global_context.get_predicate(term.functor, term.arity)
            args = [self.term2pylo(arg) for arg in term.args]
            return t(*args)
        return term

    def parse_argument(self, arg):
        if isinstance(arg, str):
            t = Term.from_string(arg)
            return self.term2pylo(t)
        return arg

    def assertz(self, functor, args):
        pylo_args = [self.parse_argument(arg) for arg in args]
        fa = global_context.get_predicate(functor, len(args))
        # TODO: Check if the arguments reformatting is needed .replace("'", '"')
        t = fa(*pylo_args)
        self.pl.assertz(t)

    def consult(self, path):
        self.pl.consult(path)

    def query(self, query):
        self.pl.query(query)

    def load(self, filename, database=None):
        raise NotImplementedError("Pylo does not support load at this moment.")
