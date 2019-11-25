from problog.clausedb import ClauseDB
from problog.engine import GenericEngine
from problog.formula import LogicFormula
from problog.logic import Term, Var
from problog.prolog_engine.swi_program import SWIProgram


class PrologEngine(GenericEngine):

    def __init__(self) -> None:
        super().__init__()

    def prepare(self, db):
        db = ClauseDB.createFrom(db, builtins={})
        db.engine = self
        return SWIProgram(db)

    def query(self, sp, term):
        def_node = sp.db.get_node(sp.db.find(term))
        nodes = [sp.db.get_node(c) for c in def_node.children]
        return [Term(n.functor, *n.args) for n in nodes]

    def ground(self, sp, term, target=None, label=None, *args, **kwargs):
        if target is None:
            target = LogicFormula(keep_all=True)
        proofs = self.get_proofs(str(term), sp, *args, **kwargs)
        result = sp.add_proofs(proofs, target=target)
        return result

    def ground_all(self, sp, target=None, queries=None, evidence=None, *args, **kwargs):
        if target is None:
            target = LogicFormula()
        if queries is None:
            queries = [q[0].args[0] for q in self.ground(sp, Term('query', Var('X')), *args, **kwargs).queries()]
        for q in queries:
            self.ground(sp, q, target, *args, **kwargs)
        return target

    def get_proofs(self, q, program, *args, **kwargs):
        return program.query('prove({},Proofs)'.format(q))['Proofs']

