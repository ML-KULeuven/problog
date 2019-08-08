from problog.clausedb import ClauseDB
from problog.engine import GenericEngine
from problog.formula import LogicFormula
from problog.logic import Term
from problog.prolog_engine.translate import translate_clausedb


class EngineProlog(GenericEngine):

    def prepare(self, db):
        result = ClauseDB.createFrom(db, builtins={})
        result.engine = self
        return result


    def query(self, db, term):
        def_node = db.get_node(db.find(term))
        nodes = [db.get_node(c)for c in def_node.children]
        return [Term(n.functor, *n.args) for n in nodes]

    def ground(self, db, term, target=None, label=None):
        translate_program = translate_clausedb(db)
        return translate_program.ground(str(term), target=target)

    def ground_all(self, db, target=None, queries=None, evidence=None):
        if target is None:
            target = LogicFormula()
        if queries is None:
            queries = [q.args[0] for q in self.query(db, Term('query', None))]
        for q in queries:
            self.ground(db, q, target)
        return target
