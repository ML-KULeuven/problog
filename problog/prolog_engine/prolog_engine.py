from problog.clausedb import ClauseDB
from problog.engine import GenericEngine
from problog.formula import LogicFormula
from problog.logic import Term, Var
from swi_program import SWIProgram


class PrologEngine(GenericEngine):

    def __init__(self) -> None:
        super().__init__()

    def prepare(self, db):
        '''Create a SWIProgram from the given LogicProgram'''
        db = ClauseDB.createFrom(db, builtins={})
        db.engine = self
        return SWIProgram(db)

    # def query(self, sp, term):
    #
    #     def_node = sp.db.get_node(sp.db.find(term))
    #     nodes = [sp.db.get_node(c) for c in def_node.children]
    #     return [Term(n.functor, *n.args) for n in nodes]

    def ground(self, sp, term, target=None, label=None, *args, **kwargs):
        """Ground a given query term and store the result in the given ground program.

       :param sp: SWIprogram
       :param term: term to ground; variables should be represented as None
       :param target: target logic formula to store grounding in (a new one is created if none is \
       given)
       :param label: optional label (query, evidence, ...)
       :returns: logic formula (target if given)
        """
        if target is None:
            target = LogicFormula()
        query_result = sp.query('prove({},Proofs,GroundQueries)'.format(term))
        result = sp.add_proofs(query_result['Proofs'], query_result['GroundQueries'], target=target)
        return result

    def ground_all(self, sp, target=None, queries=None, evidence=None, *args, **kwargs):
        """Ground all queries and evidence found in the the given database.

       :param sp: SWIprogram
       :param target: logic formula to ground into
       :param queries: list of queries to evaluate instead of the ones in the logic program
       :param evidence: list of evidence to evaluate instead of the ones in the logic program
       :returns: ground program
        """
        if target is None:
            target = LogicFormula()
        if queries is None:
            queries = [q[0].args[0] for q in self.ground(sp, Term('query', Var('X')), *args, **kwargs).queries()]
        for q in queries:
            self.ground(sp, q, target, *args, **kwargs)
        return target

