from problog.clausedb import ClauseDB
from problog.engine import GenericEngine
from problog.formula import LogicFormula
from problog.logic import Term, Var
from problog.prolog_engine.swi_program import SWIProgram
from problog.prolog_engine.swi_formula import SWI_Formula


class PrologEngine(GenericEngine):

    def __init__(self) -> None:
        super().__init__()

    def prepare(self, db):
        '''Create a SWIProgram from the given LogicProgram'''
        if type(db) is SWIProgram:
            return db
        db = ClauseDB.createFrom(db, builtins={})
        db.engine = self
        return SWIProgram(db)

    # def query(self, sp, term):
    #
    #     def_node = sp.db.get_node(sp.db.find(term))
    #     nodes = [sp.db.get_node(c) for c in def_node.children]
    #     return [Term(n.functor, *n.args) for n in nodes]

    def ground(self, sp, term, target=None, formula=None,suppress=False, label=None, *args, **kwargs):
        """Ground a given query term and store the result in the given ground program.

       :param sp: SWIprogram
       :param term: term to ground; variables should be represented as None
       :param target: target logic formula to store grounding in (a new one is created if none is \
       given)
       :param label: optional label (query, evidence, ...)
       :returns: logic formula (target if given)
        """
        if target is None:
            target = LogicFormula(auto_compact=False)
        if formula is None:
            formula = SWI_Formula()
        sp = self.prepare(sp)
        query_result = sp.query('prove({},Proofs,GroundQueries)'.format(term))
        ground_queries = list(query_result['GroundQueries'])
        for proof in query_result['Proofs']:
            formula.add_proof(proof)
        if ground_queries:
            for t in ground_queries:
                formula.names.add((t, label))
        else:
            if term.functor != 'evidence':
                formula.names.add((term, label))

        # result = sp.add_proofs(query_result['Proofs'], query_result['GroundQueries'], target=target, label=label)
        if not suppress:
            formula.to_formula(target)
        return target

    def ground_all(self, sp, target=None, queries=None, evidence=None, *args, **kwargs):
        """Ground all queries and evidence found in the the given database.

       :param sp: SWIprogram
       :param target: logic formula to ground into
       :param queries: list of queries to evaluate instead of the ones in the logic program
       :param evidence: list of evidence to evaluate instead of the ones in the logic program
       :returns: ground program
        """
        if target is None:
            target = LogicFormula(auto_compact=False)
        if queries is None:
            queries = [q[0].args[0] for q in self.ground(sp, Term('query', Var('X')), label=target.LABEL_QUERY, *args, **kwargs).queries()]
        if evidence is None:
            evidence = [e[0].args for e in self.ground(sp, Term('evidence', Var('X'), Var('Y')),  label=target.LABEL_QUERY).queries()]
        swi_formula = SWI_Formula()
        for q in queries:
            print('query: ',q)
            self.ground(sp, q, target, formula=swi_formula, suppress=True, label=target.LABEL_QUERY, *args, **kwargs)
        for e in evidence:
            if e[1] == Term('true'):
                self.ground(sp, e[0], target, formula=swi_formula, suppress=True, label=target.LABEL_EVIDENCE_POS)
            elif e[1] == Term('false'):
                self.ground(sp, e[0], target, formula=swi_formula, suppress=True, label=target.LABEL_EVIDENCE_NEG)
            else:
                print(e[1], ' evidence interpreted as maybe')
                self.ground(sp, e[0], target, formula=swi_formula, suppress=True, label=target.LABEL_EVIDENCE_MAYBE)

        swi_formula.to_formula(target)
        return target

