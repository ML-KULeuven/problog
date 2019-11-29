import logging

from problog.util import Timer
from problog.engine import DefaultEngine, ground, ClauseDB
from problog.core import transform
from problog.program import LogicProgram
from problog.logic import Term, Var, is_ground
from problog.formula import LogicFormula
from problog.engine_stack import SimpleBuiltIn


from .engine_stack import SimpleProbabilisticBuiltIn
from .formula import LogicFormulaHAL
from .engine_builtin import \
    _builtin_is, \
    _builtin_gt, _builtin_lt, _builtin_le, _builtin_ge, \
    _builtin_observation, \
    _builtin_query_density1, _builtin_query_density2 \

    # , _builtin_val_eq, _builtin_val_neq


class EngineHAL(DefaultEngine):
    def __init__(self,**kwargs):
        DefaultEngine.__init__(self,**kwargs)

    def load_builtins(self):
        DefaultEngine.load_builtins(self)
        self.add_builtin('is', 2, SimpleProbabilisticBuiltIn(_builtin_is))
        self.add_builtin('>', 2, SimpleProbabilisticBuiltIn(_builtin_gt))
        self.add_builtin('<', 2, SimpleProbabilisticBuiltIn(_builtin_lt))
        self.add_builtin('=<', 2, SimpleProbabilisticBuiltIn(_builtin_le))
        self.add_builtin('>=', 2, SimpleProbabilisticBuiltIn(_builtin_ge))

        self.add_builtin('observation_builtin', 2, SimpleBuiltIn(_builtin_observation))
        self.add_builtin('query_density_builtin1', 1, SimpleBuiltIn(_builtin_query_density1))
        self.add_builtin('query_density_builtin2', 2, SimpleBuiltIn(_builtin_query_density2))



        # self.add_builtin('=\=', 2, b(_builtin_val_neq))
        # self.add_builtin('=:=', 2, b(_builtin_val_eq))



    def query(self, db, term, gp=None, **kwdargs):
        """
       :param db:
       :param term:
       :param kwdargs:
       :return:
        """
        if gp==None:
            gp = LogicFormula()
        if term.is_negated():
            term = -term
            negative = True
        else:
            negative = False
        gp, result = self._ground(db, term, gp, **kwdargs)
        if negative:
            if not result:
                return [term]
            else:
                return []
        else:
            return [x for x, y in result]

    def ground_dqueries(self, db, target, dqueries):
        logger = logging.getLogger('problog')
        for label, dquery in dqueries:
            logger.debug("Grounding query '%s'", dquery)
            if len(dquery)==1:
                target, density_node = self._ground(db, Term('query_density_builtin1', dquery[0]), target)
            elif len(dquery)==2:
                target, density_node = self._ground(db, Term('query_density_builtin2', dquery[0], dquery[1]), target)

            density_node = density_node[0][0]
            target.add_name(density_node[0], density_node[1], target.LABEL_DQUERY)
            logger.debug("Ground program size: %s", len(target))

    def ground_observation(self, db, target, observation, propagate_evidence=False):
        logger = logging.getLogger('problog')
        # Ground evidence
        for query in observation:
            logger.debug("Grounding observation({},{})".format(query[0], query[1]))
            target, observation_node = self._ground(db, Term('observation_builtin', query[0], query[1]), target)
            observation_node = observation_node[0][0]
            target.add_name(observation_node[0], observation_node[1], target.LABEL_OBSERVATION)
            logger.debug("Ground program size: %s", len(target))

        if propagate_evidence:
            with Timer('Propagating observation'):
                target.lookup_observation = {}
                obs_nodes = [node for name, node in target.observation() if node != 0 and node is not None]
                target.propagate(obs_nodes, target.lookup_observation)


    def ground_all(self, db, target=None, queries=None, dqueries=None, evidence=None, observation=None, propagate_evidence=False, labels=None, **kwdargs):
        if labels is None:
            labels = []
        # Initialize target if not given.
        if target is None:
            target = LogicFormula()
        db = self.prepare(db)
        logger = logging.getLogger('problog')
        with Timer('Grounding'):
            # Load queries: use argument if available, otherwise load from database.
            if queries is None:
                 #THIS is different from ProbLog, wee need to pass on target
                queries = [q[0] for q in self.query(db, Term('query', None), gp=target)]
            for query in queries:
                if not isinstance(query, Term):
                    raise GroundingError('Invalid query')   # TODO can we add a location?
            if dqueries == None:
                dqueries = [(dq[0],) for dq in self.query(db, Term('query_density', None), gp=target)]
                dqueries += [(dq[0],dq[1]) for dq in self.query(db, Term('query_density', None, None), gp=target)]
            for dquery in dqueries:
                if not isinstance(dquery[0], Term):
                    raise GroundingError('Invalid query')
            # Load evidence: use argument if available, otherwise load from database.
            if evidence is None:
                #ALSO here: need to pass on target
                evidence = self.query(db, Term('evidence', None, None), gp=target)
                evidence += self.query(db, Term('evidence', None), gp=target)
            if observation is None:
                observation = self.query(db, Term('observation', None, None))
                observation += self.query(db, Term('observation', None))


            queries = [(target.LABEL_QUERY, q) for q in queries]
            for label, arity in labels:
                 #ALSO here: need to pass on target
                queries += [(label, q[0]) for q in self.query(db, Term(label, *([None] * arity)), gp=target)]

            dqueries = [(target.LABEL_DQUERY, dq) for dq in dqueries]
            for label, arity in labels:
                 #ALSO here: need to pass on target
                dqueries += [(label, dq[0]) for dq in self.query(db, Term(label, *([None] * arity)), gp=target)]

            for ev in evidence:
                if not isinstance(ev[0], Term):
                    raise GroundingError('Invalid evidence')   # TODO can we add a location?
            for ob in observation:
                if not isinstance(ob[0], Term):
                    raise GroundingError('Invalid evidence')   # TODO can we add a location?

            # Ground queries
            if propagate_evidence:
                self.ground_evidence(db, target, evidence, propagate_evidence=propagate_evidence)
                self.ground_observation(db, target, observation, propagate_evidence=propagate_evidence)
                self.ground_queries(db, target, queries)
                self.ground_dqueries(db, target, dqueries)
                if hasattr(target, 'lookup_evidence'):
                    logger.debug('Propagated evidence: %s' % list(target.lookup_evidence))
            else:
                self.ground_evidence(db, target, evidence)
                self.ground_observation(db, target, observation)
                self.ground_queries(db, target, queries)
                self.ground_dqueries(db, target, dqueries)

        return target


def init_engine(**kwdargs):
    engine = EngineHAL(**kwdargs)
    return engine

@transform(LogicProgram, LogicFormulaHAL)
def groundHAL(model, target=None, engine=None, **kwdargs):
    #change grounder here to swipl once ready
    return engine.ground_all(model, target=target, **kwdargs)
