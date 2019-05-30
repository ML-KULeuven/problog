import logging

from problog.util import Timer
from problog.engine import ground, ClauseDB

from problog.core import transform
from problog.program import LogicProgram
from problog.logic import Term, is_ground
from problog.formula import LogicFormula
from problog.engine_stack import SimpleProbabilisticBuiltIn, SimpleBuiltIn

from .engine_stack import StackBasedEngineHAL as DefaultEngineHAL


from .formula import LogicFormulaHAL
from .engine_builtin import \
    _builtin_density, \
    _builtin_free_list, _builtin_free, \
    _builtin_as, \
    _builtin_gt, _builtin_lt, _builtin_le, _builtin_ge, \
    _builtin_observation \
    # _builtin_is \

    # , _builtin_val_eq, _builtin_val_neq


class EngineHAL(DefaultEngineHAL):
    def __init__(self,**kwargs):
        DefaultEngineHAL.__init__(self,**kwargs)

    def load_builtins(self):
        DefaultEngineHAL.load_builtins(self)
        self.add_builtin('density_builtin', 1, _builtin_density)

        self.add_builtin('free', 1, _builtin_free)
        self.add_builtin('free_list', 1, _builtin_free_list)

        self.add_builtin('as_builtin', 2, _builtin_as)

        self.add_builtin('>', 2, _builtin_gt)
        self.add_builtin('<', 2, _builtin_lt)
        self.add_builtin('=<', 2, _builtin_le)
        self.add_builtin('>=', 2, _builtin_ge)
        # self.add_builtin('=\=', 2, b(_builtin_val_neq))
        # self.add_builtin('=:=', 2, b(_builtin_val_eq))
        # self.add_builtin('is', 2, SimpleBuiltIn(_builtin_is))

        self.add_builtin('obs_builtin', 2, _builtin_observation)


    def prepare(self, db, target=None):
        """Convert given logic program to suitable format for this engine.
        Calling this method is optional, but it allows to perform multiple operations on the same \
        database.
        This also executes any directives in the input model.

        :param db: logic program to prepare for evaluation
        :return: logic program in a suitable format for this engine
        :rtype: ClauseDB
        """
        result = ClauseDB.createFrom(db, builtins=self.get_builtins())
        result.engine = self

        self._process_directives(result, target=target)
        return result

    def _process_directives(self, db, target=None):
        """Process directives present in the database."""
        term = Term('_directive')
        hal_library_directive = Term.from_string(":-use_module(library(hal)).")
        db.add_clause(hal_library_directive)
        directive_node = db.find(term)

        if directive_node is None:
            return True    # no directives
        directives = db.get_node(directive_node).children
        if target==None:
            target = LogicFormulaHAL()

        while directives:
            current = directives.pop(0)
            self.execute(current, database=db, target=target, context=self.create_context((), define=None))
        return True

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

    def ground_observation(self, db, target, observation, propagate_evidence=False):
        logger = logging.getLogger('problog')
        # Ground evidence
        for query in observation:
            logger.debug("Grounding observation({},{})".format(query[0], query[1]))
            target = self.ground(db, Term('observation_builtin', query[0], query[1]), target, label=target.LABEL_OBSERVATION, is_root=True)
            logger.debug("Ground program size: %s", len(target))

        if propagate_evidence:
            with Timer('Propagating observation'):
                target.lookup_observation = {}
                obs_nodes = [node for name, node in target.observation() if node != 0 and node is not None]
                target.propagate(obs_nodes, target.lookup_observation)


    def ground_all(self, db, target=None, queries=None, evidence=None, observation=None, propagate_evidence=False, labels=None):
        if labels is None:
            labels = []
        # Initialize target if not given.
        if target is None:
            target = LogicFormula()
        db = self.prepare(db, target=target)
        logger = logging.getLogger('problog')
        with Timer('Grounding'):
            # Load queries: use argument if available, otherwise load from database.
            if queries is None:
                 #THIS is different from ProbLog, wee need to pass on target
                queries = [q[0] for q in self.query(db, Term('query', None), gp=target)]
            for query in queries:
                if not isinstance(query, Term):
                    raise GroundingError('Invalid query')   # TODO can we add a location?
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

            for ev in evidence:
                if not isinstance(ev[0], Term):
                    raise GroundingError('Invalid evidence')   # TODO can we add a location?
            for ob in observation:
                if not isinstance(ob[0], Term):
                    raise GroundingError('Invalid evidence')   # TODO can we add a location?

            # Ground queries
            if propagate_evidence:
                self.ground_evidence(db, target, evidence, propagate_evidence=propagate_evidence)
                # self.ground_evidence(db, target, evidence, propagate_evidence=False)
                self.ground_observation(db, target, observation, propagate_evidence=propagate_evidence)
                self.ground_queries(db, target, queries)
                if hasattr(target, 'lookup_evidence'):
                    logger.debug('Propagated evidence: %s' % list(target.lookup_evidence))
            else:
                self.ground_evidence(db, target, evidence)
                self.ground_observation(db, target, observation)
                self.ground_queries(db, target, queries)

        return target


def init_model(engine, model, target=None, **kwdargs):
    model = engine.prepare(model, target=target)
    evidence_facts = []
    ev_target = LogicFormulaHAL(**kwdargs)
    return model, evidence_facts, ev_target

def init_engine(**kwdargs):
    engine = EngineHAL(**kwdargs)
    return engine

@transform(LogicProgram, LogicFormulaHAL)
def groundHAL(model, target=None, **kwdargs):
    return ground(model, target=target, **kwdargs)
