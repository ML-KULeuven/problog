
from .engine import DefaultEngine
from .formula import LogicFormula
from .logic import Term, Var, Constant, LogicProgram
from .core import transform, LABEL_QUERY, LABEL_EVIDENCE_POS, LABEL_EVIDENCE_NEG, LABEL_EVIDENCE_MAYBE
from .util import Timer
import logging

@transform(LogicProgram, LogicFormula)
def ground(model, target=None, queries=None, evidence=None) :
    # TODO queries should not contain Var objects. All variables should be None => no implicit identity e.g. query(X,X)
    
    logger = logging.getLogger('problog')
    
    with Timer('Grounding'):
        engine = DefaultEngine()
        db = engine.prepare(model)
        
        if queries == None :
            queries = [ q[0] for q in engine.query(db, Term( 'query', None )) ]
        
        if evidence == None :
            evidence = engine.query(db, Term( 'evidence', None, None ))
        
        if target == None : target = LogicFormula()
        
        for query in queries :
            logger.debug("Grounding query '%s'", query)
            target = engine.ground(db, query, target, label=LABEL_QUERY)
            logger.debug("Ground program size: %s", len(target))
            
        for query in evidence :
            if str(query[1]) == 'true' :
                logger.debug("Grounding evidence '%s'", query[0])
                target = engine.ground(db, query[0], target, label=LABEL_EVIDENCE_POS)
                logger.debug("Ground program size: %s", len(target))
            elif str(query[1]) == 'false' :
                logger.debug("Grounding evidence '%s'", query[0])
                target = engine.ground(db, query[0], target, label=LABEL_EVIDENCE_NEG)
                logger.debug("Ground program size: %s", len(target))
            else :
                logger.debug("Grounding evidence '%s'", query[0])
                target = engine.ground(db, query[0], target, label=LABEL_EVIDENCE_MAYBE)
                logger.debug("Ground program size: %s", len(target))

    return target
