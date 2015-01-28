
from .engine import DefaultEngine
from .formula import LogicFormula
from .logic import Term, Var, Constant, LogicProgram
from .core import transform, LABEL_QUERY, LABEL_EVIDENCE_POS, LABEL_EVIDENCE_NEG, LABEL_EVIDENCE_MAYBE
from .util import Timer
import logging
import imp, os, inspect

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

        # Load external (python) files that are referenced in the model
        externals = {}
        filenames = [ q[0] for q in engine.query(db, Term( 'load_external', None )) ]
        for filename in filenames:
            filename = os.path.join(model.source_root, filename.value.replace('"',''))
            if not os.path.exists(filename):
              raise InvalidEngineState('External file not found: {}'.format(filename))
            with open(filename, 'r') as extfile:
                ext = imp.load_module('externals', extfile, filename, ('.py', 'U', 1))
                for func_name, func in inspect.getmembers(ext, inspect.isfunction):
                    externals[func_name] = func
        engine.addExternalCalls(externals)

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
