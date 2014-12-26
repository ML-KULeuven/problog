
from .engine import DefaultEngine
from .formula import LogicFormula
from .logic import Term, Var, Constant, LogicProgram
from .core import transform, LABEL_QUERY, LABEL_EVIDENCE_POS, LABEL_EVIDENCE_NEG, LABEL_EVIDENCE_MAYBE


@transform(LogicProgram, LogicFormula)
def ground(model, target=None, queries=None, evidence=None) :
    # TODO queries should not contain Var objects. All variables should be None => no implicit identity e.g. query(X,X)
    
    engine = DefaultEngine()
    db = engine.prepare(model)
    
    if queries == None :
        queries = [ q[0] for q in engine.query(db, Term( 'query', None )) ]
    
    if evidence == None :
        evidence = engine.query(db, Term( 'evidence', None, None ))
        
    if target == None : target = LogicFormula()

    for query in queries :
        target = engine.ground(db, query, target, label=LABEL_QUERY)

    for query in evidence :
        if str(query[1]) == 'true' :
            target = engine.ground(db, query[0], target, label=LABEL_EVIDENCE_POS)
        elif str(query[1]) == 'false' :
            target = engine.ground(db, query[0], target, label=LABEL_EVIDENCE_NEG)
        else :
            target = engine.ground(db, query[0], target, label=LABEL_EVIDENCE_MAYBE)
    return target
