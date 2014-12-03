
from .engine import DefaultEngine
from .formula import LogicFormula
from .logic import Term, Var, Constant, LogicProgram
from .core import transform

def ground(model, acyclic=True) :
    
    engine = DefaultEngine()
    db = engine.prepare(model)
    
    queries = engine.query(db, Term( 'query', None ))
    evidence = engine.query(db, Term( 'evidence', None, None ))
        
    gp = LogicFormula()
    for query in queries :
        gp = engine.ground(db, query[0], gp, label='query')

    for query in evidence :
        if str(query[1]) == 'true' :
            gp = engine.ground(db, query[0], gp, label='evidence')
        else :
            gp = engine.ground(db, query[0], gp, label='-evidence')

    if acyclic :
        return gp.makeAcyclic()
    else :
        return gp
        
@transform(LogicProgram, LogicFormula)
def ground_auto(model, gp) :
    
    engine = DefaultEngine()
    db = engine.prepare(model)
    
    queries = engine.query(db, Term( 'query', None ))
    evidence = engine.query(db, Term( 'evidence', None, None ))
        
    for query in queries :
        gp = engine.ground(db, query[0], gp, label='query')

    for query in evidence :
        if str(query[1]) == 'true' :
            gp = engine.ground(db, query[0], gp, label='evidence')
        else :
            gp = engine.ground(db, query[0], gp, label='-evidence')

    return gp
        
