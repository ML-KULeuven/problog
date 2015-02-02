
from .engine import DefaultEngine
from .formula import LogicFormula
from .logic import Term, Var, Constant, LogicProgram
from .core import transform, LABEL_QUERY, LABEL_EVIDENCE_POS, LABEL_EVIDENCE_NEG, LABEL_EVIDENCE_MAYBE
from .util import Timer
import logging

@transform(LogicProgram, LogicFormula)
def ground(model, target=None, queries=None, evidence=None) :
    return DefaultEngine().ground_all(model,target, queries=queries, evidence=evidence)
