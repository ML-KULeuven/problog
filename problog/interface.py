
from .engine import DefaultEngine
from .formula import LogicFormula
from .logic import Term, Var, Constant, LogicProgram
from .core import transform
from .util import Timer
import logging

@transform(LogicProgram, LogicFormula)
def ground(model, target=None, queries=None, evidence=None) :
    return DefaultEngine().ground_all(model,target, queries=queries, evidence=evidence)
