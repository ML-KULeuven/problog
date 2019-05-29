from __future__ import print_function

from .formula import LogicFormulaHAL

from problog.formula import LogicDAG
from problog.logic import Term
from problog.core import transform
from problog.util import Timer
from problog.cycles import _break_cycles

from collections import defaultdict
import logging


# noinspection PyUnusedLocal
@transform(LogicFormulaHAL, LogicDAG)
def break_cycles(source, target, translation=None, **kwdargs):
    """Break cycles in the source logic formula.

    :param source: logic formula with cycles
    :param target: target logic formula without cycles
    :param kwdargs: additional arguments (ignored)
    :return: target
    """

    logger = logging.getLogger('problog')
    with Timer('Cycle breaking'):
        cycles_broken = set()
        content = set()
        if translation is None:
            translation = defaultdict(list)

        for q, n, l in source.labeled():
            if source.is_probabilistic(n):
                newnode = _break_cycles(source, target, n, [], cycles_broken, content, translation)
            else:
                newnode = n
            target.add_name(q, newnode, l)

        # TODO copy constraints

        translation = defaultdict(list)
        for q, n, v in source.evidence_all():
            if source.is_probabilistic(n):
                newnode = _break_cycles(source, target, abs(n), [], cycles_broken,
                                        content, translation, is_evidence=True)
            else:
                newnode = n
            if n is not None and n < 0:
                newnode = target.negate(newnode)
            if v > 0:
                target.add_name(q, newnode, target.LABEL_EVIDENCE_POS)
            elif v < 0:
                target.add_name(q, newnode, target.LABEL_EVIDENCE_NEG)
            else:
                target.add_name(q, newnode, target.LABEL_EVIDENCE_MAYBE)

        translation = defaultdict(list)
        for q, n, v in source.observation_all():
            if source.is_probabilistic(n):
                newnode = _break_cycles(source, target, abs(n), [], cycles_broken,
                                        content, translation, is_evidence=True)
            else:
                newnode = n
            target.add_name(q, newnode, target.LABEL_OBSERVATION)

        logger.debug("Ground program size: %s", len(target))
        return target
