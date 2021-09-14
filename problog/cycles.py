"""
problog.cycles - Cycle-breaking
-------------------------------

Cycle breaking in propositional formulae.

..
    Part of the ProbLog distribution.

    Copyright 2015 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import logging
from collections import defaultdict

from .core import transform
from .formula import LogicFormula, LogicDAG
from .logic import Term
from .util import Timer

cycle_var_prefix = "problog_cv_"

# noinspection PyUnusedLocal
@transform(LogicFormula, LogicDAG)
def break_cycles(source, target, translation=None, keep_named=False, **kwdargs):
    """Break cycles in the source logic formula.

    :param source: logic formula with cycles
    :param target: target logic formula without cycles
    :param keep_named: if true, then named nodes will be preserved after cycle breaking
    :param kwdargs: additional arguments (ignored)
    :return: target
    """

    logger = logging.getLogger("problog")
    with Timer("Cycle breaking"):
        cycles_broken = set()
        content = set()
        if translation is None:
            translation = defaultdict(list)

        labeled = source.labeled()
        if keep_named:
            for name, node, label in source.get_names_with_label():
                if label == source.LABEL_NAMED:
                    labeled.append((name, node, label))
        for q, n, l in labeled:
            if source.is_probabilistic(n):
                newnode = _break_cycles(
                    source, target, n, [], cycles_broken, content, translation
                )
            else:
                newnode = n
            target.add_name(q, newnode, l)

        # TODO copy constraints

        translation = defaultdict(list)
        for q, n, v in source.evidence_all():
            if source.is_probabilistic(n):
                newnode = _break_cycles(
                    source,
                    target,
                    abs(n),
                    [],
                    cycles_broken,
                    content,
                    translation,
                    is_evidence=True,
                )
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

        logger.debug("Ground program size: %s", len(target))
        return target


def _break_cycles(
    source,
    target,
    nodeid,
    ancestors,
    cycles_broken,
    content,
    translation,
    is_evidence=False,
):
    negative_node = nodeid < 0
    nodeid = abs(nodeid)

    if not is_evidence and not source.is_probabilistic(
        source.get_evidence_value(nodeid)
    ):
        ev = source.get_evidence_value(nodeid)
        if negative_node:
            return target.negate(ev)
        else:
            return ev
    elif nodeid in ancestors:
        cycles_broken.add(nodeid)
        return None  # cyclic node: node is False
    elif nodeid in translation:
        ancset = frozenset(ancestors + [nodeid])
        for newnode, cb, cn in translation[nodeid]:
            # We can reuse this previous node iff
            #   - no more cycles have been broken that should not be broken now
            #       (cycles broken is a subset of ancestors)
            #   - no more cycles should be broken than those that have been broken in the previous
            #       (the previous node does not contain ancestors)

            if cb <= ancset and not ancset & cn:
                cycles_broken |= cb
                content |= cn
                if negative_node:
                    return target.negate(newnode)
                else:
                    return newnode

    child_cycles_broken = set()
    child_content = set()

    node = source.get_node(nodeid)
    nodetype = type(node).__name__
    if nodetype == "atom":
        newnode = target.add_atom(
            node.identifier,
            node.probability,
            group=node.group,
            name=node.name,
            is_extra=node.is_extra,
        )
    else:
        children = [
            _break_cycles(
                source,
                target,
                child,
                ancestors + [nodeid],
                child_cycles_broken,
                child_content,
                translation,
                is_evidence,
            )
            for child in node.children
        ]
        newname = node.name
        if newname is not None and child_cycles_broken:
            newfunc = (
                cycle_var_prefix + newname.functor + "_cb_" + str(len(translation[nodeid]))
            )
            newname = Term(newfunc, *newname.args)
        if nodetype == "conj":
            newnode = target.add_and(children, name=newname)
        else:
            newnode = target.add_or(children, name=newname)

        if target.is_probabilistic(newnode):
            # Don't add the node if it is None
            # Also: don't add atoms (they can't be involved in cycles)
            content.add(nodeid)

    translation[nodeid].append(
        (newnode, child_cycles_broken, child_content - child_cycles_broken)
    )
    content |= child_content
    cycles_broken |= child_cycles_broken

    if negative_node:
        return target.negate(newnode)
    else:
        return newnode
