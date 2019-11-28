"""
problog.forward - Forward compilation and evaluation
----------------------------------------------------

Forward compilation using TP-operator.

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
import copy
import logging
import random
import signal
import time
from collections import defaultdict

from .bdd_formula import BDD
from .core import transform
from .core import transform_create_as
from .dd_formula import DD
from .dd_formula import build_dd
from .evaluator import Evaluator, EvaluatableDSP, InconsistentEvidenceError
from .formula import LogicFormula, OrderedSet, atom
from .sdd_formula import SDD
from .util import UHeap


def timeout_handler(signum, frame):
    raise SystemError("Process timeout (Python) [%s]" % signum)


class ForwardInference(DD):
    def __init__(self, compile_timeout=None, **kwdargs):
        super(ForwardInference, self).__init__(auto_compact=False, **kwdargs)

        self._inodes_prev = None
        self._inodes_old = None
        self._inodes_neg = None
        self._facts = None
        self._atoms_in_rules = None
        self._completed = None

        self.timeout = compile_timeout

        self._update_listeners = []

        self._node_depths = None
        self.evidence_node = 0

    def register_update_listener(self, obj):
        self._update_listeners.append(obj)

    def _create_atom(
        self, identifier, probability, group, name=None, source=None, is_extra=False
    ):
        return atom(identifier, probability, group, name, source, is_extra)

    def is_complete(self, node):
        node = abs(node)
        return self._completed[node - 1]

    def set_complete(self, node):
        self._completed[node - 1] = True

    def init_build(self):
        if self.evidence():
            ev = [n for q, n in self.evidence() if n is None or n != 0]
            if ev:
                if len(ev) == 1:
                    self.evidence_node = ev[0]
                else:
                    self.evidence_node = self.add_and(ev)
            else:
                # Only deterministically true evidence
                self.evidence_node = 0

        self._facts = []  # list of facts
        self._atoms_in_rules = defaultdict(
            OrderedSet
        )  # lookup all rules in which an atom is used
        self._completed = [False] * len(self)

        self._compute_node_depths()
        for index, node, nodetype in self:
            if self._node_depths[index - 1] is not None:
                # only include nodes that are reachable from a query or evidence
                if nodetype == "atom":  # it's a fact
                    self._facts.append(index)
                    self.set_complete(index)
                else:  # it's a compound
                    for atom in node.children:
                        self._atoms_in_rules[abs(atom)].add(index)
        self.build_constraint_dd()
        self.inodes = [None] * len(self)
        self._inodes_prev = [None] * len(self)
        self._inodes_old = [None] * len(self)
        self._inodes_neg = [None] * len(self)
        self._compute_minmax_depths()

    def _propagate_complete(self, interrupted=False):
        if not interrupted:
            for i, c in enumerate(self._completed):
                if not c:
                    self._completed[i] = True
                    self.notify_node_completed(i + 1)
        else:
            updated_nodes = set([(i + 1) for i, c in enumerate(self._completed) if c])
            while updated_nodes:
                next_updates = set()
                # Find all heads that are affected
                affected_nodes = set()
                for node in updated_nodes:
                    for rule in self._atoms_in_rules[node]:
                        if not self.is_complete(rule):
                            affected_nodes.add(rule)
                for head in affected_nodes:
                    # head must be compound
                    node = self.get_node(head)
                    children = [self.is_complete(c) for c in node.children]
                    if False not in children:
                        self.is_complete(head)
                        self.notify_node_completed(head)
                        next_updates.add(head)
                updated_nodes = next_updates

    def _compute_node_depths(self):
        """Compute node depths in breadth-first manner."""
        self._node_depths = [None] * len(self)
        self._node_levels = []
        # Start with current nodes
        current_nodes = set(
            abs(n) for q, n, l in self.labeled() if self.is_probabilistic(n)
        )
        if self.is_probabilistic(self.evidence_node):
            current_nodes.add(abs(self.evidence_node))
        current_level = 0
        while current_nodes:
            self._node_levels.append(current_nodes)
            next_nodes = set()
            for index in current_nodes:
                self._node_depths[index - 1] = current_level
                node = self.get_node(index)
                nodetype = type(node).__name__
                if nodetype != "atom":
                    for c in node.children:
                        if self.is_probabilistic(c):
                            if self._node_depths[abs(c) - 1] is None:
                                next_nodes.add(abs(c))
            current_nodes = next_nodes
            current_level += 1

    def _compute_minmax_depths(self):
        self._node_minmax = [None] * len(self)
        for level, nodes in reversed(list(enumerate(self._node_levels))):
            for index in nodes:
                # Get current node's minmax
                minmax = self._node_minmax[index - 1]
                if minmax is None:
                    minmax = level
                for rule in self._atoms_in_rules[index]:
                    rule_minmax = self._node_minmax[rule - 1]
                    if rule_minmax is None:
                        self._node_minmax[rule - 1] = minmax
                    else:
                        node = self.get_node(rule)
                        nodetype = type(node).__name__
                        if nodetype == "conj":
                            rule_minmax = max(minmax, rule_minmax)
                        else:  # disj
                            rule_minmax = min(minmax, rule_minmax)
                        self._node_minmax[rule - 1] = rule_minmax

    def _update_minmax_depths(self, index, new_minmax=0):
        """Update the minmax depth data structure when the given node is completed.

        :param index:
        :return:
        """
        current_minmax = self._node_minmax[index - 1]
        self._node_minmax[index - 1] = new_minmax

        for parent in self._atoms_in_rules[index]:
            parent_minmax = self._node_minmax[parent - 1]

            if current_minmax == parent_minmax:
                # Current node is best child => we need to recompute
                parent_node = self.get_node(parent)
                parent_nodetype = type(parent_node).__name__
                parent_children_minmax = [
                    self._node_minmax[c - 1]
                    for c in parent_node.children
                    if not self.is_complete(c)
                ]
                if not parent_children_minmax:
                    # No incomplete children
                    self.set_complete(parent)
                    parent_minmax = 0
                elif parent_nodetype == "conj":
                    parent_minmax == max(parent_children_minmax)
                else:
                    parent_minmax == min(parent_children_minmax)
                self._update_minmax_depths(parent, parent_minmax)

    def sort_nodes(self, nodes):
        return sorted(nodes, key=lambda i: self._node_depths[i - 1])

    def notify_node_updated(self, node, complete):
        for obj in self._update_listeners:
            obj.node_updated(self, node, complete)

    def notify_node_completed(self, node):
        for obj in self._update_listeners:
            obj.node_completed(self, node)

    def _heuristic_key_depth(self, node):
        # For OR: D(n) is min(D(c) for c in children)
        # For AND: D(n) is max(D(c) for c in children)
        return self._node_minmax[node - 1], self._node_depths[node - 1], random.random()

    def _heuristic_key(self, node):
        return self._heuristic_key_depth(node)

    def build_iteration(self, updated_nodes):
        to_recompute = UHeap(key=self._heuristic_key)
        for node in updated_nodes:
            for rule in self._atoms_in_rules[node]:
                to_recompute.push(rule)

        # nodes_to_recompute should be an updateable heap without duplicates
        while to_recompute:
            key, node = to_recompute.pop_with_key()
            if self.update_inode(node):  # The node has changed
                # Find rules that may be affected
                for rule in self._atoms_in_rules[node]:
                    to_recompute.push(rule)
                # Notify listeners that node was updated
                self.notify_node_updated(node, self.is_complete(node))
            elif self.is_complete(node):
                self.notify_node_completed(node)
                # if self.is_complete(node):
                #     self._update_minmax_depths(node)

    def build_iteration_levelwise(self, updated_nodes):
        while updated_nodes:
            next_updates = OrderedSet()
            # Find all heads that are affected
            affected_nodes = OrderedSet()
            for node in updated_nodes:
                for rule in self._atoms_in_rules[node]:
                    affected_nodes.add(rule)
            affected_nodes = self.sort_nodes(affected_nodes)
            # print (affected_nodes, [self._node_depths[i-1] for i in affected_nodes])
            for head in affected_nodes:
                if self.update_inode(head):
                    next_updates.add(head)
                    self.notify_node_updated(head, self.is_complete(head))
                elif self.is_complete(head):
                    self.notify_node_completed(head)
            updated_nodes = next_updates

    def build_stratum(self, updated_nodes):
        self.build_iteration(updated_nodes)
        updated_nodes = OrderedSet()
        for i, nodes in enumerate(zip(self.inodes, self._inodes_old)):
            if not self.get_manager().same(*nodes):
                updated_nodes.add(i + 1)
                # self.notify_node_updated(i + 1)
        self.get_manager().ref(*filter(None, self.inodes))
        self.get_manager().deref(*filter(None, self._inodes_prev))
        self.get_manager().deref(*filter(None, self._inodes_neg))

        # Only completed nodes should be used for negation in the next stratum.
        self._inodes_old = self.inodes[:]
        self._inodes_prev = [None] * len(self)
        for i, n in enumerate(self.inodes):
            if self._completed[i]:
                self._inodes_prev[i] = n
        self._inodes_neg = [None] * len(self)

        return updated_nodes

    def build_dd(self):
        required_nodes = set(
            [abs(n) for q, n, l in self.labeled() if self.is_probabilistic(n)]
        )
        required_nodes |= set(
            [abs(n) for q, n, v in self.evidence_all() if self.is_probabilistic(n)]
        )
        if self.timeout:
            # signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            signal.signal(signal.SIGALRM, timeout_handler)
            logging.getLogger("problog").info("Set timeout:", self.timeout)
        try:
            self.init_build()
            updated_nodes = OrderedSet(self._facts)
            while updated_nodes:
                # TODO only check nodes that are actually used in negation
                updated_nodes = self.build_stratum(updated_nodes)
            self._propagate_complete(False)
        except SystemError as err:
            self._propagate_complete(True)
            logging.getLogger("problog").warning(err)
        except KeyboardInterrupt as err:
            self._propagate_complete(True)
            logging.getLogger("problog").warning(err)

        signal.alarm(0)
        self.build_constraint_dd()

    def current(self):
        destination = LogicFormula(auto_compact=False)
        source = self
        # TODO maintain a translation table
        for i, n, t in source:
            inode = self.get_inode(i)
            if inode is not None:
                inode = int(inode)
            if t == "atom":
                j = destination.add_atom(
                    n.identifier,
                    n.probability,
                    n.group,
                    name=inode,
                    is_extra=n.is_extra,
                )
            elif t == "conj":
                children = [c for c in n.children if self.get_inode(c) is not None]
                j = destination.add_and(children, name=inode)
            elif t == "disj":
                children = [c for c in n.children if self.get_inode(c) is not None]
                j = destination.add_or(children, name=inode)
            else:
                raise TypeError("Unknown node type")
            assert i == j

        for name, node, label in source.get_names_with_label():
            if label != self.LABEL_NAMED:
                destination.add_name(name, node, label)

        for c in source.constraints():
            if c.is_nontrivial():
                destination.add_constraint(c)
        return destination

    def update_inode(self, index):
        """Recompute the inode at the given index."""

        was_complete = self.is_complete(index)

        oldnode = self.get_inode(index)
        node = self.get_node(index)
        assert index > 0
        nodetype = type(node).__name__
        if nodetype == "conj":
            children = [self.get_inode(c) for c in node.children]
            children_complete = [self.is_complete(c) for c in node.children]
            if None in children:
                newnode = None  # don't compute if some children are still unknown
            else:
                newnode = self.get_manager().conjoin(*children)
            if False not in children_complete:
                self.set_complete(index)

        elif nodetype == "disj":
            children = [self.get_inode(c) for c in node.children]
            children_complete = [self.is_complete(c) for c in node.children]
            children = list(
                filter(None, children)
            )  # discard children that are still unknown
            if children:
                newnode = self.get_manager().disjoin(*children)
            else:
                newnode = None
            if False not in children_complete:
                self.set_complete(index)

        else:
            raise TypeError("Unexpected node type.")

        # Add constraints
        if newnode is not None:
            newernode = self.get_manager().conjoin(newnode, self.get_constraint_inode())
            self.get_manager().deref(newnode)
            newnode = newernode

        if self.get_manager().same(oldnode, newnode):
            return self.is_complete(index) != was_complete  # no change occurred
        else:
            if oldnode is not None:
                self.get_manager().deref(oldnode)
            self.set_inode(index, newnode)
            return True

    def get_evidence_inode(self):
        if not self.is_probabilistic(self.evidence_node):
            return self.get_manager().true()
        else:
            inode = self.get_inode(self.evidence_node, final=True)
            if inode:
                return inode
            else:
                return self.get_manager().true()

    def get_inode(self, index, final=False):
        """
        Get the internal node corresponding to the entry at the given index.
        :param index: index of node to retrieve
        :return: SDD node corresponding to the given index
        :rtype: SDDNode
        """
        assert self.is_probabilistic(index)
        node = self.get_node(abs(index))
        if type(node).__name__ == "atom":
            av = self.atom2var.get(abs(index))
            if av is None:
                av = self.get_manager().add_variable()
                self.atom2var[abs(index)] = av
                self.var2atom[av] = abs(index)
            result = self.get_manager().literal(av)
            if index < 0:
                return self.get_manager().negate(result)
            else:
                return result
        elif index < 0 and not final:
            # We are requesting a negated node => use previous stratum's result
            result = self._inodes_neg[-index - 1]
            if result is None and self._inodes_prev[-index - 1] is not None:
                result = self.get_manager().negate(self._inodes_prev[-index - 1])
                self._inodes_neg[-index - 1] = result
            return result
        elif index < 0:
            return self.get_manager().negate(self.inodes[-index - 1])
        else:
            return self.inodes[index - 1]

    def set_inode(self, index, node):
        """Set the internal node for the given index.

        :param index: index at which to set the new node
        :type index: int > 0
        :param node: new node
        """
        assert index is not None
        assert index > 0
        self.inodes[index - 1] = node

    def add_constraint(self, c):
        LogicFormula.add_constraint(self, c)


class _ForwardSDD(SDD, ForwardInference):
    transform_preference = 1000

    def __init__(self, sdd_auto_gc=False, **kwdargs):
        SDD.__init__(self, sdd_auto_gc=sdd_auto_gc, **kwdargs)
        ForwardInference.__init__(self, **kwdargs)

    @classmethod
    def is_available(cls):
        return SDD.is_available()

    def to_explicit_encoding(self):
        """
        Transform the current implicit encoding to an SDD with explicit encoding. The latter will contain indicator
        variables for inferred literals and contains both the 'true and false' case. For example, while the implicit
        encoding for c :- a,b will result in c = a * b., the explicit encoding will result in (c * a * b) + (-c * (-a + -b))
        In the explicit encoding, evidence can be incorporated as changing the weights and a negation can also easily be
        queried. Furthermore, the explicit encoding implementation can guarantee a circuit with one root instead of
        multiple.

        :return: The explicit encoding of this formula.
        :rtype: SDDExplicit
        """
        from .sdd_formula_explicit import build_explicit_from_forwardsdd
        from .sdd_formula_explicit import SDDExplicit

        return build_explicit_from_forwardsdd(source=self, destination=SDDExplicit())

    def copy_to_noref(self, destination):
        """
        Copy the relevant data structures without any refcounts to the new inodes.
        Because of this, the destination should have auto_gc disabled.
        :param destination: The DD to copy the data structures' references to.
        :type destination: DD
        """
        destination.atom2var = self.atom2var.copy()
        destination.var2atom = self.var2atom.copy()

        old = self.inode_manager.nodes
        self.inode_manager.nodes = self.inodes
        destination.inode_manager = self.inode_manager.get_deepcopy_noref()
        self.inode_manager.nodes = old

        destination._atomcount = self._atomcount
        destination._weights = self._weights.copy()
        destination._constraints = self._constraints.copy()
        destination.evidence_node = self.evidence_node
        destination._nodes = self._nodes.copy()
        destination._index_atom = self._index_atom.copy()
        destination._index_next = self._index_next
        destination._index_conj = self._index_conj.copy()
        destination._index_disj = self._index_disj.copy()
        destination._names = copy.deepcopy(self._names)
        destination._constraints_me = copy.deepcopy(self._constraints_me)


class _ForwardBDD(BDD, ForwardInference):
    transform_preference = 1000

    def __init__(self, **kwdargs):
        BDD.__init__(self, **kwdargs)
        ForwardInference.__init__(self, **kwdargs)

    @classmethod
    def is_available(cls):
        return BDD.is_available()


@transform(LogicFormula, _ForwardSDD)
def build_sdd(source, destination, **kwdargs):
    result = build_dd(source, destination, **kwdargs)
    return result


@transform(LogicFormula, _ForwardBDD)
def build_sdd(source, destination, **kwdargs):
    result = build_dd(source, destination, **kwdargs)
    return result


class ForwardSDD(LogicFormula, EvaluatableDSP):
    transform_preference = 30

    def __init__(self, **kwargs):
        LogicFormula.__init__(self, **kwargs)
        EvaluatableDSP.__init__(self)
        self.kwargs = kwargs
        self.internal = _ForwardSDD(**self.kwargs)

    @classmethod
    def is_available(cls):
        return SDD.is_available()

    def _create_evaluator(self, semiring, weights, **kwargs):
        return ForwardEvaluator(self, semiring, self.internal, weights, **kwargs)

    def to_formula(self):
        build_dd(self, self.internal)
        return self.internal.to_formula()


class ForwardBDD(LogicFormula, EvaluatableDSP):
    transform_preference = 40

    def __init__(self, **kwargs):
        LogicFormula.__init__(self, **kwargs)
        EvaluatableDSP.__init__(self)
        self.kwargs = kwargs

    @classmethod
    def is_available(cls):
        return BDD.is_available()

    def _create_evaluator(self, semiring, weights, **kwargs):
        return ForwardEvaluator(
            self, semiring, _ForwardBDD(**self.kwargs), weights, **kwargs
        )


# Inform the system that we can create a ForwardFormula in the same way as a LogicFormula.
transform_create_as(ForwardSDD, LogicFormula)
transform_create_as(ForwardBDD, LogicFormula)


class ForwardEvaluator(Evaluator):
    """An evaluator using anytime forward compilation."""

    def __init__(self, formula, semiring, fdd, weights=None, verbose=None, **kwargs):
        Evaluator.__init__(self, formula, semiring, weights, **kwargs)

        self.fsdd = fdd
        self._z = None
        self._verbose = verbose

        self._results = {}
        self._complete = set()

        self._start_time = None

    def node_updated(self, source, node, complete):
        is_evidence = node == abs(source.evidence_node)
        name = [
            n
            for n, i, l in self.formula.labeled()
            if source.is_probabilistic(i) and abs(i) == node
        ]
        if is_evidence:
            name = ("evidence",)
        if name:
            name = name[0]
            weights = {}
            for atom, weight in self.weights.items():
                av = source.atom2var.get(atom)
                if av is not None:
                    weights[av] = weight
            inode = source.get_inode(node, final=True)
            if inode is not None:
                enode = source.get_manager().conjoin(
                    source.get_evidence_inode(), source.get_constraint_inode()
                )
                if self.fsdd.get_manager().is_false(enode):
                    raise InconsistentEvidenceError(context=" during compilation")

                qnode = source.get_manager().conjoin(inode, enode)
                tvalue = source.get_manager().wmc(enode, weights, self.semiring)
                value = source.get_manager().wmc(qnode, weights, self.semiring)
                result = self.semiring.normalize(value, tvalue)
                self._results[node] = result

                debug_msg = "update query %s: %s after %ss" % (
                    name,
                    self.semiring.result(result, self.formula),
                    "%.4f" % (time.time() - self._start_time),
                )
                logging.getLogger("problog").debug(debug_msg)

            if complete:
                self._complete.add(node)
        # if is_evidence:
        #     for c in self._complete:
        #         if c != node:
        #             self.node_updated(source, c, complete)

    def node_completed(self, source, node):
        qs = set(
            abs(qi) for qn, qi, ql in source.labeled() if source.is_probabilistic(qi)
        )
        if node in qs:
            self._complete.add(node)

    def initialize(self):
        self.weights = self.formula.extract_weights(self.semiring, self.given_weights)

        # We should do all compilation here.
        self.fsdd.register_update_listener(self)
        self._start_time = time.time()
        if len(self.fsdd) == 0:
            if self.fsdd.init_varcount == -1:
                self.fsdd.init_varcount = self.formula.atomcount
            build_dd(self.formula, self.fsdd)

        # Update weights with constraints and evidence
        enode = self.fsdd.get_manager().conjoin(
            self.fsdd.get_evidence_inode(), self.fsdd.get_constraint_inode()
        )

        if self.fsdd.get_manager().is_false(enode):
            raise InconsistentEvidenceError(context=" during compilation")

        # Make sure all atoms exist in atom2var.
        for name, node, label in self.fsdd.labeled():
            if self.fsdd.is_probabilistic(node):
                self.fsdd.get_inode(node)

        weights = {}
        for atom, weight in self.weights.items():
            av = self.fsdd.atom2var.get(atom)
            if av is not None:
                weights[av] = weight
            elif atom == 0:
                weights[0] = weight

        for name, node, label in self.fsdd.labeled():
            if self.fsdd.is_probabilistic(node):
                inode = self.fsdd.get_inode(node)
                qnode = self.fsdd.get_manager().conjoin(inode, enode)
                tvalue = self.fsdd.get_manager().wmc(enode, weights, self.semiring)
                value = self.fsdd.get_manager().wmc(qnode, weights, self.semiring)
                result = self.semiring.normalize(value, tvalue)
            elif self.fsdd.is_true(node):
                result = self.semiring.one()
            else:
                result = self.semiring.zero()
            self._results[node] = result

    def propagate(self):
        self.initialize()

    def evaluate(self, index):
        """Compute the value of the given node."""
        # We should get results from cache here.

        ub = 1.0
        if index is None:
            return 0.0
        elif index == 0:
            if not self.semiring.is_nsp():
                return self.semiring.result(self.semiring.one(), self.formula)
            else:
                weights = {}
                for atom, weight in self.weights.items():
                    av = self.fsdd.atom2var.get(atom)
                    if av is not None:
                        weights[av] = weight
                    elif atom == 0:
                        weights[0] = weight
                enode = self.fsdd.get_manager().conjoin(
                    self.fsdd.get_evidence_inode(), self.fsdd.get_constraint_inode()
                )
                tvalue = self.fsdd.get_manager().wmc(enode, weights, self.semiring)
                result = self.semiring.normalize(tvalue, tvalue)
                return self.semiring.result(result, self.formula)
        else:
            n = self.formula.get_node(abs(index))
            nt = type(n).__name__
            if nt == "atom":
                wp = self._results[index]
                # wp, wn = self.weights.get(abs(index))
                if index < 0:
                    # wn = self.semiring.negate(wp)
                    return self.semiring.result(wp, self.formula)
                else:
                    return self.semiring.result(wp, self.formula)
            else:
                # TODO report correct bounds in case of evidence
                if index < 0:
                    if -index in self._results:
                        if -index in self._complete:
                            return self.semiring.result(
                                self.semiring.negate(self._results[-index]),
                                self.formula,
                            )
                        else:
                            return (
                                self.semiring.result(self.semiring.zero()),
                                self.semiring.result(
                                    self.semiring.negate(self._results[-index]),
                                    self.formula,
                                ),
                            )
                    else:
                        return (
                            self.semiring.result(self.semiring.zero()),
                            self.semiring.result(self.semiring.one()),
                        )
                else:
                    if index in self._results:
                        if index in self._complete:
                            return self.semiring.result(
                                self._results[index], self.formula
                            )
                        else:
                            return (
                                self.semiring.result(
                                    self._results[index], self.formula
                                ),
                                self.semiring.result(self.semiring.one()),
                            )
                    else:
                        return (
                            self.semiring.result(self.semiring.zero()),
                            self.semiring.result(self.semiring.one()),
                        )

    def evaluate_evidence(self):
        raise NotImplementedError("Evaluator.evaluate_evidence is an abstract method.")

    # def add_evidence(self, node):
    #     """Add evidence"""
    #     warnings.warn('Evidence is not supported by this evaluation method and will be ignored.')

    def has_evidence(self):
        return self.__evidence != []

    def clear_evidence(self):
        self.__evidence = []

    def evidence(self):
        return iter(self.__evidence)
