"""
problog.dd_formula - Decision Diagrams
--------------------------------------

Common interface to decision diagrams (BDD, SDD).

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

from __future__ import print_function


from .util import Timer, mktempfile
from .formula import LogicFormula, atom, LogicNNF
from .evaluator import (
    EvaluatableDSP,
    Evaluator,
    FormulaEvaluatorNSP,
    SemiringLogProbability,
    SemiringProbability,
)
from .errors import InconsistentEvidenceError


class DD(LogicFormula, EvaluatableDSP):
    """Root class for bottom-up compiled decision diagrams."""

    def __init__(self, **kwdargs):
        LogicFormula.__init__(self, **kwdargs)

        self.inode_manager = None

        self.atom2var = {}  # index to node
        self.var2atom = {}

        # self._constraint_dd = None

    def _create_manager(self):
        """Create and return a new underlying manager."""
        raise NotImplementedError("Abstract method.")

    def get_manager(self):
        """Get the underlying manager"""
        if self.inode_manager is None:
            self.inode_manager = self._create_manager()
        return self.inode_manager

    def _create_atom(self, identifier, probability, group, name=None, source=None):
        index = len(self) + 1
        var = self.get_manager().add_variable()
        self.atom2var[index] = var
        self.var2atom[var] = index
        return atom(identifier, probability, group, name, source)

    def get_inode(self, index):
        """Get the internal node corresponding to the entry at the given index.

        :param index: index of node to retrieve
        :return: internal node corresponding to the given index
        """
        mgr = self.get_manager()  # type: DDManager
        if self.is_false(index):
            return mgr.false()
        elif self.is_true(index):
            return mgr.true()

        negate = index < 0
        index = abs(index)
        node = self.get_node(index)

        if type(node).__name__ == "atom":
            result = mgr.literal(self.atom2var[index])
        else:
            # Extend list
            while len(mgr.nodes) < index:
                mgr.nodes.append(None)
            result = mgr.nodes[index - 1]
            if result is None:
                result = self._create_inode(node)
                mgr.nodes[index - 1] = result
        return mgr.negate(result) if negate else result

    def _create_inode(
        self, node
    ):  # TODO: Recursion is slow in python. Change build_dd to not use this and get_inode?
        if type(node).__name__ == "conj":
            return self.get_manager().conjoin(
                *[self.get_inode(c) for c in node.children]
            )
        else:
            return self.get_manager().disjoin(
                *[self.get_inode(c) for c in node.children]
            )

    def set_inode(self, index, node):
        """Set the internal node for the given index.

        :param index: index at which to set the new node
        :type index: int > 0
        :param node: new node
        """
        assert index is not None
        assert index > 0
        self.get_manager().nodes[index - 1] = node

    def get_constraint_inode(self):
        """Get the internal node representing the constraints for this formula."""
        if self.get_manager().constraint_dd is None:
            return self.get_manager().true()
        else:
            return self.get_manager().constraint_dd

    def _create_evaluator(self, semiring, weights, **kwargs):
        if semiring.is_nsp():
            return FormulaEvaluatorNSP(self.to_formula(), semiring, weights)
        else:
            return DDEvaluator(self, semiring, weights, **kwargs)

    def build_dd(self):
        """Build the internal representation of the formula."""
        required_nodes = set(
            [abs(n) for q, n, l in self.labeled() if self.is_probabilistic(n)]
        )
        required_nodes |= set(
            [abs(n) for q, n, l in self.labeled() if self.is_probabilistic(n)]
        )  # TODO self.evidence_all() ipv self.labeled() ? see forward.py

        for n in required_nodes:
            self.get_inode(n)

        self.build_constraint_dd()

    def build_constraint_dd(self):
        """Build the internal representation of the constraint of this formula."""
        self.get_manager().constraint_dd = self.get_manager().true()
        for c in self.constraints():
            for rule in c.as_clauses():
                rule_sdd = self.get_manager().disjoin(
                    *[self.get_inode(r) for r in rule]
                )
                new_constraint_dd = self.get_manager().conjoin(
                    self.get_manager().constraint_dd, rule_sdd
                )
                self.get_manager().deref(self.get_manager().constraint_dd)
                self.get_manager().deref(rule_sdd)
                self.get_manager().constraint_dd = new_constraint_dd

    def to_dot(self, *args, **kwargs):
        if kwargs.get("use_internal"):
            dot_text = [
                "digraph {\n",
                "overlap=false\n",
            ]  # TODO Support multiple nodes in write_to_dot
            for qn, qi in self.queries():
                filename = mktempfile(".dot")
                self.get_manager().write_to_dot(self.get_inode(qi), filename)
                with open(filename) as f:
                    dot_text += f.readlines()[4:-1]

            dot_text.append("}")
            return "".join(dot_text)
        else:
            return self.to_formula().to_dot(*args, **kwargs)


class DDManager(object):
    """
    Manager for decision diagrams.
    """

    def __init__(self):
        self.nodes = (
            []
        )  # Stores inodes (only conjoin and disjoin, the remaining atoms are in formula.atom2var)
        self.constraint_dd = None

    def set_node(self, index, node):
        self.nodes[index] = node

    def get_node(self, index):
        return self.nodes[index]

    def add_node(self, node):
        self.nodes.append(node)

    def add_variable(self, label=0):
        """Add a variable to the manager and return its label.

        :param label: suggested label of the variable
        :type label: int
        :return: label of the new variable
        :rtype: int
        """
        raise NotImplementedError("abstract method")

    def literal(self, label):
        """Return an SDD node representing a literal.

        :param label: label of the literal
        :type label: int
        :return: internal node representing the literal
        """
        raise NotImplementedError("abstract method")

    def is_true(self, node):
        """Checks whether the SDD node represents True.

        :param node: node to verify
        :return: True if the node represents True
        :rtype: bool
        """
        raise NotImplementedError("abstract method")

    def true(self):
        """Return an internal node representing True.

        :return:
        """
        raise NotImplementedError("abstract method")

    def is_false(self, node):
        """Checks whether the internal node represents False

        :param node: node to verify
        :type node: SDDNode
        :return: False if the node represents False
        :rtype: bool
        """
        raise NotImplementedError("abstract method")

    def false(self):
        """Return an internal node representing False."""
        raise NotImplementedError("abstract method")

    def conjoin2(self, a, b):
        """Base method for conjoining two internal nodes.

        :param a: first internal node
        :param b: second internal node
        :return: conjunction of given nodes
        """
        raise NotImplementedError("abstract method")

    def disjoin2(self, a, b):
        """Base method for disjoining two internal nodes.

        :param a: first internal node
        :param b: second internal node
        :return: disjunction of given nodes
        """
        raise NotImplementedError("abstract method")

    def conjoin(self, *nodes):
        """Create the conjunction of the given nodes.

        :param nodes: nodes to conjoin
        :return: conjunction of the given nodes

        This method handles node reference counting, that is, all intermediate results are marked
        for garbage collection, and the output node has a reference count greater than one.
        Reference count on input nodes is not touched (unless one of the inputs becomes the output).
        """
        r = self.true()
        for s in nodes:
            r1 = self.conjoin2(r, s)
            self.ref(r1)
            self.deref(r)
            r = r1
        return r

    def disjoin(self, *nodes):
        """Create the disjunction of the given nodes.

        :param nodes: nodes to conjoin
        :return: disjunction of the given nodes

        This method handles node reference counting, that is, all intermediate results are marked
        for garbage collection, and the output node has a reference count greater than one.
        Reference count on input nodes is not touched (unless one of the inputs becomes the output).
        """
        r = self.false()
        for s in nodes:
            r1 = self.disjoin2(r, s)
            self.ref(r1)
            self.deref(r)
            r = r1
        return r

    def equiv(self, node1, node2):
        """Enforce the equivalence between node1 and node2.

        :param node1:
        :param node2:
        :return:
        """
        not1 = self.negate(node1)
        not2 = self.negate(node2)
        i1 = self.disjoin(not1, node2)
        self.deref(not1)
        i2 = self.disjoin(node1, not2)
        self.deref(not2)
        r = self.conjoin(i1, i2)
        self.deref(i1, i2)
        return r

    def negate(self, node):
        """Create the negation of the given node.

        This method handles node reference counting, that is, all intermediate results are marked \
        for garbage collection, and the output node has a reference count greater than one.
        Reference count on input nodes is not touched (unless one of the inputs becomes the output).

        :param node: negation of the given node
        :return: negation of the given node
        """
        raise NotImplementedError("abstract method")

    def same(self, node1, node2):
        """Checks whether two SDD nodes are equivalent.

        :param node1: first node
        :param node2: second node
        :return: True if the given nodes are equivalent, False otherwise.
        :rtype: bool
        """
        # Assumes SDD library always reuses equivalent nodes.
        raise NotImplementedError("abstract method")

    def ref(self, *nodes):
        """Increase the reference count for the given nodes.

        :param nodes: nodes to increase count on
        :type nodes: tuple of SDDNode
        """
        raise NotImplementedError("abstract method")

    def deref(self, *nodes):
        """Decrease the reference count for the given nodes.

        :param nodes: nodes to decrease count on
        :type nodes: tuple of SDDNode
        """
        raise NotImplementedError("abstract method")

    def write_to_dot(self, node, filename):
        """Write SDD node to a DOT file.

        :param node: SDD node to output
        :type node: SDDNode
        :param filename: filename to write to
        :type filename: basestring
        """
        raise NotImplementedError("abstract method")

    def wmc(self, node, weights, semiring):
        """Perform Weighted Model Count on the given node.

        :param node: node to evaluate
        :param weights: weights for the variables in the node
        :param semiring: use the operations defined by this semiring
        :return: weighted model count
        """
        raise NotImplementedError("abstract method")

    def wmc_literal(self, node, weights, semiring, literal):
        """Evaluate a literal in the decision diagram.

        :param node: root of the decision diagram
        :param weights: weights for the variables in the node
        :param semiring: use the operations defined by this semiring
        :param literal: literal to evaluate
        :return: weighted model count
        """
        raise NotImplementedError("abstract method")

    def wmc_true(self, weights, semiring):
        """Perform weighted model count on a true node.
        This can be used to obtain a normalization constant.

        :param weights: weights for the variables in the node
        :param semiring: use the operations defined by this semiring
        :return: weighted model count
        """
        raise NotImplementedError("abstract method")

    def __del__(self):
        """Clean up the internal structure."""
        raise NotImplementedError("abstract method")


class DDEvaluator(Evaluator):
    """Generic evaluator for bottom-up compiled decision diagrams.

    :param formula:
    :type: DD
    :param semiring:
    :param weights:
    :return:
    """

    def __init__(self, formula, semiring, weights=None, **kwargs):
        Evaluator.__init__(self, formula, semiring, weights, **kwargs)
        self.formula = formula
        self.normalization = None
        self._evidence_weight = None
        self.evidence_inode = None

    def _get_manager(self):
        return self.formula.get_manager()

    def _get_z(self):
        return self.normalization

    def _initialize(self, with_evidence=True):
        self.weights.clear()

        weights = self.formula.extract_weights(self.semiring, self.given_weights)
        for atom, weight in weights.items():
            av = self.formula.atom2var.get(atom)
            if av is not None:
                self.weights[av] = weight
            elif atom == 0:
                self.weights[0] = weight

        if with_evidence:
            for ev in self.evidence():
                if abs(ev) in self.formula.atom2var:
                    # Only for atoms
                    self.set_evidence(self.formula.atom2var[abs(ev)], ev > 0)

    def propagate(self):
        self._initialize()
        if isinstance(self.semiring, SemiringLogProbability) or isinstance(
            self.semiring, SemiringProbability
        ):
            self.normalization = self._get_manager().wmc_true(
                self.weights, self.semiring
            )
        else:
            self.normalization = None
        self.evaluate_evidence(recompute=True)

    def _evaluate_evidence(self, recompute=False):
        if self._evidence_weight is None or recompute:
            constraint_inode = self.formula.get_constraint_inode()
            evidence_nodes = [self.formula.get_inode(ev) for ev in self.evidence()]
            self.evidence_inode = self._get_manager().conjoin(
                constraint_inode, *evidence_nodes
            )

            if isinstance(self.semiring, SemiringLogProbability) or isinstance(
                self.semiring, SemiringProbability
            ):
                result = self._get_manager().wmc(
                    self.evidence_inode, self.weights, self.semiring
                )
                if self.semiring.is_zero(result):
                    raise InconsistentEvidenceError(context=" during compilation")
                self._evidence_weight = self.semiring.normalize(
                    result, self.normalization
                )
            else:
                formula = LogicNNF()
                i = self.formula._to_formula(formula, self.evidence_inode, {})
                if i is None:  # = Node(False)
                    # evaluate(index=i=None,..) evaluates the LABEL.QUERY's of formula.
                    formula.add_name(
                        name="tempFalse", key=None, label=formula.LABEL_QUERY
                    )
                    result = formula.evaluate(index=i, semiring=self.semiring)
                    result = result["tempFalse"]
                else:
                    result = formula.evaluate(index=i, semiring=self.semiring)

                if self.semiring.is_zero(result):
                    raise InconsistentEvidenceError(context=" during compilation")
                self._evidence_weight = result

        return self._evidence_weight

    def evaluate_evidence(self, recompute=False):
        return self.semiring.result(
            self._evaluate_evidence(recompute=recompute), self.formula
        )

    def evaluate(self, node):
        if isinstance(self.semiring, SemiringLogProbability) or isinstance(
            self.semiring, SemiringProbability
        ):
            return self.evaluate_standard(node)
        else:
            return self.evaluate_custom(node)

    def evaluate_standard(self, node):
        # Trivial case: node is deterministically True or False
        if node == self.formula.TRUE:
            result = self.semiring.one()
        elif node == self.formula.FALSE:
            result = self.semiring.zero()
        else:
            query_def_inode = self.formula.get_inode(node)
            evidence_inode = self.evidence_inode
            # Construct the query SDD
            # if not evidence propagated or (query and evidence share variables):
            query_sdd = self._get_manager().conjoin(query_def_inode, evidence_inode)
            # else:
            #    query_sdd = query_def_inode

            result = self._get_manager().wmc(query_sdd, self.weights, self.semiring)
            self._get_manager().deref(query_sdd)
            # TODO only normalize when there are evidence or constraints.
            result = self.semiring.normalize(result, self.normalization)
            result = self.semiring.normalize(result, self._evidence_weight)
        return self.semiring.result(result, self.formula)

    def evaluate_custom(self, node):
        # Trivial case: node is deterministically True or False
        if node == self.formula.TRUE:
            result = self.semiring.one()
        elif node is self.formula.FALSE:
            result = self.semiring.zero()
        else:
            query_def_inode = self.formula.get_inode(node)
            evidence_inode = self.evidence_inode
            query_sdd = self._get_manager().conjoin(query_def_inode, evidence_inode)

            formula = LogicNNF()
            i = self.formula._to_formula(formula, query_sdd, {})

            if i is None:  # = SddNode(False)
                # evaluate(index=i=None,..) evaluates the LABEL.QUERY's of formula.
                name = self.formula.get_name(node)
                formula.add_name(name=name, key=None, label=formula.LABEL_QUERY)
                result = formula.evaluate(index=i, semiring=self.semiring)
                # index=None such that this if i is None is unnecessary. (same structure at _evaluate_evidence(..))
                result = result[name]
            else:
                result = formula.evaluate(index=i, semiring=self.semiring)

            self._get_manager().deref(query_sdd)

            # TODO only normalize when there are evidence or constraints.
            #            result = self.semiring.normalize(result, self.normalization)
            result = self.semiring.normalize(result, self._evidence_weight)
        return self.semiring.result(result, self.formula)

    def evaluate_fact(self, node):
        if node == self.formula.TRUE:
            return self.semiring.one()
        elif node is self.formula.FALSE:
            return self.semiring.zero()

        inode = self.evidence_inode

        literal_result = self._get_manager().wmc_literal(
            inode, self.weights, self.semiring, self.formula.atom2var[node]
        )
        result = self.semiring.result(literal_result, self.formula)
        return result

    def set_evidence(self, index, value):
        curr_pos_weight, curr_neg_weight = self.weights.get(index)
        pos, neg = self.semiring.to_evidence(
            curr_pos_weight, curr_neg_weight, sign=value
        )

        if (value and self.semiring.is_zero(curr_pos_weight)) or (
            not value and self.semiring.is_zero(curr_neg_weight)
        ):
            raise InconsistentEvidenceError(self._deref_node(index))

        self.set_weight(index, pos, neg)

    def set_weight(self, index, pos, neg):
        # index = index of atom in weights, so atom2var[key] = index
        self.weights[index] = (pos, neg)

    def _deref_node(self, index):
        term = self.formula.get_node(self.formula.var2atom[index]).name
        return term

    def __del__(self):
        if self.evidence_inode is not None:
            self._get_manager().deref(self.evidence_inode)


# noinspection PyUnusedLocal
def build_dd(source, destination, **kwdargs):
    """Build a DD from another formula.

    :param source: source formula
    :param destination: destination formula
    :param kwdargs: extra arguments
    :return: destination
    """
    with Timer("Compiling %s" % destination.__class__.__name__):

        # TODO maintain a translation table
        for i, n, t in source:
            if t == "atom":
                j = destination.add_atom(
                    n.identifier,
                    n.probability,
                    n.group,
                    name=source.get_name(i),
                    cr_extra=False,
                )
            elif t == "conj":
                j = destination.add_and(n.children, name=n.name)
            elif t == "disj":
                j = destination.add_or(n.children, name=n.name)
            else:
                raise TypeError("Unknown node type")
            # assert i == j  # Does not hold with constraints. See if-comment.

        for name, node, label in source.get_names_with_label():
            destination.add_name(name, node, label)

        destination.build_dd()

    return destination
