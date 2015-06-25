"""
Common interface to decision diagrams (BDD, SDD).
"""

from __future__ import print_function


from .util import Timer
from .formula import LogicFormula
from .evaluator import Evaluatable, Evaluator
from .evaluator import InconsistentEvidenceError


class DD(LogicFormula, Evaluatable):

    def __init__(self, **kwdargs):
        LogicFormula.__init__(self, **kwdargs)

        self.inode_manager = None
        self.inodes = []

        self.atom2var = {}

        self._constraint_dd = None

    def create_manager(self):
        raise NotImplementedError('Abstract method.')

    def get_manager(self):
        if self.inode_manager is None:
            self.inode_manager = self.create_manager()
        return self.inode_manager

    def _create_atom(self, identifier, probability, group, name=None):
        index = len(self)+1
        self.atom2var[index] = self.get_manager().add_variable()
        return self._atom(identifier, probability, group, name)

    def get_inode(self, index):
        """
        Get the internal node corresponding to the entry at the given index.
        :param index: index of node to retrieve
        :return: SDD node corresponding to the given index
        :rtype: SDDNode
        """
        negate = False
        if index < 0:
            index = -index
            negate = True
        node = self.getNode(index)
        if type(node).__name__ == 'atom':
            result = self.get_manager().literal(self.atom2var[index])
        else:
            # Extend list
            while len(self.inodes) < index:
                self.inodes.append(None)
            if self.inodes[index-1] is None:
                self.inodes[index-1] = self.create_inode(node)
            result = self.inodes[index-1]
        if negate:
            new_sdd = self.get_manager().negate(result)
            return new_sdd
        else:
            return result

    def create_inode(self, node):
        # Create SDD
        if type(node).__name__ == 'conj':
            return self.get_manager().conjoin(*[self.get_inode(c) for c in node.children])
        else:
            return self.get_manager().disjoin(*[self.get_inode(c) for c in node.children])

    def add_inode(self, node):
        self.inodes.append(node)
        return len(self.inodes)-1

    def set_inode(self, index, node):
        assert index is not None
        assert index > 0
        self.inodes[index-1] = node

    def get_constraint_inode(self):
        if self._constraint_dd is None:
            return self.get_manager().true()
        else:
            return self._constraint_dd

    def create_evaluator(self, semiring, weights):
        return DDEvaluator(self, semiring, weights)

    def build_dd(self):
        required_nodes = set([abs(n) for q, n in self.queries() if self.isProbabilistic(n)])
        required_nodes |= set([abs(n) for q, n in self.queries() if self.isProbabilistic(n)])

        for n in required_nodes:
            self.get_inode(n)

        self.build_constraint_dd()

    def build_constraint_dd(self):
        self._constraint_dd = self.get_manager().true()
        for c in self.constraints():
            for rule in c.encodeCNF():
                rule_sdd = self.get_manager().disjoin(*[self.get_inode(r) for r in rule])
                new_constraint_dd = self.get_manager().conjoin(self._constraint_dd, rule_sdd)
                self.get_manager().deref(self._constraint_dd)
                self.get_manager().deref(rule_sdd)
                self._constraint_dd = new_constraint_dd




class DDManager(object):
    """
    Manager for decision diagrams.
    """

    def __init__(self):
        pass

    def add_variable(self, label=0):
        """
        Add a variable to the manager and return its label.
        :param label: suggested label of the variable
        :type label: int
        :return: label of the new variable
        :rtype: int
        """
        raise NotImplementedError('abstract method')

    def literal(self, label):
        """
        Return an SDD node representing a literal.
        :param label: label of the literal
        :type label: int
        :return: SDD node representing the literal
        :rtype: SDDNode
        """
        raise NotImplementedError('abstract method')

    def is_true(self, node):
        """
        Checks whether the SDD node represents True
        :param node: node to verify
        :type node: SDDNode
        :return: True if the node represents True
        :rtype: bool
        """
        raise NotImplementedError('abstract method')

    def true(self):
        """
        Return an SDD node representing True
        :return:
        """
        raise NotImplementedError('abstract method')

    def is_false(self, node):
        """
        Checks whether the SDD node represents False
        :param node: node to verify
        :type node: SDDNode
        :return: False if the node represents False
        :rtype: bool
        """
        raise NotImplementedError('abstract method')

    def false(self):
        """
        Return an SDD node representing False
        :return:
        """
        raise NotImplementedError('abstract method')

    def conjoin2(self, a, b):
        raise NotImplementedError('abstract method')

    def disjoin2(self, a, b):
        raise NotImplementedError('abstract method')

    def conjoin(self, *nodes):
        """
        Create the conjunction of the given nodes.
        :param nodes: nodes to conjoin
        :type: SDDNode
        :return: conjunction of the given nodes
        :rtype: SDDNode

        This method handles node reference counting, that is, all intermediate results
         are marked for garbage collection, and the output node has a reference count greater than one.
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
        """
        Create the disjunction of the given nodes.
        :param nodes: nodes to conjoin
        :type: SDDNode
        :return: disjunction of the given nodes
        :rtype: SDDNode

        This method handles node reference counting, that is, all intermediate results
         are marked for garbage collection, and the output node has a reference count greater than one.
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
        """
        Enforce the equivalence between node1 and node2 in the SDD.
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
        """
        Create the negation of the given node.
        :param node: negation of the given node
        :type node: SDDNode
        :return: negation of the given node
        :rtype: SDDNode

        This method handles node reference counting, that is, all intermediate results
         are marked for garbage collection, and the output node has a reference count greater than one.
        Reference count on input nodes is not touched (unless one of the inputs becomes the output).

        """
        raise NotImplementedError('abstract method')

    def same(self, node1, node2):
        """
        Checks whether two SDD nodes are equivalent.
        :param node1: first node
        :type: SDDNode
        :param node2: second node
        :type: SDDNode
        :return: True if the given nodes are equivalent, False otherwise.
        :rtype: bool
        """
        # Assumes SDD library always reuses equivalent nodes.
        raise NotImplementedError('abstract method')

    def ref(self, *nodes):
        """
        Increase the reference count for the given nodes.
        :param nodes: nodes to increase count on
        :type nodes: tuple of SDDNode
        """
        raise NotImplementedError('abstract method')

    def deref(self, *nodes):
        """
        Decrease the reference count for the given nodes.
        :param nodes: nodes to decrease count on
        :type nodes: tuple of SDDNode
        """
        raise NotImplementedError('abstract method')

    def write_to_dot(self, node, filename):
        """
        Write SDD node to a DOT file.
        :param node: SDD node to output
        :type node: SDDNode
        :param filename: filename to write to
        :type filename: basestring
        """
        raise NotImplementedError('abstract method')

    def wmc(self, node, weights, semiring):
        """
        Perform Weighted Model Count on the given node.
        :param node: node to evaluate
        :param weights: weights for the variables in the node
        :param semiring: use the operations defined by this semiring
        :return: weighted model count
        """

    def __del__(self):
        """
        Clean up the SDD manager.
        """
        raise NotImplementedError('abstract method')


class DDEvaluator(Evaluator):

    def __init__(self, formula, semiring, weights=None):
        """
        :param formula:
        :type: DD
        :param semiring:
        :param weights:
        :return:
        """
        Evaluator.__init__(self, formula, semiring)
        self.formula = formula
        self.weights = {}
        self.original_weights = weights
        self.normalization = None

        self.evidence_inode = None

    def get_manager(self):
        return self.formula.get_manager()

    def get_names(self, label=None):
        return self.formula.get_names(label)

    def get_z(self):
        return self.normalization

    def initialize(self, with_evidence=True):
        self.weights.clear()

        weights = self.formula.extractWeights(self.semiring, self.original_weights)
        for atom, weight in weights.items():
            av = self.formula.atom2var.get(atom)
            if av is not None:
                self.weights[av] = weight

        if with_evidence:
            for ev in self.evidence():
                if ev in self.formula.atom2var:
                    # Only for atoms
                    self.set_evidence(self.formula.atom2var[ev], ev > 0)

    def propagate(self):
        self.initialize()
        self.normalization = self.evaluate_evidence()

    def evaluate_evidence(self):
        constraint_inode = self.formula.get_constraint_inode()
        evidence_nodes = [self.formula.get_inode(ev) for ev in self.evidence()]
        self.evidence_inode = self.get_manager().conjoin(constraint_inode, *evidence_nodes)
        result = self.get_manager().wmc(self.evidence_inode, self.weights, self.semiring)
        if result == self.semiring.zero():
            raise InconsistentEvidenceError()
        return result

    def evaluate(self, node):
        # Trivial case: node is deterministically True or False
        if node == self.formula.TRUE:
            return self.semiring.one()
        elif node is self.formula.FALSE:
            return self.semiring.zero()

        query_def_inode = self.formula.get_inode(node)
        evidence_inode = self.evidence_inode
        # Construct the query SDD
        query_sdd = self.get_manager().conjoin(query_def_inode, evidence_inode)
        result = self.get_manager().wmc(query_sdd, self.weights, self.semiring)
        self.get_manager().deref(query_sdd)
        # TODO only normalize when there are evidence or constraints.
        result = self.semiring.normalize(result, self.normalization)
        return result

    def set_evidence(self, index, value):
        pos = self.semiring.one()
        neg = self.semiring.zero()
        if value:
            self.set_weight(index, pos, neg)
        else:
            self.set_weight(index, neg, pos)

    def set_weight(self, index, pos, neg):
        self.weights[index] = (pos, neg)

    def __del__(self):
        self.get_manager().deref(self.evidence_inode)


def build_dd(source, destination, ddname, **kwdargs):

    with Timer('Compiling %s' % ddname):

        # TODO maintain a translation table
        for i, n, t in source:
            if t == 'atom':
                j = destination.addAtom(n.identifier, n.probability, n.group, source.get_name(i))
            elif t == 'conj':
                j = destination.addAnd(n.children, source.get_name(i))
            elif t == 'disj':
                j = destination.addOr(n.children, source.get_name(i))
            else:
                raise TypeError('Unknown node type')
            assert i == j

        for name, node, label in source.getNamesWithLabel():
            destination.addName(name, node, label)

        for c in source.constraints():
            if c.isActive():
                destination.add_constraint(c)
        destination.build_dd()

    return destination
