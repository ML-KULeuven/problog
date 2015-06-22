"""
__author__ = Anton Dries

Provides access to Binary Decision Diagrams (BDDs).

"""

from __future__ import print_function

from collections import namedtuple
from .formula import LogicDAG
from .evaluator import Evaluator, SemiringProbability, Evaluatable, InconsistentEvidenceError
from .core import transform, InstallError
from .util import Timer

try:
    import pyeda.boolalg.bdd as bdd
    import pyeda.boolalg.expr as bdd_expr

except Exception:
    bdd = None


class BDDManager(object):
    """
    Manager for BDDs.
    It wraps around the pyeda BDD module
    """

    def __init__(self, varcount=0, auto_gc=True):
        """
        Create a new BDD manager.
        :param varcount: number of initial variables
        :type varcount: int
        :param auto_gc: use automatic garbage collection and minimization
        :type auto_gc: bool
        """
        self.varcount = 1
        self.ZERO = bdd.expr2bdd(bdd_expr.expr('0'))
        self.ONE = bdd.expr2bdd(bdd_expr.expr('1'))

    def add_variable(self, label=0):
        """
        Add a variable to the manager and return its label.
        :param label: suggested label of the variable
        :type label: int
        :return: label of the new variable
        :rtype: int
        """
        if label == 0 or label > self.varcount:
            self.varcount += 1
            label = self.varcount
            return bdd.bddvar('v' + str(label))
        else:
            return bdd.bddvar('v' + str(label))

    def literal(self, label):
        """
        Return an BDD node representing a literal.
        :param label: label of the literal
        :type label: int
        :return: BDD node representing the literal
        :rtype: BDDNode
        """
        return self.add_variable(label)

    def is_true(self, node):
        """
        Checks whether the BDD node represents True
        :param node: node to verify
        :type node: BDDNode
        :return: True if the node represents True
        :rtype: bool
        """
        return node.is_one()

    def true(self):
        """
        Return an BDD node representing True
        :return:
        """
        return self.ONE

    def is_false(self, node):
        """
        Checks whether the BDD node represents False
        :param node: node to verify
        :type node: BDDNode
        :return: False if the node represents False
        :rtype: bool
        """
        return node.is_zero()

    def false(self):
        """
        Return an BDD node representing False
        :return:
        """
        return self.ZERO

    def conjoin(self, *nodes):
        """
        Create the conjunction of the given nodes.
        :param nodes: nodes to conjoin
        :type: BDDNode
        :return: conjunction of the given nodes
        :rtype: BDDNode

        This method handles node reference counting, that is, all intermediate results
         are marked for garbage collection, and the output node has a reference count greater than one.
        Reference count on input nodes is not touched (unless one of the inputs becomes the output).
        """
        r = self.ONE
        for s in nodes:
            r = r & s
        return r

    def disjoin(self, *nodes):
        """
        Create the disjunction of the given nodes.
        :param nodes: nodes to conjoin
        :type: BDDNode
        :return: disjunction of the given nodes
        :rtype: BDDNode

        This method handles node reference counting, that is, all intermediate results
         are marked for garbage collection, and the output node has a reference count greater than one.
        Reference count on input nodes is not touched (unless one of the inputs becomes the output).
        """
        r = self.ZERO
        for s in nodes:
            r = r | s
        return r

    def negate(self, node):
        """
        Create the negation of the given node.
        :param node: negation of the given node
        :type node: BDDNode
        :return: negation of the given node
        :rtype: BDDNode

        This method handles node reference counting, that is, all intermediate results
         are marked for garbage collection, and the output node has a reference count greater than one.
        Reference count on input nodes is not touched (unless one of the inputs becomes the output).

        """
        return ~node

    def same(self, node1, node2):
        """
        Checks whether two BDD nodes are equivalent.
        :param node1: first node
        :type: BDDNode
        :param node2: second node
        :type: BDDNode
        :return: True if the given nodes are equivalent, False otherwise.
        :rtype: bool
        """
        # Assumes BDD library always reuses equivalent nodes.
        return node1 is node2

    # def equiv(self, node1, node2):
    #     """
    #     Enforce the equivalence between node1 and node2 in the BDD.
    #     :param node1:
    #     :param node2:
    #     :return:
    #     """
    #     not1 = self.negate(node1)
    #     not2 = self.negate(node2)
    #     i1 = self.disjoin(not1, node2)
    #     self.deref(not1)
    #     i2 = self.disjoin(node1, not2)
    #     self.deref(not2)
    #     r = self.conjoin(i1, i2)
    #     self.deref(i1, i2)
    #     return r

    def ref(self, *nodes):
        """
        Increase the reference count for the given nodes.
        :param nodes: nodes to increase count on
        :type nodes: tuple of BDDNode
        """
        pass

    def deref(self, *nodes):
        """
        Decrease the reference count for the given nodes.
        :param nodes: nodes to decrease count on
        :type nodes: tuple of BDDNode
        """
        pass

    def write_to_dot(self, node, filename):
        """
        Write BDD node to a DOT file.
        :param node: BDD node to output
        :type node: BDDNode
        :param filename: filename to write to
        :type filename: basestring
        """
        print (node.to_dot())

    def __del__(self):
        """
        Clean up the BDD manager.
        """
        pass


class BDD(LogicDAG, Evaluatable):
    """A propositional logic formula consisting of and, or, not and atoms represented as an BDD.

    This class has two restrictions with respect to the default LogicFormula:

        * The number of atoms in the BDD should be known at construction time.
        * It does not support updatable nodes.

    This means that this class can not be used directly during grounding.
    It can be used as a target for the ``makeAcyclic`` method.
    """

    _atom = namedtuple('atom', ('identifier', 'probability', 'group', 'bddlit', 'name'))
    _conj = namedtuple('conj', ('children', 'bddnode', 'name'))
    _disj = namedtuple('disj', ('children', 'bddnode', 'name'))
    # negation is encoded by using a negative number for the key

    def __init__(self, **kwdargs):
        LogicDAG.__init__(self, auto_compact=False)

        if bdd is None:
            raise InstallError('The BDD library is not available.')

        self.bdd_manager = BDDManager()
        self._constraint_bdd = None

    def _create_atom(self, identifier, probability, group, name=None):
        new_lit = self.getAtomCount()+1
        self.bdd_manager.add_variable(len(self)+1)
        return self._atom(identifier, probability, group, len(self)+1, name)

    def _create_conj(self, children, name=None):
        new_bdd = self.bdd_manager.conjoin(*[self.get_bddnode(c) for c in children])
        self.bdd_manager.add_variable(len(self)+1)
        return self._conj(children, new_bdd, name)

    def _create_disj(self, children, name=None):
        new_bdd = self.bdd_manager.disjoin(*[self.get_bddnode(c) for c in children])
        self.bdd_manager.add_variable(len(self)+1)
        return self._disj(children, new_bdd, name)

    def addName(self, name, node_id, label=None):
        LogicDAG.addName(self, name, node_id, label)
        if label == self.LABEL_QUERY:
            pass
        elif label in (self.LABEL_EVIDENCE_MAYBE, self.LABEL_EVIDENCE_NEG, self.LABEL_EVIDENCE_POS):
            pass

    def get_bddnode(self, index):
        """
        Get the BDD node corresponding to the entry at the given index.
        :param index: index of node to retrieve
        :return: BDD node corresponding to the given index
        :rtype: BDDNode
        """
        negate = False
        if index < 0:
            index = -index
            negate = True 
        node = self.getNode(index)
        if type(node).__name__ == 'atom':
            result = self.bdd_manager.literal(node.bddlit)
        else:
            result = node.bddnode
        if negate:
            new_bdd = self.bdd_manager.negate(result)
            return new_bdd
        else:
            return result

    def get_constraint_bdd(self):
        if self._constraint_bdd is None:
            return self.bdd_manager.true()
        else:
            return self._constraint_bdd

    def addConstraint(self, c):
        if self._constraint_bdd is None:
            self._constraint_bdd = self.bdd_manager.true()
        LogicDAG.addConstraint(self, c)
        for rule in c.encodeCNF():
            rule_bdd = self.bdd_manager.disjoin(*[self.get_bddnode(r) for r in rule])
            new_constraint_bdd = self.bdd_manager.conjoin(self._constraint_bdd, rule_bdd)
            self.bdd_manager.deref(self._constraint_bdd)
            self.bdd_manager.deref(rule_bdd)
            self._constraint_bdd = new_constraint_bdd

    def write_to_dot(self, filename, index=None):
        """
        Write an BDD node to a DOT file.
        :param filename: filename to write to
        :param index: index of the node in the BDD data structure
        """
        pass

    def addDisjunct(self, key, component):
        """Add a component to the node with the given key."""
        raise NotImplementedError('BDD formula does not support node updates.')

    def _createEvaluator(self, semiring, weights) :
        if not isinstance(semiring,SemiringProbability) :
            raise ValueError('BDD evaluation currently only supports probabilities!')
        return BDDEvaluator(self, semiring, weights)
        
    @classmethod
    def is_available(cls) :
        return bdd is not None
    
            
@transform(LogicDAG, BDD)
def buildBDD(source, destination, **kwdargs):

    with Timer('Compiling BDD'):
        for i, n, t in source:
            if t == 'atom':
                destination.addAtom(n.identifier, n.probability, n.group)
            elif t == 'conj':
                destination.addAnd(n.children)
            elif t == 'disj':
                destination.addOr(n.children)
            else:
                raise TypeError('Unknown node type')
                
        for name, node, label in source.getNamesWithLabel():
            destination.addName(name, node, label)
        
        for c in source.constraints():
            if c.isActive():
                destination.addConstraint(c)

    return destination
        

class BDDEvaluator(Evaluator):

    def __init__(self, formula, semiring, weights=None):
        Evaluator.__init__(self, formula, semiring)
        self.__bdd = formula
        self.bdd_manager = formula.bdd_manager
        self.__probs = {}
        self.__given_weights = weights
        self.__z = None

    def getNames(self, label=None) :
        return self.__bdd.getNames(label)
    
    def initialize(self, with_evidence=True) :
        self.__probs.clear()
    
        self.__probs.update(self.__bdd.extractWeights(self.semiring, self.__given_weights))
                            
        if with_evidence :
            for ev in self.iterEvidence() :
                self.setEvidence( abs(ev), ev > 0 )
        
            self.__z = self.evaluateEvidence()
            
    def propagate(self) :
        self.initialize()

    def evaluateEvidence(self):
        self.initialize(False)

        evidence_bdd = self.bdd_manager.conjoin(*[self.__bdd.get_bddnode(ev) for ev in self.iterEvidence()])

        query_evidence_bdd = self.bdd_manager.conjoin(evidence_bdd, self.__bdd.get_constraint_bdd())
        query_bdd = query_evidence_bdd

        if self.bdd_manager.is_false(query_bdd):
            raise InconsistentEvidenceError()

        pall = self.semiring.zero()
        for path in query_bdd.satisfy_all():
            pw = self.semiring.one()
            for var, val in path.items():
                var = int(var.name[1:])
                pos, neg = self.__probs[var]
                if val:
                    p = pos
                else:
                    p = neg
                pw = self.semiring.times(p, pw)
            pall = self.semiring.plus(pw, pall)
        return pall

    def evaluate(self, node):
        """
        Evaluate the given node in the BDD.
        :param node: identifier of the node to evaluate
        :return: weight of the node
        """

        # Trivial case: node is deterministically True or False
        if node == 0:
            return self.semiring.one()
        elif node is None:
            return self.semiring.zero()

        query_def_bdd = self.__bdd.get_bddnode(node)
        constraint_bdd = self.__bdd.get_constraint_bdd()

        # Construct the query BDD
        evidence_bdd = self.bdd_manager.conjoin(
            *[self.__bdd.get_bddnode(ev) for ev in self.iterEvidence()])

        query_bdd = self.bdd_manager.conjoin(query_def_bdd, evidence_bdd, constraint_bdd)

        if self.bdd_manager.is_true(query_bdd):
            if node < 0:
                return self.__probs[-node][1]
            else:
                return self.__probs[node][0]
        elif self.bdd_manager.is_false(query_bdd):
            return self.semiring.zero()
        else:
            pall = self.semiring.zero()
            for path in query_bdd.satisfy_all():
                pw = self.semiring.one()
                for var, val in path.items():
                    var = int(var.name[1:])
                    pos, neg = self.__probs[var]
                    if val:
                        p = pos
                    else:
                        p = neg
                    pw = self.semiring.times(p, pw)
                pall = self.semiring.plus(pw, pall)
            return self.semiring.normalize(pall, self.__z)

    def setEvidence(self, index, value):
        pos = self.semiring.one()
        neg = self.semiring.zero()
        if value :
            self.setWeight( index, pos, neg )
        else :
            self.setWeight( index, neg, pos )

    def setWeight(self, index, pos, neg):
        self.__probs[index] = (pos, neg)
