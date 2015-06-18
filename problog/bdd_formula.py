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
        Create a new SDD manager.
        :param varcount: number of initial variables
        :type varcount: int
        :param auto_gc: use automatic garbage collection and minimization
        :type auto_gc: bool
        """
        if varcount is None or varcount == 0:
            varcount = 1
        self.varcount = varcount
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
        Return an SDD node representing a literal.
        :param label: label of the literal
        :type label: int
        :return: SDD node representing the literal
        :rtype: SDDNode
        """
        return self.add_variable(label)

    def is_true(self, node):
        """
        Checks whether the SDD node represents True
        :param node: node to verify
        :type node: SDDNode
        :return: True if the node represents True
        :rtype: bool
        """
        return node.is_one()

    def true(self):
        """
        Return an SDD node representing True
        :return:
        """
        return self.ONE

    def is_false(self, node):
        """
        Checks whether the SDD node represents False
        :param node: node to verify
        :type node: SDDNode
        :return: False if the node represents False
        :rtype: bool
        """
        return node.is_zero()

    def false(self):
        """
        Return an SDD node representing False
        :return:
        """
        return self.ZERO

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
        r = self.ONE
        for s in nodes:
            r = r & s
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
        r = self.ZERO
        for s in nodes:
            r = r | s
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
        return ~node

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
        return node1 is node2

    # def equiv(self, node1, node2):
    #     """
    #     Enforce the equivalence between node1 and node2 in the SDD.
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
        :type nodes: tuple of SDDNode
        """
        pass

    def deref(self, *nodes):
        """
        Decrease the reference count for the given nodes.
        :param nodes: nodes to decrease count on
        :type nodes: tuple of SDDNode
        """
        pass

    def write_to_dot(self, node, filename):
        """
        Write SDD node to a DOT file.
        :param node: SDD node to output
        :type node: SDDNode
        :param filename: filename to write to
        :type filename: basestring
        """
        print (node.to_dot())

    def __del__(self):
        """
        Clean up the SDD manager.
        """
        pass


# Changes to be made:
#  Currently, variables in the SDD correspond to internal nodes in the LogicFormula.
#  This means that many of them are not used. Literals should refer to facts instead.
#  The only exceptions should be the query nodes.
#  This should also fix some problems with minimization.
#
#  get_atom_literal
#  get_query_literal
#  get_evidence_literal (can be same as get_query_literal)


class BDD(LogicDAG, Evaluatable):
    """A propositional logic formula consisting of and, or, not and atoms represented as an SDD.

    This class has two restrictions with respect to the default LogicFormula:

        * The number of atoms in the SDD should be known at construction time.
        * It does not support updatable nodes.

    This means that this class can not be used directly during grounding.
    It can be used as a target for the ``makeAcyclic`` method.
    """

    _atom = namedtuple('atom', ('identifier', 'probability', 'group', 'sddlit') )
    _conj = namedtuple('conj', ('children', 'sddnode') )
    _disj = namedtuple('disj', ('children', 'sddnode') )
    # negation is encoded by using a negative number for the key

    def __init__(self, sdd_auto_gc=False, **kwdargs):
        LogicDAG.__init__(self, auto_compact=False)

        if bdd is None:
            raise InstallError('The BDD library is not available.')

        self.sdd_manager = BDDManager()
        self._constraint_sdd = None

    def set_varcount(self, varcount):
        """
        Set the variable count for the SDD.
        This method should be called before any nodes are added to the SDD.
        :param varcount: number of variables in the SDD
        """
        self.sdd_manager = SDDManager(varcount=varcount+1, auto_gc=self.auto_gc)

    def _create_atom(self, identifier, probability, group):
        new_lit = self.getAtomCount()+1
        self.sdd_manager.add_variable(len(self)+1)
        return self._atom(identifier, probability, group, len(self)+1)

    def _create_conj(self, children):
        new_sdd = self.sdd_manager.conjoin(*[self.get_sddnode(c) for c in children])
        self.sdd_manager.add_variable(len(self)+1)
        return self._conj(children, new_sdd)

    def _create_disj(self, children):
        new_sdd = self.sdd_manager.disjoin(*[self.get_sddnode(c) for c in children])
        self.sdd_manager.add_variable(len(self)+1)
        return self._disj(children, new_sdd)

    def addName(self, name, node_id, label=None):
        LogicDAG.addName(self, name, node_id, label)
        if label == self.LABEL_QUERY:
            pass
        elif label in (self.LABEL_EVIDENCE_MAYBE, self.LABEL_EVIDENCE_NEG, self.LABEL_EVIDENCE_POS):
            pass

    def get_sddnode(self, index):
        """
        Get the SDD node corresponding to the entry at the given index.
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
            # was node.sddlit
            result = self.sdd_manager.literal(node.sddlit)
        else:
            result = node.sddnode
        if negate:
            new_sdd = self.sdd_manager.negate(result)
            return new_sdd
        else:
            return result

    def get_constraint_sdd(self):
        if self._constraint_sdd is None:
            return self.sdd_manager.true()
        else:
            return self._constraint_sdd

    def addConstraint(self, c):
        if self._constraint_sdd is None:
            self._constraint_sdd = self.sdd_manager.true()
        LogicDAG.addConstraint(self, c)
        for rule in c.encodeCNF():
            rule_sdd = self.sdd_manager.disjoin(*[self.get_sddnode(r) for r in rule])
            new_constraint_sdd = self.sdd_manager.conjoin(self._constraint_sdd, rule_sdd)
            self.sdd_manager.deref(self._constraint_sdd)
            self.sdd_manager.deref(rule_sdd)
            self._constraint_sdd = new_constraint_sdd

    def write_to_dot(self, filename, index=None):
        """
        Write an SDD node to a DOT file.
        :param filename: filename to write to
        :param index: index of the node in the SDD data structure
        """
        pass

    def _update(self, key, value):
        """Replace the node with the given node."""
        raise NotImplementedError('SDD formula does not support node updates.')
        
    def addDisjunct(self, key, component):
        """Add a component to the node with the given key."""
        raise NotImplementedError('SDD formula does not support node updates.')

    def _createEvaluator(self, semiring, weights) :
        if not isinstance(semiring,SemiringProbability) :
            raise ValueError('SDD evaluation currently only supports probabilities!')
        return BDDEvaluator(self, semiring, weights)
        
    @classmethod
    def is_available(cls) :
        return bdd is not None
    
            
@transform(LogicDAG, BDD)
def buildBDD(source, destination, **kwdargs):

    with Timer('Compiling BDD'):
        if kwdargs.get('sdd_preset_variables'):
            s = len(source)
            destination.set_varcount(s+1)
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
        self.__sdd = formula
        self.sdd_manager = formula.sdd_manager
        self.__probs = {}
        self.__given_weights = weights
        self.__z = None

    def getNames(self, label=None) :
        return self.__sdd.getNames(label)
    
    def initialize(self, with_evidence=True) :
        self.__probs.clear()
    
        self.__probs.update(self.__sdd.extractWeights(self.semiring, self.__given_weights))
                            
        if with_evidence :
            for ev in self.iterEvidence() :
                self.setEvidence( abs(ev), ev > 0 )
        
            # evidence sdd => conjoin evidence nodes
            self.__z = self.evaluateEvidence()
            
    def propagate(self) :
        self.initialize()

    def evaluateEvidence(self):
        self.initialize(False)


        evidence_sdd = self.sdd_manager.conjoin(*[self.__sdd.get_sddnode(ev) for ev in self.iterEvidence()])

        query_evidence_sdd = self.sdd_manager.conjoin(evidence_sdd, self.__sdd.get_constraint_sdd())
        query_sdd = query_evidence_sdd

        if self.sdd_manager.is_false(query_sdd):
            raise InconsistentEvidenceError()


        pall = self.semiring.zero()
        for path in query_sdd.satisfy_all():
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
        Evaluate the given node in the SDD.
        :param node: identifier of the node to evaluate
        :return: weight of the node
        """

        # Trivial case: node is deterministically True or False
        if node == 0:
            return self.semiring.one()
        elif node is None:
            return self.semiring.zero()

        query_def_sdd = self.__sdd.get_sddnode(node)
        constraint_sdd = self.__sdd.get_constraint_sdd()

        # Construct the query SDD
        evidence_sdd = self.sdd_manager.conjoin(
            *[self.__sdd.get_sddnode(ev) for ev in self.iterEvidence()])

        query_sdd = self.sdd_manager.conjoin(query_def_sdd, evidence_sdd, constraint_sdd)

        if self.sdd_manager.is_true(query_sdd):
            if node < 0:
                return self.__probs[-node][1]
            else:
                return self.__probs[node][0]
        elif self.sdd_manager.is_false(query_sdd):
            return self.semiring.zero()
        else:
            pall = self.semiring.zero()
            for path in query_sdd.satisfy_all():
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

    def setEvidence(self, index, value ) :
        pos = self.semiring.one()
        neg = self.semiring.zero()
        if value :
            self.setWeight( index, pos, neg )
        else :
            self.setWeight( index, neg, pos )

    def setWeight(self, index, pos, neg) :
        self.__probs[index] = (pos, neg)
