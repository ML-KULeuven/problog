"""
__author__ = Anton Dries

Provides access to Sentential Decision Diagrams (SDDs).

"""

from __future__ import print_function

from .formula import LogicDAG
from .core import transform, InstallError
from .dd_formula import DD, build_dd, DDManager

try:
    import sdd
except Exception:
    sdd = None


class SDD(DD):
    """A propositional logic formula consisting of and, or, not and atoms represented as an SDD.

    This class has two restrictions with respect to the default LogicFormula:

        * The number of atoms in the SDD should be known at construction time.
        * It does not support updatable nodes.

    This means that this class can not be used directly during grounding.
    It can be used as a target for the ``makeAcyclic`` method.
    """

    def __init__(self, sdd_auto_gc=False, **kwdargs):
        if sdd is None:
            raise InstallError('The SDD library is not available. Please run the installer.')
        self.auto_gc = sdd_auto_gc
        DD.__init__(self, auto_compact=False, **kwdargs)

    def create_manager(self):
        return SDDManager(auto_gc=self.auto_gc)

    @classmethod
    def is_available(cls) :
        return sdd is not None


class SDDManager(DDManager):
    """
    Manager for SDDs.
    It wraps around the SDD library and offers some additional methods.
    """

    def __init__(self, varcount=0, auto_gc=True):
        """
        Create a new SDD manager.
        :param varcount: number of initial variables
        :type varcount: int
        :param auto_gc: use automatic garbage collection and minimization
        :type auto_gc: bool
        """
        DDManager.__init__(self)
        if varcount is None or varcount == 0:
            varcount = 1
        self.__manager = sdd.sdd_manager_create(varcount, auto_gc)
        self.varcount = varcount

    def get_manager(self):
        """
        Get the underlying sdd manager.
        :return:
        """
        return self.__manager

    def add_variable(self, label=0):
        """
        Add a variable to the manager and return its label.
        :param label: suggested label of the variable
        :type label: int
        :return: label of the new variable
        :rtype: int
        """
        if label == 0 or label > self.varcount:
            sdd.sdd_manager_add_var_after_last(self.__manager)
            self.varcount += 1
            return self.varcount
        else:
            return label

    def literal(self, label):
        """
        Return an SDD node representing a literal.
        :param label: label of the literal
        :type label: int
        :return: SDD node representing the literal
        :rtype: SDDNode
        """
        self.add_variable(label)
        return sdd.sdd_manager_literal(label, self.__manager)

    def is_true(self, node):
        """
        Checks whether the SDD node represents True
        :param node: node to verify
        :type node: SDDNode
        :return: True if the node represents True
        :rtype: bool
        """
        return sdd.sdd_node_is_true(node)

    def true(self):
        """
        Return an SDD node representing True
        :return:
        """
        return sdd.sdd_manager_true(self.__manager)

    def is_false(self, node):
        """
        Checks whether the SDD node represents False
        :param node: node to verify
        :type node: SDDNode
        :return: False if the node represents False
        :rtype: bool
        """
        return sdd.sdd_node_is_false(node)

    def false(self):
        """
        Return an SDD node representing False
        :return:
        """
        return sdd.sdd_manager_false(self.__manager)

    def conjoin2(self, node1, node2):
        return sdd.sdd_conjoin(node1, node2, self.__manager)

    def disjoin2(self, node1, node2):
        return sdd.sdd_disjoin(node1, node2, self.__manager)

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
        new_sdd = sdd.sdd_negate(node, self.__manager)
        self.ref(new_sdd)
        return new_sdd

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
        if node1 is None or node2 is None:
            return node1 == node2
        else:
            return int(node1) == int(node2)

    def ref(self, *nodes):
        """
        Increase the reference count for the given nodes.
        :param nodes: nodes to increase count on
        :type nodes: tuple of SDDNode
        """
        for node in nodes:
            sdd.sdd_ref(node, self.__manager)

    def deref(self, *nodes):
        """
        Decrease the reference count for the given nodes.
        :param nodes: nodes to decrease count on
        :type nodes: tuple of SDDNode
        """
        for node in nodes:
            sdd.sdd_deref(node, self.__manager)

    def write_to_dot(self, node, filename):
        """
        Write SDD node to a DOT file.
        :param node: SDD node to output
        :type node: SDDNode
        :param filename: filename to write to
        :type filename: basestring
        """
        sdd.sdd_save_as_dot(filename, node)

    def wmc(self, node, weights, semiring):
        """
        Perform Weighted Model Count on the given node.
        :param node: node to evaluate
        :param weights: weights for the variables in the node
        :param semiring: use the operations defined by this semiring
        :return: weighted model count
        """
        logspace = 0
        if semiring.isLogspace():
            logspace = 1
        wmc_manager = sdd.wmc_manager_new(node, logspace, self.get_manager())
        for i, n in enumerate(sorted(weights)):
            i += 1
            pos, neg = weights[n]
            sdd.wmc_set_literal_weight(n, pos, wmc_manager)   # Set positive literal weight
            sdd.wmc_set_literal_weight(-n, neg, wmc_manager)  # Set negative literal weight
        result = sdd.wmc_propagate(wmc_manager)
        sdd.wmc_manager_free(wmc_manager)
        return result

    def wmc_literal(self, node, weights, semiring, literal):
        logspace = 0
        if semiring.isLogspace():
            logspace = 1
        wmc_manager = sdd.wmc_manager_new(node, logspace, self.get_manager())
        for i, n in enumerate(sorted(weights)):
            i += 1
            pos, neg = weights[n]
            sdd.wmc_set_literal_weight(n, pos, wmc_manager)   # Set positive literal weight
            sdd.wmc_set_literal_weight(-n, neg, wmc_manager)  # Set negative literal weight
        sdd.wmc_propagate(wmc_manager)

        result = sdd.wmc_literal_pr(literal, wmc_manager)
        sdd.wmc_manager_free(wmc_manager)
        return result

    def wmc_true(self, weights, semiring):
        logspace = 0
        if semiring.isLogspace():
            logspace = 1
        wmc_manager = sdd.wmc_manager_new(self.true(), logspace, self.get_manager())
        for i, n in enumerate(sorted(weights)):
            i += 1
            pos, neg = weights[n]
            sdd.wmc_set_literal_weight(n, pos, wmc_manager)   # Set positive literal weight
            sdd.wmc_set_literal_weight(-n, neg, wmc_manager)  # Set negative literal weight
        result = sdd.wmc_propagate(wmc_manager)
        sdd.wmc_manager_free(wmc_manager)
        return result



    def __del__(self):
        """
        Clean up the SDD manager.
        """
        sdd.sdd_manager_free(self.__manager)
        self.__manager = None


@transform(LogicDAG, SDD)
def build_sdd(source, destination, **kwdargs):
    return build_dd(source, destination, 'SDD', **kwdargs)
