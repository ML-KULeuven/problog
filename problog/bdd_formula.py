"""
__author__ = Anton Dries

Provides access to Binary Decision Diagrams (BDDs).

"""

from __future__ import print_function

from .formula import LogicDAG
from .core import transform, InstallError
from .dd_formula import DD, build_dd, DDManager

try:
    import pyeda.boolalg.bdd as bdd
    import pyeda.boolalg.expr as bdd_expr

except Exception:
    bdd = None


class BDD(DD):
    """A propositional logic formula consisting of and, or, not and atoms represented as an BDD."""

    def __init__(self, **kwdargs):
        if bdd is None:
            raise InstallError('The BDD library is not available.')

        DD.__init__(self, auto_compact=False, **kwdargs)

    def create_manager(self):
        return BDDManager()

    @classmethod
    def is_available(cls) :
        return bdd is not None


class BDDManager(DDManager):
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
        DDManager.__init__(self)
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
            return self.varcount
        else:
            return label

    def literal(self, label):
        """
        Return an BDD node representing a literal.
        :param label: label of the literal
        :type label: int
        :return: BDD node representing the literal
        :rtype: BDDNode
        """
        return bdd.bddvar('v' + str(self.add_variable(label)))

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

    def conjoin2(self, r, s):
        return r & s

    def disjoin2(self, r, s):
        return r | s

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
        with open(filename, 'w') as f:
            print (node.to_dot(), file=f)

    def wmc(self, node, weights, semiring):
        if self.is_true(node):
            if node < 0:
                return weights[-node][1]
            else:
                return weights[node][0]
        elif self.is_false(node):
            return semiring.zero()
        else:
            pall = semiring.zero()
            for path in node.satisfy_all():
                pw = semiring.one()
                for var, val in path.items():
                    var = int(var.name[1:])
                    pos, neg = weights[var]
                    if val:
                        p = pos
                    else:
                        p = neg
                    pw = semiring.times(p, pw)
                pall = semiring.plus(pw, pall)
            return pall

    def __del__(self):
        """
        Clean up the BDD manager.
        """
        pass

            
@transform(LogicDAG, BDD)
def build_bdd(source, destination, **kwdargs):
    return build_dd(source, destination, 'BDD', **kwdargs)