"""
problog.bdd_formula - Binary Decision Diagrams
----------------------------------------------

Provides access to Binary Decision Diagrams (BDDs).

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

from .formula import LogicDAG
from .core import transform
from .errors import InstallError
from .dd_formula import DD, build_dd, DDManager

# noinspection PyBroadException
# noinspection PyUnresolvedReferences
try:
    # noinspection PyPackageRequirements
    import pyeda.boolalg.bdd as bdd
    # noinspection PyPackageRequirements
    import pyeda.boolalg.expr as bdd_expr
except Exception:
    bdd = None


class BDD(DD):
    """A propositional logic formula consisting of and, or, not and atoms represented as an BDD."""

    def __init__(self, **kwdargs):
        if bdd is None:
            raise InstallError('The BDD library is not available.')

        DD.__init__(self, auto_compact=False, **kwdargs)

    def _create_manager(self):
        return BDDManager()

    def get_atom_from_inode(self, node):
        """Get the original atom given an internal node.

        :param node: internal node
        :return: atom represented by the internal node
        """
        return self.var2atom[self.get_manager().get_variable(node)]

    @classmethod
    def is_available(cls):
        """Checks whether the BDD library is available."""
        return bdd is not None


class BDDManager(DDManager):
    """
    Manager for BDDs.
    It wraps around the pyeda BDD module
    """

    # noinspection PyUnusedLocal
    def __init__(self, varcount=0, auto_gc=True):
        """Create a new BDD manager.

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
        if label == 0 or label > self.varcount:
            self.varcount += 1
            return self.varcount
        else:
            return label

    def get_variable(self, node):
        """Get the variable represented by the given node.

        :param node: internal node
        :return: original node
        """
        # noinspection PyProtectedMember
        return int(bdd._VARS[node.root].name[1:])

    def literal(self, label):
        return bdd.bddvar('v' + str(self.add_variable(label)))

    def is_true(self, node):
        return node.is_one()

    def true(self):
        return self.ONE

    def is_false(self, node):
        return node.is_zero()

    def false(self):
        return self.ZERO

    def conjoin2(self, r, s):
        return r & s

    def disjoin2(self, r, s):
        return r | s

    def negate(self, node):
        return ~node

    def same(self, node1, node2):
        # Assumes BDD library always reuses equivalent nodes.
        return node1 is node2

    def ref(self, *nodes):
        pass

    def deref(self, *nodes):
        pass

    def write_to_dot(self, node, filename):
        with open(filename, 'w') as f:
            print(node.to_dot(), file=f)

    def wmc(self, node, weights, semiring):
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

    def wmc_literal(self, node, weights, semiring, literal):
        raise NotImplementedError('not supported')

    def wmc_true(self, weights, semiring):
        return semiring.one()

    def __del__(self):
        pass


@transform(LogicDAG, BDD)
def build_bdd(source, destination, **kwdargs):
    """Build an SDD from another formula.

    :param source: source formula
    :param destination: destination formula
    :param kwdargs: extra arguments
    :return: destination
    """
    return build_dd(source, destination, **kwdargs)
