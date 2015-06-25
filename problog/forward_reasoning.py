"""
Module name
"""

from __future__ import print_function
from .formula import LogicFormula, OrderedSet
from .dd_formula import DD, build_dd
from .sdd_formula import SDD
from .bdd_formula import BDD
from .core import transform

import signal

from collections import OrderedDict, defaultdict, namedtuple

def timeout_handler(signum, frame):
    raise SystemError('Process timeout (Python) [%s]' % signum)

class ForwardInference(DD):

    def __init__(self, timeout=None, **kwdargs):
        super(ForwardInference, self).__init__(**kwdargs)

        self._inodes_prev = None
        self._inodes_neg = None
        self._facts = None
        self._atoms_in_rules = None

        self.timeout = timeout

        self._node_depths = None

    def _create_atom(self, identifier, probability, group, name=None):
        return self._atom(identifier, probability, group, name)

    def init_build(self):
        self._facts = []    # list of facts
        self._atoms_in_rules = defaultdict(set)    # lookup all rules in which an atom is used

        self.compute_node_depths()
        for index, node, nodetype in self:
            if self._node_depths[index-1] is not None:
                # only include nodes that are reachable from a query or evidence
                if nodetype == 'atom':   # it's a fact
                    self._facts.append(index)
                    self.atom2var[index] = self.get_manager().add_variable(index)
                else:    # it's a compound
                    for atom in node.children:
                        self._atoms_in_rules[abs(atom)].add(index)
        self.inodes = [None] * len(self)
        self._inodes_prev = [None] * len(self)
        self._inodes_neg = [None] * len(self)

    def compute_node_depths(self):
        self._node_depths = [None] * len(self)
        for q, n in self.queries():
            if n != 0 and n is not None:
                self._compute_node_depths(abs(n), 0)

        for q, n in self.evidence():
            if n != 0 and n is not None:
                self._compute_node_depths(abs(n), 0)

    def _compute_node_depths(self, index, depth):
        current_depth = self._node_depths[index-1]
        if current_depth is None or current_depth > depth:
            self._node_depths[index-1] = depth
            node = self.get_node(index)
            nodetype = type(node).__name__
            if nodetype != 'atom':
                for c in node.children:
                    self._compute_node_depths(abs(c), depth+1)

    def sort_nodes(self, nodes):
        return sorted(nodes, key=lambda i: -self._node_depths[i-1])

    def build_stratum(self, updated_nodes):
        while updated_nodes:
            next_updates = OrderedSet()
            # Find all heads that are affected
            affected_nodes = OrderedSet()
            for node in updated_nodes:
                for rule in self._atoms_in_rules[node]:
                    affected_nodes.add(rule)
            affected_nodes = self.sort_nodes(affected_nodes)
            for head in affected_nodes:
                if self.update_inode(head):
                    next_updates.add(head)
            updated_nodes = next_updates

        updated_nodes = OrderedSet()
        for i, nodes in enumerate(zip(self.inodes, self._inodes_prev)):
            if not self.get_manager().same(*nodes):
                updated_nodes.add(i)
        self.get_manager().ref(*filter(None, self.inodes))
        self.get_manager().deref(*filter(None, self._inodes_prev))
        self.get_manager().deref(*filter(None, self._inodes_neg))
        self._inodes_prev = self.inodes[:]
        self._inodes_neg = [None] * len(self)
        return updated_nodes

    def build_dd(self):
        if self.timeout:
            # signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)
            signal.signal(signal.SIGALRM, timeout_handler)
            print ('Set timeout:', self.timeout)
        try:
            self.init_build()
            updated_nodes = OrderedSet(self._facts)
            while updated_nodes:
                updated_nodes = self.build_stratum(updated_nodes)
        except SystemError as err:
            print (err)
        signal.alarm(0)

    def update_inode(self, index):
        """Recompute the inode at the given index."""
        oldnode = self.get_inode(index)
        node = self.get_node(index)
        nodetype = type(node).__name__
        if nodetype == 'conj':
            children = [self.get_inode(c) for c in node.children]
            if None in children:
                newnode = None  # don't compute if some children are still unknown
            else:
                newnode = self.get_manager().conjoin(*children)
        elif nodetype == 'disj':
            children = [oldnode] + [self.get_inode(c) for c in node.children]
            children = list(filter(None, children))   # discard children that are still unknown
            if children:
                newnode = self.get_manager().disjoin(*children)
            else:
                newnode = None
        else:
            raise TypeError('Unexpected node type.')
        if self.get_manager().same(oldnode, newnode):
            return False   # no change occurred
        else:
            if oldnode is not None:
                self.get_manager().deref(oldnode)
            self.set_inode(index, newnode)
            return True

    def get_inode(self, index):
        """
        Get the internal node corresponding to the entry at the given index.
        :param index: index of node to retrieve
        :return: SDD node corresponding to the given index
        :rtype: SDDNode
        """
        node = self.get_node(abs(index))
        if type(node).__name__ == 'atom':
            result = self.get_manager().literal(self.atom2var[abs(index)])
            if index < 0:
                return self.get_manager().negate(result)
            else:
                return result
        elif index < 0:
            # We are requesting a negated node => use previous stratum's result
            result = self._inodes_neg[-index-1]
            if result is None and self._inodes_prev[-index-1] is not None:
                result = self.get_manager().negate(self._inodes_prev[-index-1])
                self._inodes_neg[-index-1] = result
            return result
        else:
            return self.inodes[index-1]

class ForwardSDD(SDD, ForwardInference):

    def __init__(self, **kwdargs):
        SDD.__init__(self, sdd_auto_gc=True, **kwdargs)
        ForwardInference.__init__(self, **kwdargs)

class ForwardBDD(BDD, ForwardInference):

    def __init__(self, **kwdargs):
        BDD.__init__(self, **kwdargs)
        ForwardInference.__init__(self, **kwdargs)


@transform(LogicFormula, ForwardSDD)
def build_sdd(source, destination, **kwdargs):
    result = build_dd(source, destination, 'ForwardSDD', **kwdargs)
    result.build_dd()
    return result

@transform(LogicFormula, ForwardBDD)
def build_sdd(source, destination, **kwdargs):
    result = build_dd(source, destination, 'ForwardBDD', **kwdargs)
    result.build_dd()
    return result
