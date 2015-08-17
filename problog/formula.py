"""
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

from collections import namedtuple, defaultdict
import warnings

from .core import transform, InconsistentEvidenceError

from .base_formula import BaseFormula

from .util import Timer
from .logic import Term

import logging
import tempfile
import subprocess
import os


class LogicFormula(BaseFormula):
    """A logic formula is a data structure that is used to represent generic And-Or graphs.
    It can typically contain three types of nodes:

        - atom ( or terminal)
        - and (compound)
        - or (compound)

    The compound nodes contain a list of children which point to other nodes in the formula.
    These pointers can be positive or negative.

    In addition to the basic logical structure of the formula, it also maintains a table of labels,
    which can be used to easily retrieve certain nodes.
    These labels typically contain the literals from the original program.

    Upon addition of new nodes, the logic formula can perform certain optimizations, for example,
    by simplifying nodes or by reusing existing nodes.
    """

    _atom = namedtuple('atom', ('identifier', 'probability', 'group', 'name'))
    _conj = namedtuple('conj', ('children', 'name'))
    _disj = namedtuple('disj', ('children', 'name'))
    # negation is encoded by using a negative number for the key

    def _create_atom(self, identifier, probability, group, name=None):
        return self._atom(identifier, probability, group, name)

    def _create_conj(self, children, name=None):
        return self._conj(children, name)

    def _create_disj( self, children, name=None):
        return self._disj(children, name)

    def __init__(self, auto_compact=True, avoid_name_clash=False, keep_order=False,
                 use_string_names=False, keep_all=False, propagate_weights=None, **kwdargs):
        BaseFormula.__init__(self)

        # List of nodes
        self._nodes = []
        # Lookup index for 'atom' nodes, key is identifier passed to addAtom()
        self._index_atom = {}
        # Lookup index for 'and' nodes, key is tuple of sorted children
        self._index_conj = {}
        # Lookup index for 'or' nodes, key is tuple of sorted children
        self._index_disj = {}

        self._atomcount = 0

        self._auto_compact = auto_compact
        self._avoid_name_clash = avoid_name_clash
        self._keep_order = keep_order
        self._keep_all = keep_all

        self._constraints_me = {}

        self.semiring = propagate_weights

        self._use_string_names = use_string_names

    # ====================================================================================== #
    # ==========                         MANAGE LABELS                           =========== #
    # ====================================================================================== #

    def add_name(self, name, key, label=None):
        """Associates a name to the given node identifier.

            :param name: name of the node
            :param key: id of the node
            :param label: type of node (see LogicFormula.LABEL_*)
        """
        if self._use_string_names:
            name = str(name)

        if self.is_probabilistic(key):
            node = self.get_node(abs(key))
            node = type(node)(*(node[:-1] + (name,)))
            self._update(abs(key), node)

        BaseFormula.add_name(self, name, key, label)

    # ====================================================================================== #
    # ==========                          MANAGE LOGIC                           =========== #
    # ====================================================================================== #

    def is_trivial(self):
        """Test whether the formula contains any logical construct.

        :return: False if the formula only contains atoms.
        """
        return self.atomcount == len(self)

    def _add(self, node, key=None, reuse=True):
        """Adds a new node, or reuses an existing one.

        :param node: node to add
        :param reuse: (default True) attempt to map the new node onto an existing one based on its \
         content

        """
        if reuse:
            # Determine the node's key and lookup identifier base on node type.
            ntype = type(node).__name__
            if ntype == 'atom':
                key = node.identifier
                collection = self._index_atom
            elif ntype == 'conj':
                key = node.children
                collection = self._index_conj
            elif ntype == 'disj':
                key = node.children
                collection = self._index_disj
            else:
                raise TypeError("Unexpected node type: '%s'." % ntype)

            if key not in collection:
                # Create a new entry, starting from 1
                index = len(self._nodes) + 1
                # Add the entry to the collection
                collection[key] = index
                # Add entry to the set of nodes
                self._nodes.append(node)
            else:
                # Retrieve the entry from collection
                index = collection[key]
        else:
            # Don't reuse, just add node.
            index = len(self._nodes) + 1
            self._nodes.append(node)

        # Return the entry
        return index

    def _update(self, key, value):
        """Replace the node with the given node."""
        assert(self.is_probabilistic(key))
        assert(key > 0)
        self._nodes[key - 1] = value

    def _add_constraint_me(self, group, node):
        if group is None:
            return node
        constraint = self._constraints_me.get(group)
        if constraint is None:
            constraint = ConstraintAD(group)
            self._constraints_me[group] = constraint
        node = constraint.add(node, self)
        return node

    def add_atom(self, identifier, probability, group=None, name=None):
        """Add an atom to the formula.

        :param identifier: a unique identifier for the atom
        :param probability: probability of the atom
        :param group: a group identifier that identifies mutually exclusive atoms (or None if no \
        constraint)
        :returns: the identifiers of the node in the formula (returns self.TRUE for deterministic \
        atoms)

        This function has the following behavior :

        * If ``probability`` is set to ``None`` then the node is considered to be deterministically\
         true and the function will return :attr:`TRUE`.
        * If a node already exists with the given ``identifier``, the id of that node is returned.
        * If ``group`` is given, a mutual exclusivity constraint is added for all nodes sharing the\
         same group.
        * To add an explicitly present deterministic node you can set the probability to ``True``.
        """
        if probability is None and not self._keep_all:
            return self.TRUE
        elif probability != self.WEIGHT_NEUTRAL and self.semiring and \
                self.semiring.is_zero(self.semiring.value(probability)):
            return self.FALSE
        elif probability != self.WEIGHT_NEUTRAL and self.semiring and \
                self.semiring.is_one(self.semiring.value(probability)):
            return self.TRUE
        else:
            atom = self._create_atom(identifier, probability, group, name)
            node_id = self._add(atom, key=identifier)
            self.set_weight(node_id, probability)
            if node_id == len(self._nodes):
                self._atomcount += 1
            return self._add_constraint_me(group, node_id)

    def add_and(self, components, key=None, name=None):
        """Add a conjunction to the logic formula.

        :param components: a list of node identifiers that already exist in the logic formula.
        :returns: the key of the node in the formula (returns 0 for deterministic atoms)
        """
        return self._add_compound('conj', components, self.FALSE, self.TRUE, key=key, name=name)

    def add_or(self, components, key=None, readonly=True, name=None):
        """Add a disjunction to the logic formula.

        :param components: a list of node identifiers that already exist in the logic formula.
        :param readonly: indicates whether the node should be modifiable. This will allow \
        additional disjunct to be added without changing the node key. Modifiable nodes are \
        less optimizable.
        :returns: the key of the node in the formula (returns 0 for deterministic atoms)
        :rtype: :class:`int`

        By default, all nodes in the data structure are immutable (i.e. readonly).
        This allows the data structure to optimize nodes, but it also means that cyclic formula \
        can not be stored because the identifiers of all descendants must be known add creation \
        time.

        By setting `readonly` to False, the node is made mutable and will allow adding disjunct \
        later using the :func:`addDisjunct` method.
        This may cause the data structure to contain superfluous nodes.
        """
        return self._add_compound('disj', components, self.TRUE, self.FALSE, key=key,
                                  readonly=readonly, name=name)

    def add_disjunct(self, key, component):
        """Add a component to the node with the given key.

        :param key: id of the node to update
        :param component: the component to add
        :return: key
        :raises: :class:`ValueError` if ``key`` points to an invalid node

        This may only be called with a key that points to a disjunctive node or :attr:`TRUE`.
        """
        if self.is_true(key):
            return key
        elif self.is_false(key):
            raise ValueError("Cannot update failing node")
        else:
            node = self.get_node(key)
            if type(node).__name__ != 'disj':
                raise ValueError("Can only update disjunctive node")

            if component is None:
                # Don't do anything
                pass
            elif component == 0:
                return self._update(key, self._create_disj((0,), name=node.name))
            else:
                return self._update(key, self._create_disj(node.children + (component,),
                                                           name=node.name))
            return key

    def add_not(self, component):
        """Returns the key to the negation of the node.

        :param component: the node to negate
        """
        return self.negate(component)

    # def isAtom(self, key) :
    #     return self._getNodeType(key) == 'atom'
    #
    # def isAnd(self, key) :
    #     return self._getNodeType(key) == 'conj'
    #
    # def isOr(self, key) :
    #     return self._getNodeType(key) == 'disj'
    #
    # def isCompound(self, key) :
    #     return not self.isAtom(key)

    def get_node(self, key):
        assert self.is_probabilistic(key)
        assert key > 0
        return self._nodes[key - 1]

    # def _getNodeType(self, key) :
    #     """Get the type of the given node (fact, disj, conj)."""
    #     return type(self.getNode(key)).__name__

    def _add_compound(self, nodetype, content, t, f, key=None,
                      readonly=True, update=None, name=None):
        """Add a compound term (AND or OR)."""
        assert content   # Content should not be empty

        if self._auto_compact:
            # If there is a t node, (true for OR, false for AND)
            if t in content:
                return t

            # Eliminate unneeded node nodes (false for OR, true for AND)
            content = filter(lambda x: x != f, content)

            # Put into fixed order and eliminate duplicate nodes
            if self._keep_order:
                content = tuple(content)
            else:
                content = tuple(sorted(set(content)))

            # Empty OR node fails, AND node is true
            if not content:
                return f

            # Contains opposites: return 'TRUE' for or, 'FALSE' for and
            if len(set(content)) > len(set(map(abs, content))):
                return t

            # If node has only one child, just return the child.
            # Don't do this for modifiable nodes, we need to keep a separate node.
            if (readonly and update is None) and len(content) == 1:
                if self._avoid_name_clash:
                    name_old = self.get_node(abs(content[0])).name
                    if name is None or name_old is None or name == name_old:
                        return content[0]
                else:
                    return content[0]
        else:
            content = tuple(content)

        if nodetype == 'conj':
            node = self._create_conj(content, name)
            return self._add(node, reuse=self._auto_compact)
        elif nodetype == 'disj':
            node = self._create_disj(content, name)
            if update is not None:
                # If an update key is set, update that node
                return self._update(update, node)
            elif readonly:
                # If the node is readonly, we can try to reuse an existing node.
                return self._add(node, reuse=self._auto_compact)
            else:
                # If node is modifiable, we shouldn't reuse an existing node.
                return self._add(node, reuse=False)
        else:
            raise TypeError("Unexpected node type: '%s'." % nodetype)

    def __iter__(self):
        """Iterate over the nodes in the formula.

            :returns: iterator over tuples ( key, node, type )
        """
        for i, n in enumerate(self._nodes):
            yield (i+1, n, type(n).__name__)

    def __len__(self):
        """Returns the number of nodes in the formula."""
        return len(self._nodes)

    # ====================================================================================== #
    # ==========                       MANAGE CONSTRAINTS                        =========== #
    # ====================================================================================== #

    def constraints(self):
        """Returns a list of all constraints."""
        return list(self._constraints_me.values()) + BaseFormula.constraints(self)

    # ====================================================================================== #
    # ==========                      EVIDENCE PROPAGATION                       =========== #
    # ====================================================================================== #

    # TODO this code does not belong here?

    def has_evidence_values(self):
        return hasattr(self, 'lookup_evidence')

    def get_evidence_value(self, node):
        if node == 0 or node is None:
            return node
        elif self.has_evidence_values():
            result = self.lookup_evidence.get(abs(node), abs(node))
            if node < 0:
                return self.negate(result)
            else:
                return result
        else:
            return node

    def set_evidence_value(self, node, value):
        if node < 0:
            self.lookup_evidence[-node] = self.negate(value)
        else:
            self.lookup_evidence[node] = value

    def propagate(self, nodeids, current=None):
        """Propagate the value of the given node (true if node is positive, false if node is negative)
        The propagation algorithm is not complete.

        :param nodeid:
        :param current:
        :return:
        """
        if current is None:
            current = {}

        values = {True: self.TRUE, False: self.FALSE}
        atoms_in_rules = defaultdict(set)

        updated = set()
        queue = set(nodeids)
        while queue:
            nid = queue.pop()

            if abs(nid) not in current:
                updated.add(abs(nid))
                for at in atoms_in_rules[abs(nid)]:
                    if at in current:
                        if current[abs(at)] == 0:
                            queue.add(abs(at))
                        else:
                            queue.add(-abs(at))
                current[abs(nid)] = values[nid > 0]

            n = self.get_node(abs(nid))
            t = type(n).__name__
            if t == 'atom':
                pass
            else:
                children = []
                for c in n.children:
                    ch = current.get(abs(c), abs(c))
                    if c < 0:
                        ch = self.negate(ch)
                    children.append(ch)
                if t == 'conj' and None in children and nid > 0:
                    raise InconsistentEvidenceError()
                elif t == 'disj' and 0 in children and nid < 0:
                    raise InconsistentEvidenceError()
                children = list(filter(lambda x: x != 0 and x is not None, children))
                if len(children) == 1:  # only one child
                    if abs(children[0]) not in current:
                        if nid < 0:
                            queue.add(-children[0])
                        else:
                            queue.add(children[0])
                        atoms_in_rules[abs(children[0])].discard(abs(nid))
                elif nid > 0 and t == 'conj':
                    # Conjunction is true
                    for c in children:
                        if abs(c) not in current:
                            queue.add(c)
                        atoms_in_rules[abs(c)].discard(abs(nid))
                elif nid < 0 and t == 'disj':
                    # Disjunction is false
                    for c in children:
                        if abs(c) not in current:
                            queue.add(-c)
                else:
                    for c in children:
                        atoms_in_rules[abs(c)].add(abs(nid))
        return current

    # ====================================================================================== #
    # ==========                        EXPORT TO STRING                         =========== #
    # ====================================================================================== #

    def __str__(self):
        s = '\n'.join('%s: %s' % (i, n) for i, n, t in self)
        f = True
        for q in self.queries():
            if f:
                f = False
                s += '\nQueries : '
            s += '\n* %s : %s' % q

        f = True
        for q in self.evidence():
            if f:
                f = False
                s += '\nEvidence : '
            s += '\n* %s : %s' % q

        f = True
        for c in self.constraints():
            if c.isActive():
                if f:
                    f = False
                    s += '\nConstraints : '
                s += '\n* ' + str(c)
        return s + '\n'

    def to_prolog(self):
        """Convert the Logic Formula to a Prolog program.

        To make this work correctly some flags should be set on the engine and LogicFormula prior \
        to grounding.
        The following code should be used:

        .. code-block:: python

            pl = problog.program.PrologFile(inFile)
            eng = problog.engine.DefaultEngine(label_all=True)

            gp = problog.formula.LogicFormula(avoid_name_clash=True, keep_order=True)
            gp = eng.ground_all(pl, target=gp)

            prologfile = gp.to_prolog()

        :return: Prolog program
        :rtype: str
        """

        lines = []
        for head, body in self.enumerate_clauses():
            if body:    # clause with a body
                body = ', '.join(map(str, map(self.get_name, body)))
                lines.append('%s :- %s.' % (self.get_name(head), body))
            else:   # fact
                prob = self.get_node(head).probability
                if prob is not None:
                    lines.append('%s::%s.' % (prob, self.get_name(head)))
                else:
                    lines.append('%s.' % self.get_name(head))
        return '\n'.join(lines)

    def get_name(self, nodeid):
        """Get the name of the given node."""
        if nodeid == 0:
            return 'true'
        elif nodeid is None:
            return 'false'
        else:
            node = self.get_node(abs(nodeid))
            name = node.name
            if name is None:
                name = Term('node_%s' % abs(nodeid))
            if nodeid < 0:
                return -name
            else:
                return name

    def enumerate_clauses(self, relevant_only=True):
        """Enumerate the clauses of this logic formula.
            Clauses are represented as ('head, [body]').
        """

        enumerated = OrderedSet()

        if relevant_only:
            to_enumerate = OrderedSet(abs(n) for q, n in self.queries() if self.is_probabilistic(n))
            to_enumerate |= OrderedSet(abs(n)
                                       for q, n in self.evidence() if self.is_probabilistic(n))
        else:
            to_enumerate = OrderedSet(range(1, len(self)+1))

        while to_enumerate:
            i = to_enumerate.pop()
            enumerated.add(i)
            n = self.get_node(i)
            t = type(n).__name__

            if t == 'atom':
                yield i, []
            elif t == 'conj':
                # In case a query or evidence is directly referring to a conjunctive node.
                body = self._unroll_conj(n)
                to_enumerate |= (OrderedSet((map(abs, body))) - enumerated)
                yield i, body
            else:   # t == 'disj'
                for c_i in n.children:
                    negc = (c_i < 0)
                    c_n = self.get_node(abs(c_i))
                    c_t = type(c_n).__name__
                    if c_t == 'atom':
                        yield i, [c_i]
                        if abs(c_i) not in enumerated:
                            to_enumerate.add(abs(c_i))
                    elif c_t == 'conj':
                        if negc:
                            yield i, [c_i]
                            if abs(c_i) not in enumerated:
                                to_enumerate.add(abs(c_i))
                        else:
                            body = self._unroll_conj(c_n)
                            to_enumerate |= (OrderedSet(map(abs, body)) - enumerated)
                            yield i, body
                    else:
                        yield i, [c_i]
                        if abs(c_i) not in enumerated:
                            to_enumerate.add(abs(c_i))

    def _unroll_conj(self, node):
        assert type(node).__name__ == 'conj'

        if len(node.children) == 1:
            return node.children
        elif len(node.children) == 2 and node.children[1] > 0:
            children = [node.children[0]]
            current = node.children[1]
            current_node = self.get_node(current)
            while type(current_node).__name__ == 'conj' and len(current_node.children) == 2:
                children.append(current_node.children[0])
                current = current_node.children[1]
                if current > 0:
                    current_node = self.get_node(current)
                else:
                    current_node = None
            children.append(current)
        else:
            children = node.children
        return children

    def to_dot(self, not_as_node=True):
        """Write out in GraphViz (dot) format.

        :param not_as_node: represent negation as a node
        :return: string containing dot representation
        """

        not_as_edge = not not_as_node

        # Keep track of mutually disjunctive nodes.
        clusters = defaultdict(list)

        queries = set([(name, node) for name, node, label in self.get_names_with_label()])
        for i, n, t in self:
            if n.name is not None:
                queries.add((n.name, i))

        # Keep a list of introduced not nodes to prevent duplicates.
        negative = set([])

        s = 'digraph GP {\n'
        for index, node, nodetype in self:

            if nodetype == 'conj':
                s += '%s [label="AND", shape="box", style="filled", fillcolor="white"];\n' % index
                for c in node.children:
                    opt = ''
                    if c < 0 and c not in negative and not_as_node:
                        s += '%s [label="NOT"];\n' % c
                        s += '%s -> %s;\n' % (c, -c)
                        negative.add(c)

                    if c < 0 and not_as_edge:
                        opt = '[arrowhead="odotnormal"]'
                        c = -c
                    if c != 0:
                        s += '%s -> %s%s;\n' % (index, c, opt)
            elif nodetype == 'disj':
                s += '%s [label="OR", shape="diamond", style="filled", fillcolor="white"];\n' \
                     % index
                for c in node.children:
                    opt = ''
                    if c < 0 and c not in negative and not_as_node:
                        s += '%s [label="NOT"];\n' % c
                        s += '%s -> %s;\n' % (c, -c)
                        negative.add(c)
                    if c < 0 and not_as_edge:
                        opt = '[arrowhead="odotnormal"]'
                        c = -c
                    if c != 0:
                        s += '%s -> %s%s;\n' % (index, c, opt)
            elif nodetype == 'atom':
                if node.probability == self.WEIGHT_NEUTRAL:
                    pass
                elif node.group is None:
                    s += '%s [label="%s", shape="ellipse", style="filled", fillcolor="white"];\n' \
                         % (index, node.probability)
                else:
                    clusters[node.group].append('%s [ shape="ellipse", label="%s", '
                                                'style="filled", fillcolor="white" ];\n'
                                                % (index, node.probability))
            else:
                raise TypeError("Unexpected node type: '%s'" % nodetype)

        c = 0
        for cluster, text in clusters.items():
            if len(text) > 1:
                s += 'subgraph cluster_%s { style="dotted"; color="red"; \n\t%s\n }\n' \
                     % (c, '\n\t'.join(text))
            else:
                s += text[0]
            c += 1

        q = 0
        for name, index in set(queries):
            opt = ''
            if index is None:
                index = 'false'
                if not_as_node:
                    s += '%s [label="NOT"];\n' % index
                    s += '%s -> %s;\n' % (index, 0)
                elif not_as_edge:
                    opt = ', arrowhead="odotnormal"'
                if 0 not in negative:
                    s += '%s [label="true"];\n' % 0
                    negative.add(0)
            elif index < 0:  # and not index in negative :
                if not_as_node:
                    s += '%s [label="NOT"];\n' % index
                    s += '%s -> %s;\n' % (index, -index)
                    negative.add(index)
                elif not_as_edge:
                    index = -index
                    opt = ', arrowhead="odotnormal"'
            elif index == 0 and index not in negative:
                s += '%s [label="true"];\n' % index
                negative.add(0)
            s += 'q_%s [ label="%s", shape="plaintext" ];\n' % (q, name)
            s += 'q_%s -> %s [style="dotted" %s];\n' % (q, index, opt)
            q += 1
        return s + '}'

    # ====================================================================================== #
    # ==========                        DEPRECATED NAMES                         =========== #
    # ====================================================================================== #

    def addAtom(self, identifier, probability, group=None, name=None):
        warnings.warn('addAtom() is deprecated', FutureWarning)
        return self.add_atom(identifier, probability, group, name)

    def addAnd(self, components, key=None, name=None):
        warnings.warn('addAnd() is deprecated', FutureWarning)
        return self.add_and(components, key, name)

    def addOr(self, components, key=None, readonly=True, name=None):
        warnings.warn('addOr() is deprecated', FutureWarning)
        return self.add_or(components, key, readonly, name)

    def addDisjunct(self, key, component):
        warnings.warn('addDisjunct() is deprecated', FutureWarning)
        return self.add_disjunct(key, component)

    def addNot(self, key):
        warnings.warn('addNot() is deprecated', FutureWarning)
        return self.add_not(key)

    def extractWeights(self, *a, **k):
        warnings.warn('extractWeights() is deprecated', FutureWarning)
        return self.extract_weights(*a, **k)

    def isTrue(self, key):
        """Indicates whether the given node is deterministically True."""
        warnings.warn('isTrue() is deprecated', FutureWarning)
        return self.is_true(key)

    def isFalse(self, key):
        """Indicates whether the given node is deterministically False."""
        warnings.warn('isFalse() is deprecated', FutureWarning)
        return self.is_false(key)

    def isProbabilistic(self, key):
        """Indicates whether the given node is probabilistic."""
        warnings.warn('isProbabilistic() is deprecated', FutureWarning)
        return self.is_probabilistic(key)

    def getNode(self, key) :
        warnings.warn('getNode() is deprecated', FutureWarning)
        return self.get_node(key)

    def _getNode(self, key):
        """Get the content of the given node."""
        warnings.warn('LogicFormula._getNode(key) is deprecated.', FutureWarning)
        return self.get_node(key)

    def getAtomCount(self):
        warnings.warn('getAtomCount() is deprecated', FutureWarning)
        return self.atomcount

    def isTrivial(self):
        warnings.warn('isTrivial() is deprecated', FutureWarning)
        return self.is_trivial()

    def addName(self, name, key, label=None):
        warnings.warn('addName() is deprecated', FutureWarning)
        return self.add_name(name, key, label)

    def addQuery(self, name, key):
        warnings.warn('addQuery() is deprecated', FutureWarning)
        return self.add_query(name, key)

    def addEvidence(self, name, key, value):
        warnings.warn('addEvidence() is deprecated', FutureWarning)
        return self.add_evidence(name, key, value)

    def getNames(self, label=None):
        warnings.warn('getNames() is deprecated', FutureWarning)
        return self.get_names(label)

    def getNodeByName(self, name):
        warnings.warn('getNodeByName() is deprecated', FutureWarning)
        return self.get_node_by_name(name)

    def getWeights(self) :
        """deprecated: see get_weights"""
        warnings.warn('LogicFormula.getWeights() is deprecated. ', FutureWarning)
        return self.get_weights()

    def iterNodes(self) :
        warnings.warn('LogicFormula.iterNodes() is deprecated. ', FutureWarning)
        return self.__iter__()


class LogicDAG(LogicFormula):

    def __init__(self, auto_compact=True, **kwdargs):
        LogicFormula.__init__(self, auto_compact, **kwdargs)


class DeterministicLogicFormula(LogicFormula):

    def __init__(self, **kwdargs):
        LogicFormula.__init__(self, **kwdargs)

    def add_atom(self, identifier, probability, group=None, name=None):
        return self.TRUE


@transform(LogicFormula, LogicDAG)
def break_cycles(source, target, **kwdargs):
    logger = logging.getLogger('problog')
    with Timer('Cycle breaking'):
        cycles_broken = set()
        content = set()
        translation = defaultdict(list)

        for q, n in source.queries():
            if source.is_probabilistic(n):
                newnode = _break_cycles(source, target, n, [], cycles_broken, content, translation)
            else:
                newnode = n
            target.add_name(q, newnode, target.LABEL_QUERY)

        translation = defaultdict(list)
        for q, n in source.evidence():
            if source.is_probabilistic(n):
                newnode = _break_cycles(source, target, abs(n), [], cycles_broken, content, translation, is_evidence=True)
            else:
                newnode = n
            if n < 0:
                target.add_name(q, newnode, target.LABEL_EVIDENCE_NEG)
            else:
                target.add_name(q, newnode, target.LABEL_EVIDENCE_POS)

        logger.debug("Ground program size: %s", len(target))
        return target

def _break_cycles(source, target, nodeid, ancestors, cycles_broken, content, translation, is_evidence=False):
    negative_node = nodeid < 0
    nodeid = abs(nodeid)

    if not is_evidence and not source.is_probabilistic(source.get_evidence_value(nodeid)):
        return source.get_evidence_value(nodeid)
    elif nodeid in ancestors:
        cycles_broken.add(nodeid)
        return None     # cyclic node: node is False
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
    if nodetype == 'atom':
        newnode = target.add_atom(node.identifier, node.probability, node.group, node.name)
    else:
        children = [_break_cycles(source, target, child, ancestors + [nodeid], child_cycles_broken, child_content, translation, is_evidence) for child in node.children]
        newname = node.name
        if newname is not None and child_cycles_broken:
            newfunc = newname.functor + '_cb_' + str(len(translation[nodeid]))
            newname = Term(newfunc, *newname.args)
        if nodetype == 'conj':
            newnode = target.add_and(children, name=newname)
        else:
            newnode = target.add_or(children, name=newname)

        if target.is_probabilistic(newnode):
            # Don't add the node if it is None
            # Also: don't add atoms (they can't be involved in cycles)
            content.add(nodeid)

    translation[nodeid].append((newnode, child_cycles_broken, child_content-child_cycles_broken))
    content |= child_content
    cycles_broken |= child_cycles_broken

    if negative_node:
        return target.negate(newnode)
    else:
        return newnode

class StringKeyLogicFormula(LogicFormula) :
    """A propositional logic formula consisting of and, or, not and atoms."""

    TRUE = 'true'
    FALSE = 'false'

    def __init__(self) :
        LogicFormula.__init__(self)

        self.__nodes = defaultdict(list)

        self.__constraints_me = {}
        self.__constraints = []

    def _add( self, node, key=None, reuse=True ) :
        """Adds a new node, or reuses an existing one.

        :param node: node to add
        :param reuse: (default True) attempt to map the new node onto an existing one based on its content

        """
        self.__nodes[key].append(node)
        return key

    def _update( self, key, value ) :
        """Replace the node with the given node."""
        self.__nodes[ key ] = [value]

    def add_not( self, component ) :
        """Returns the key to the negation of the node."""
        if self.isTrue(component) :
            return self.FALSE
        elif self.isFalse(component) :
            return self.TRUE
        elif component.startswith('-') :
            return component[1:]
        else :
            return '-' + component

    def getNode(self, key) :
        """Get the content of the given node."""
        warnings.warn('LogicFormula._getNode(key) is deprecated. Use LogicFormula.getNode(key) instead.', FutureWarning)
        n = self.__nodes[key]
        if len(n) > 1 :
            return self._create_disj(n)
        else :
            return n[0]

    def _getNode(self, key) :
        """Get the content of the given node."""
        warnings.warn('LogicFormula._getNode(key) is deprecated. Use LogicFormula.getNode(key) instead.', FutureWarning)
        n = self.__nodes[key]
        if len(n) > 1 :
            return self._create_disj(n)
        else :
            return n[0]

    def _addCompound(self, nodetype, content, t, f, key=None, readonly=True, update=None) :
        """Add a compound term (AND or OR)."""
        assert( content )   # Content should not be empty

        # #If there is a t node, (true for OR, false for AND)
        # if t in content : return t
        #
        # # Eliminate unneeded node nodes (false for OR, true for AND)
        # content = filter( lambda x : x != f, content )
        #
        # # Put into fixed order and eliminate duplicate nodes
        # content = tuple(sorted(set(content)))
        #
        # # Empty OR node fails, AND node is true
        # if not content : return f

        # # Contains opposites: return 'TRUE' for or, 'FALSE' for and
        # if len(set(content)) > len(set(map(abs,content))) : return t

        # If node has only one child, just return the child.
        # Don't do this for modifiable nodes, we need to keep a separate node.
        if len(content) == 1 : return self._add(content[0], key=key)

        content = tuple(content)

        if nodetype == 'conj' :
            node = self._create_conj( content )
            return self._add( node, key=key )
        elif nodetype == 'disj' :
            node = self._create_disj( content )
            if update != None :
                # If an update key is set, update that node
                return self._update( update, node )
            elif readonly :
                # If the node is readonly, we can try to reuse an existing node.
                return self._add( node, key=key )
            else :
                # If node is modifiable, we shouldn't reuse an existing node.
                return self._add( node, key=key, reuse=False )
        else :
            raise TypeError("Unexpected node type: '%s'." % nodetype)

    def _resolve(self, key) :
        if type(key) == str :
            return key

        res = self.__nodes.get(key,key)
        if type(res) == list :
            assert(len(res) == 1)
            return res[0]
        else :
            return res

    def _deref(self, x) :
        c = x
        neg = 0
        while type(c) == str :
            if c[0] == '-' :
                c = c[1:]
                neg += 1
            x = c
            if not c in self.__nodes : break
            nn = self.__nodes[c]
            if len(nn) == 1 :
                c = nn[0]
            else :
                break
        if neg % 2 == 0 :
            return x
        else :
            return '-' + x

    def __iter__(self) :
        for k in self.__nodes :
            n = self.__nodes[k]
            child_names = []
            children = []
            for x in n :
                if type(x) == str :
                    x = self._deref(x)
                    child_names.append(x)
                else :
                    key = '%s_%s' % (k,len(child_names) )
                    child_names.append( key )
                    if type(x).__name__ != 'atom' :
                        x_children = [self._deref(y) for y in x.children]
                        children.append( (key, self._create_conj(x_children) ) )
                    else :
                        children.append( (key, x ))
            if len(child_names) > 1 :
                yield (k, self._create_disj(child_names), 'disj')
                for i,c in children :
                    yield (i, c, type(c).__name__)
            else :
                for i,c in children :
                    yield (k, c, type(c).__name__)

    def __len__(self) :
        return len(self.__nodes)

    ##################################################################################
    ####                            OUTPUT GENERATION                             ####
    ##################################################################################

    def toLogicFormula(self) :
        target = LogicFormula(auto_compact=False)
        translate = {}
        i = 0
        for k,n,t in self :
            i += 1
            translate[k] = i
            translate['-' + str(k) ] = -i
        for k,n,t in self :
            if t == 'atom' :
                i = target.add_atom( n.identifier, n.probability, n.group )
            elif t == 'disj' :
                i = target.add_or( [ translate[x] for x in n.children ] )
            elif t == 'conj' :
                i = target.add_and( [ translate[x] for x in n.children ] )
            assert(i == translate[k])

        for name, key, label in self.get_names_with_label():
            key = self._deref(key)
            target.add_name(name, translate[key], label)

        return target

    @classmethod
    def loadFrom(cls, lp) :
        interm = StringKeyLogicFormula()
        for c in lp :
            if type(c).__name__ == 'Clause' :
                key = str(c.head)
                body = []
                current = c.body
                while type(current).__name__ == 'And' :
                    if type(current.op1).__name__ == 'Not' :
                        body.append('-' + str(current.op1.child))
                    else :
                        body.append(str(current.op1))
                    current = current.op2
                if type(current).__name__ == 'Not' :
                    body.append('-' + str(current.child))
                else :
                    body.append(str(current))
                interm.add_and( body, key=key )
                interm.add_name(key, key, interm.LABEL_NAMED)
            elif type(c).__name__ == 'Term' :
                key = str(c.withProbability())
                interm.add_atom( key, c.probability, None )
                interm.add_name(key, key, interm.LABEL_NAMED)
            else :
                raise Exception("Unexpected type: '%s'" % type(c).__name__)
        return interm



class Constraint(object) :

    def getNodes(self) :
        """Get all nodes involved in this constraint."""
        return NotImplemented('Constraint.getNodes() is an abstract method.')

    def updateWeights(self, weights, semiring) :
        # Typically, constraints don't update weights
        pass

class ConstraintAD(Constraint) :
    """Annotated disjunction constraint (mutually exclusive with weight update)."""

    def __init__(self, group) :
        self.nodes = set()
        self.group = group
        self.extra_node = None

    def __str__(self) :
        return 'annotated_disjunction(%s, %s)' % (list(self.nodes), self.extra_node)

    def getNodes(self) :
        if self.extra_node :
            return list(self.nodes) + [self.extra_node]
        else :
            return self.nodes

    def isTrue(self) :
        return len(self.nodes) <= 1

    def isFalse(self) :
        return False

    def isActive(self) :
        return not self.isTrue() and not self.isFalse()

    def add(self, node, formula):
        if formula.has_evidence_values():
            # Propagate constraint: if one of the other nodes is True: this one is false
            for n in self.nodes:
                if formula.get_evidence_value(n) == formula.TRUE:
                    return formula.FALSE
            if formula.get_evidence_value(node) == formula.FALSE:
                return node
            elif formula.get_evidence_value(node) == formula.TRUE:
                for n in self.nodes:
                    formula.set_evidence_value(n, formula.FALSE)

            if formula.semiring:
                sr = formula.semiring
                w = formula.get_weight(node, sr)
                for n in self.nodes:
                    w = sr.plus(w, formula.get_weight(n, sr))
                if sr.is_one(w):
                    unknown = None
                    if formula.get_evidence_value(node) != formula.FALSE:
                        unknown = node
                    for n in self.nodes:
                        if formula.get_evidence_value(n) != formula.FALSE:
                            if unknown is not None:
                                unknown = None
                                break
                            else:
                                unknown = n
                    if unknown is not None:
                        formula.set_evidence_value(unknown, formula.TRUE)

        self.nodes.add(node)
        if len(self.nodes) > 1 and self.extra_node is None:
            # If there are two or more choices -> add extra choice node
            self.updateLogic(formula)
        return node

    def encodeCNF(self) :
        if self.isActive() :
            nodes = list(self.nodes) + [self.extra_node]
            lines = []
            for i,n in enumerate(nodes) :
                for m in nodes[i+1:] :
                    lines.append( (-n, -m ))    # mutually exclusive
            lines.append( nodes )   # pick one
            return lines
        else :
            return []

    def updateLogic(self, formula) :
        """Add extra information to the logic structure of the formula."""

        if self.isActive() :
            self.extra_node = formula.add_atom( ('%s_extra' % (self.group,)), True, None )
            # formula.addConstraintOnNode(self, self.extra_node)

    def updateWeights(self, weights, semiring) :
        """Update the weights of the logic formula accordingly."""
        if self.isActive() :
            s = semiring.zero()
            for n in self.nodes :
                pos, neg = weights.get(n, (semiring.one(), semiring.one()))
                weights[n] = (pos, semiring.one())
                s = semiring.plus(s, pos)
            complement = semiring.negate(s)
            weights[self.extra_node] = (complement, semiring.one())

    def copy( self, rename={} ) :
        result = ConstraintAD( self.group )
        result.nodes = set(rename.get(x,x) for x in self.nodes)
        result.extra_node = rename.get( self.extra_node, self.extra_node )
        return result




# Alternative cycle breaking below: loop formula's
#   ASSAT: Computing Answer Sets of A Logic Program By SAT Solvers
#   Fangzhen Lin and Yuting Zhao
#   AAAI'02
# Not in use:
#   - does not work with SDD
#   - added constraints can become extremely large

#@transform(LogicFormula, LogicDAG)
def breakCyclesConstraint(source, target) :
    relevant = [False] * (len(source)+1)
    cycles = {}
    for name, node, label in source.getNamesWithLabel() :
        if label != interm.LABEL_NAMED :
            for c_in in findCycles( source, node, [], relevant) :
                c_in = tuple(sorted(c_in))
                if c_in in cycles :
                    pass
                else :
                    cycles[c_in] = splitCycle(source, c_in)

    for c_in, c_out in cycles.items() :
        source.addConstraint(ConstraintLoop(c_in, c_out))
    return source

def splitCycle(src, loop) :
    cycle_free = []
    for n in loop :
        n = src.getNode(n)
        t = type(n).__name__
        if t == 'disj' :
            cycle_free += [ c for c in n.children if not c in loop ]
        elif t == 'conj' :
            pass
        else :
            raise Exception('?')
    return cycle_free

def findCycles( src, a, path, relevant=None ) :
    n = src.getNode(a)
    t = type(n).__name__
    if relevant != None : relevant[a] = True
    try :
        s = path.index(a)
        yield path[s:]
    except ValueError :
        if t == 'atom' :
            pass
        else :
            for c in n.children :
                for p in findCycles( src, c, path + [a], relevant ) :
                    yield p


class ConstraintLoop(Constraint) :
    """Loop breaking constraint."""

    def __init__(self, cycle_nodes, noncycle_nodes) :
        self.in_loop = cycle_nodes
        self.ex_loop = noncycle_nodes
        self.in_node = None

    def __str__(self) :
        return 'loop_break(%s, %s)' % (list(self.in_loop), list(self.ex_loop))

    def isTrue(self) :
        return False

    def isFalse(self) :
        return False

    def isActive(self) :
        return True

    def encodeCNF(self) :
        if self.isActive() :
            ex_loop = tuple(self.ex_loop)
            lines = []
            for m in self.in_loop :
                lines.append( ex_loop + (-m,) )
            return lines
        else :
            return []

    def updateWeights(self, weights, semiring) :
        """Update the weights of the logic formula accordingly."""
        pass

    def copy( self, rename={} ) :
        cycle_nodes = set(rename.get(x,x) for x in self.in_loop)
        noncycle_nodes = set(rename.get(x,x) for x in self.ex_loop)
        result = ConstraintLoop( cycle_nodes, noncycle_nodes )
        return result

class TrueConstraint(Constraint) :

    def __init__(self, node) :
        self.node = node

    def isActive(self) :
        return True

    def encodeCNF(self) :
        return [[self.node]]

    def copy(self, rename={}) :
        return TrueConstraint( rename.get(self.node, self.node) )

    # def updateWeights(self, weights, semiring) :
    #     weights[self.node] = (semiring.one(), semiring.zero())

    def __str__(self) :
        return '%s is true' % self.node

def copyFormula(source, target) :
    for i, n, t in source :
        if t == 'atom' :
            target.add_atom( n.identifier, n.probability, n.group )
        elif t == 'conj' :
            target.add_and( n.children )
        elif t == 'disj' :
            target.add_or( n.children )
        else :
            raise TypeError("Unknown node type '%s'" % t)

    for name, node, label in source.get_names_with_label():
        target.add_name(name, node, label)

def breakCycles_lp(source, target=None) :

    if target != None :
        copyFormula(source,target)
    else :
        target = source

    tmp_file = tempfile.mkstemp('.lp')[1]
    with open(tmp_file, 'w') as f :
        lf_to_smodels(target, f)
    output = subprocess.check_output(['lp2acyc', tmp_file])
    smodels_to_lf( target, output )

    try :
        os.remove(tmp_file)
    except OSError :
        pass

    return target



def expand_node( formula, node_id ) :
    """Expand conjunctions by their body until a disjunction or atom is encountered.
    This method assumes that all cycles go through a disjunctive node.
    """
    node = formula.getNode(abs(node_id))
    nodetype = type(node).__name__
    conjuncts = []
    if nodetype == 'disj' :
        return [node_id]
    elif nodetype == 'atom' :
        return [node_id]
    elif node_id < 0 :
        return [node_id]
    else : # conj
        for c in node.children :
            conjuncts += expand_node(formula,c)
        return conjuncts

def lf_to_smodels( formula, out ) :

    # '1' is an internal atom => false

    #print (formula)
    # Write rules
    # Basic rule:
    #   1 head #lits #neglits [ body literals with negative first ]
    for i,n,t in formula :
        if t == 'disj' :
            for c in n.children :
                body = expand_node(formula, c)
                l = len(body)
                nl = len([ b for b in body if b < 0 ])
                body = [ abs(b)+1 for b in sorted(body) ]
                print('1 %s %s %s %s' % (i+1, l, nl, ' '.join(map(str,sorted(body))) ), file=out)
    print (0, file=out)

    for i,n,t in formula :
        if t == 'atom' or t == 'disj' :
            print (i+1, i, file=out)

    # Symbol table => must contain all (otherwise hidden in output)
    # Facts and disjunctions
    #   2 a
    #   3 b

    print (0, file=out)
    print ('B+', file=out)
    # B+  positive evidence?

    print (0, file=out)
    print ('B-', file=out)
    # B-  negative evidence?

    print (0, file=out)

    # Number of models
    print (1, file=out)

def smodels_to_lf( formula, acyclic ) :

    section = 0
    rules = defaultdict(list)
    data = [[]]
    for line in acyclic.split('\n') :
        if line == '0' :
            section += 1
            data.append([])
        else :
            data[-1].append( line )

    acyc_nodes = frozenset([ int(x.split()[0]) for x in data[1] if '_acyc_' in x ])
    given_nodes = frozenset([ int(x.split()[0]) for x in data[1] if not '_acyc_' in x ])
    if len(data[3]) > 1 :
        root_node = int(data[3][1])
    else :
        root_node = None

    for line in data[0] :
        line = line.split()
        line_head = int(line[1])
        line_neg = int(line[3])
        line_body_neg = frozenset(map(int,line[4:4+line_neg]))
        line_body_pos = frozenset(map(int,line[4+line_neg:]))

        # acyc_nodes are true
        if acyc_nodes & line_body_neg : continue
        # part of original program
        if line_head in given_nodes : continue
        # acyc_nodes are true
        line_body_pos -= acyc_nodes
        body = sorted([ -a for a in line_body_neg ] + list(line_body_pos))
        rules[line_head].append(body)

    translate = {}
    for head in rules :
        acyc_insert(formula, rules, head, given_nodes, translate)

    if root_node != None :
        formula.addConstraint(TrueConstraint(translate[root_node]))

    #print (formula)
    # print (translate[root_node])

def acyc_insert( formula, rules, head, given, translate ) :
    if head < 0 :
        f = -1
    else :
        f = 1

    if abs(head) in translate :
        return f * translate[abs(head)]
    elif abs(head) in given :
        return f * (abs(head)-1)
    else :
        disjuncts = []
        for body in rules[abs(head)] :
            new_body = [ acyc_insert(formula, rules, x, given, translate ) for x in body ]
            disjuncts.append(formula.add_and( new_body ))
        new_node = formula.add_or( disjuncts )
        translate[abs(head)] = new_node
        return f*new_node

import collections

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)
