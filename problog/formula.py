"""
problog.formula - Ground programs
---------------------------------

Data structures for propositional logic.

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

from collections import namedtuple, defaultdict, OrderedDict

from .core import ProbLogObject
from .errors import InconsistentEvidenceError

from .util import OrderedSet
from .logic import Term, Or, Clause, And, is_ground

from .evaluator import Evaluatable, FormulaEvaluator, FormulaEvaluatorNSP

from .constraint import ConstraintAD


class BaseFormula(ProbLogObject):
    """Defines a basic logic formula consisting of nodes in some logical relation.

        Each node is represented by a key. This key has the following properties:
         - None indicates false
         - 0 indicates true
         - a number larger than 0 indicates a positive node
         - the key -a with a a number larger than 0 indicates the negation of a

        This data structure also support weights on nodes, names on nodes and constraints.
    """

    # Define special keys
    TRUE = 0
    FALSE = None

    WEIGHT_NEUTRAL = True
    WEIGHT_NO = None

    LABEL_QUERY = "query"
    LABEL_EVIDENCE_POS = "evidence+"
    LABEL_EVIDENCE_NEG = "evidence-"
    LABEL_EVIDENCE_MAYBE = "evidence?"
    LABEL_NAMED = "named"

    def __init__(self):
        self._weights = {}               # Node weights: dict(key: Term)

        self._constraints = []           # Constraints: list of Constraint

        self._names = defaultdict(OrderedDict)  # Node names: dict(label: dict(key, Term))

        self._atomcount = 0

    @property
    def atomcount(self):
        """Number of atoms in the formula."""
        return self._atomcount

    # ====================================================================================== #
    # ==========                          NODE WEIGHTS                           =========== #
    # ====================================================================================== #

    def get_weights(self):
        """Get weights of the atoms in the formula.

        :return: dictionary of weights
        :rtype: dict[int, Term]
        """
        return self._weights

    def set_weights(self, weights):
        """Set weights of the atoms in the formula.

        :param weights: dictionary of weights
        :type weights: dict[int, Term]
        """
        self._weights = weights

    def get_weight(self, key, semiring):
        """Get actual value of the node with the given key according to the given semiring.

        :param key: key of the node (can be TRUE, FALSE or positive or negative)
        :param semiring: semiring to use to transform stored weight term into actual value
        :type semiring: problog.evaluator.Semiring
        :return: actual value of the weight of the given node
        """
        if self.is_false(key):
            return semiring.zero()
        elif self.is_true(key):
            return semiring.one()
        elif key < 0:
            return semiring.neg_value(self._weights[-key], key)
        else:
            return semiring.pos_value(self._weights[key], key)

    def extract_weights(self, semiring, weights=None):
        """Extracts the positive and negative weights for all atoms in the data structure.

        :param semiring: semiring that determines the interpretation of the weights
        :param weights: dictionary of { node name : weight } that overrides the builtin weights
        :returns: dictionary { key: (positive weight, negative weight) }
        :rtype: dict[int, tuple[any]]

        Atoms with weight set to neutral will get weight ``(semiring.one(), semiring.one())``.

        If the weights argument is given, it completely replaces the formula's weights.

        All constraints are applied to the weights.
        """

        if weights is None:
            weights = self.get_weights()
        else:
            oweights = dict(self.get_weights().items())
            oweights.update({self.get_node_by_name(n): v for n, v in weights.items()})
            weights = oweights

        result = {}
        for n, w in weights.items():
            if hasattr(self, 'get_name'):
                name = self.get_name(n)
            else:
                name = n
            if w == self.WEIGHT_NEUTRAL and type(self.WEIGHT_NEUTRAL) == type(w):
                result[n] = semiring.one(), semiring.one()
            else:
                result[n] = semiring.pos_value(w, name), semiring.neg_value(w, name)

        for c in self.constraints():
            c.update_weights(result, semiring)
        return result

    # ====================================================================================== #
    # ==========                           NODE NAMES                            =========== #
    # ====================================================================================== #

    def add_name(self, name, key, label=None):
        """Add a name to the given node.

        :param name: name of the node
        :type name: Term
        :param key: key of the node
        :type key: int | TRUE | FALSE
        :param label: type of label (one of LABEL_*)
        """
        if label is None:
            label = self.LABEL_NAMED
        self._names[label][name] = key

    def get_node_by_name(self, name):
        """Get node corresponding to the given name.

        :param name: name of the node to find
        :return: key of the node
        :raises: KeyError if no node with the given name was found
        """
        for names in self._names.values():
            res = names.get(name, '#NOTFOUND#')
            if res != '#NOTFOUND#':
                return res
        raise KeyError(name)

    # def get_name(self, key):
    #     names = self.get_names()
    #     print (names)

    def add_query(self, name, key):
        """Add a query name.

        Same as ``add_name(name, key, self.LABEL_QUERY)``.

        :param name: name of the query
        :param key: key of the query node
        """
        self.add_name(name, key, self.LABEL_QUERY)

    def add_evidence(self, name, key, value):
        """Add an evidence name.

        Same as ``add_name(name, key, self.LABEL_EVIDENCE_???)``.

        :param name: name of the query
        :param key: key of the query node
        :param value: value of the evidence (True, False or None)
        """
        if value is None:
            self.add_name(name, key, self.LABEL_EVIDENCE_MAYBE)
        elif value:
            self.add_name(name, key, self.LABEL_EVIDENCE_POS)
        else:
            self.add_name(name, key, self.LABEL_EVIDENCE_NEG)

    def clear_evidence(self):
        """Remove all evidence."""
        self._names[self.LABEL_EVIDENCE_MAYBE] = {}
        self._names[self.LABEL_EVIDENCE_POS] = {}
        self._names[self.LABEL_EVIDENCE_NEG] = {}

    def clear_queries(self):
        """Remove all evidence."""
        self._names[self.LABEL_QUERY] = {}

    def clear_labeled(self, label):
        """Remove all evidence."""
        self._names[label] = {}

    def get_names(self, label=None):
        """Get a list of all node names in the formula.

        :param label: restrict to given label. If not set, all nodes are returned.
        :return: list of all nodes names (of the requested type) as a list of tuples (name, key)
        """
        if label is None:
            result = OrderedSet()
            for names in self._names.values():
                for name, node in names.items():
                    result.add((name, node))
            return result
        else:
            return self._names.get(label, {}).items()

    def get_names_with_label(self):
        """Get a list of all node names in the formula with their label type.

        :return: list of all nodes names with their type
        """
        result = []
        for label in self._names:
            for name, key in self._names[label].items():
                result.append((name, key, label))
        return result

    def queries(self):
        """Get a list of all queries.

        :return: ``get_names(LABEL_QUERY)``
        """
        return self.get_names(self.LABEL_QUERY)

    def labeled(self):
        """Get a list of all query-like labels.

        :return:
        """
        result = []
        for name, node, label in self.get_names_with_label():
            if label not in (self.LABEL_NAMED, self.LABEL_EVIDENCE_POS, self.LABEL_EVIDENCE_NEG, self.LABEL_EVIDENCE_MAYBE):
                result.append((name, node, label))
        return result

    def evidence(self):
        """Get a list of all determined evidence.
        Keys are negated for negative evidence.
        Unspecified evidence (value None) is not included.

        :return: list of tuples (name, key) for positive and negative evidence
        """
        evidence_true = self.get_names(self.LABEL_EVIDENCE_POS)
        evidence_false = self.get_names(self.LABEL_EVIDENCE_NEG)
        return list(evidence_true) + [(name, self.negate(node)) for name, node in evidence_false]

    def evidence_all(self):
        """Get a list of all evidence (including undetermined).

        :return: list of tuples (name, key, value) where value can be -1, 0 or 1
        """
        evidence_true = [x + (1,) for x in self.get_names(self.LABEL_EVIDENCE_POS)]
        evidence_false = [x + (-1,) for x in self.get_names(self.LABEL_EVIDENCE_NEG)]
        evidence_maybe = [x + (0,) for x in self.get_names(self.LABEL_EVIDENCE_MAYBE)]
        return evidence_true + evidence_false + evidence_maybe

    # ====================================================================================== #
    # ==========                        KEY MANIPULATION                         =========== #
    # ====================================================================================== #

    def is_true(self, key):
        """Does the key represent deterministic True?

        :param key: key
        :return: ``key == self.TRUE``
        """
        return key == self.TRUE

    def is_false(self, key):
        """Does the key represent deterministic False?

        :param key: key
        :return: ``key == self.FALSE``
        """

        return key == self.FALSE

    def is_probabilistic(self, key):
        """Does the key represent a probabilistic node?

        :param key: key
        :return: ``not is_true(key) and not is_false(key)``
        """
        return not self.is_true(key) and not self.is_false(key)

    def negate(self, key):
        """Negate the key.

        For TRUE, returns FALSE;
        For FALSE, returns TRUE;
        For x returns -x

        :param key: key to negate
        :return: negation of the key
        """
        if key == self.TRUE:
            return self.FALSE
        elif key == self.FALSE:
            return self.TRUE
        else:
            return -key

    # ====================================================================================== #
    # ==========                          CONSTRAINTS                            =========== #
    # ====================================================================================== #

    def constraints(self):
        """Return the list of constraints.

        :return: list of constraints
        """
        return self._constraints

    def add_constraint(self, constraint):
        """Add a constraint

        :param constraint: constraint to add
        :type constraint: problog.constraint.Constraint
        """
        self._constraints.append(constraint)

    def flag(self, flag):
        flag = '_%s' % flag
        return hasattr(self, flag) and getattr(self, flag)


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

    _atom = namedtuple('atom', ('identifier', 'probability', 'group', 'name', 'source'))
    _conj = namedtuple('conj', ('children', 'name'))
    _disj = namedtuple('disj', ('children', 'name'))
    # negation is encoded by using a negative number for the key

    def _create_atom(self, identifier, probability, group, name=None, source=None):
        return self._atom(identifier, probability, group, name, source)

    def _create_conj(self, children, name=None):
        return self._conj(children, name)

    def _create_disj(self, children, name=None):
        return self._disj(children, name)

    # noinspection PyUnusedLocal
    def __init__(self, auto_compact=True, avoid_name_clash=False, keep_order=False,
                 use_string_names=False, keep_all=False, propagate_weights=None,
                 max_arity=0, keep_duplicates=False, keep_builtins=False, hide_builtins=False,
                 **kwdargs):
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
        self._keep_builtins = (keep_all or keep_builtins) and not hide_builtins
        self._keep_duplicates = keep_duplicates

        self._max_arity = max_arity

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
            ntype = type(node).__name__
            if key < 0:
                lname = -name
            else:
                lname = name
            if ntype == 'atom':
                node = type(node)(*(node[:-2] + (lname, node[-1])))
            else:
                node = type(node)(*(node[:-1] + (lname,)))
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

    # noinspection PyUnusedLocal
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
        """Replace the node with the given content.

        :param key: key of the node to replace.
        :type key: int > 0
        :param value: new content of the node
        """
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

    def add_atom(self, identifier, probability, group=None, name=None, source=None):
        """Add an atom to the formula.

        :param identifier: a unique identifier for the atom
        :param probability: probability of the atom
        :param group: a group identifier that identifies mutually exclusive atoms (or None if no \
        constraint)
        :param name: name of the new node
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
            atom = self._create_atom(identifier, probability, group, name, source)
            node_id = self._add(atom, key=identifier)

            self.get_weights()[node_id] = probability
            if name is not None:
                self.add_name(name, node_id, self.LABEL_NAMED)
            if node_id == len(self._nodes):
                # The node was not reused?
                self._atomcount += 1
                # TODO if the next call return 0 or None, the node is still added?
                node_id = self._add_constraint_me(group, node_id)
            return node_id

    def add_and(self, components, key=None, name=None):
        """Add a conjunction to the logic formula.

        :param components: a list of node identifiers that already exist in the logic formula.
        :param key: preferred key to use
        :param name: name of the node
        :returns: the key of the node in the formula (returns 0 for deterministic atoms)
        """
        return self._add_compound('conj', components, self.FALSE, self.TRUE, key=key, name=name)

    def add_or(self, components, key=None, readonly=True, name=None):
        """Add a disjunction to the logic formula.

        :param components: a list of node identifiers that already exist in the logic formula.
        :param key: preferred key to use
        :param readonly: indicates whether the node should be modifiable. This will allow \
        additional disjunct to be added without changing the node key. Modifiable nodes are \
        less optimizable.
        :param name: name of the node
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
            elif component in node.children and not self._keep_duplicates:
                pass    # already there
            else:
                if 0 < self._max_arity == len(node.children):
                    child = self.add_or(node.children)
                    return self._update(key, self._create_disj((child, component), name=node.name))
                else:
                    return self._update(key, self._create_disj(node.children + (component,),
                                                               name=node.name))
            return key

    def add_not(self, component):
        """Returns the key to the negation of the node.

        :param component: the node to negate
        """
        return self.negate(component)

    def get_node(self, key):
        """Get the content of the node with the given key.

        :param key: key of the node
        :type key: int > 0
        :return: content of the node
        """
        assert self.is_probabilistic(key)
        assert key > 0
        return self._nodes[key - 1]

    # noinspection PyUnusedLocal
    def _add_compound(self, nodetype, content, t, f, key=None,
                      readonly=True, update=None, name=None):
        """Add a compound term (AND or OR)."""
        assert content   # Content should not be empty

        name_clash = False
        if self._auto_compact:
            # If there is a t node, (true for OR, false for AND)
            if t in content:
                return t

            # Eliminate unneeded node nodes (false for OR, true for AND)
            content = filter(lambda x: x != f, content)

            # Put into fixed order and eliminate duplicate nodes
            if self._keep_duplicates:
                content = tuple(content)
            elif self._keep_order:
                content = tuple(OrderedSet(content))
            else:  # any_order
                # can also merge (a, b) and (b, a)
                content = tuple(set(content))

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
                        if name is not None:
                            self.add_name(name, content[0], self.LABEL_NAMED)
                        return content[0]
                    else:
                        name_clash = True
                else:
                    if name is not None:
                        name_old = self.get_node(abs(content[0])).name
                        if name_old is None:
                            self.add_name(name, content[0], self.LABEL_NAMED)

                    return content[0]
        else:
            content = tuple(content)

        if nodetype == 'conj':
            node = self._create_conj(content, name)
            return self._add(node, reuse=self._auto_compact and not self._keep_all)
        elif nodetype == 'disj':
            node = self._create_disj(content, name)
            if update is not None:
                # If an update key is set, update that node
                return self._update(update, node)
            elif readonly:
                # If the node is readonly, we can try to reuse an existing node.
                new_node = self._add(node, reuse=self._auto_compact and not name_clash and not self._keep_all)
                return new_node
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
            yield (i + 1, n, type(n).__name__)

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

    def has_evidence_values(self):
        """Checks whether the current formula contains information for evidence propagation."""
        return hasattr(self, 'lookup_evidence')

    def get_evidence_values(self):
        """Retrieves evidence propagation information."""
        assert(self.has_evidence_values())
        return getattr(self, 'lookup_evidence')

    def get_evidence_value(self, key):
        """Get value of the given node based on evidence propagation.

        :param key: key of the node
        :return: value of the node (key, TRUE or FALSE)
        """
        if key == 0 or key is None:
            return key
        elif self.has_evidence_values():
            result = self.get_evidence_values().get(abs(key), abs(key))
            if key < 0:
                return self.negate(result)
            else:
                return result
        else:
            return key

    def set_evidence_value(self, key, value):
        """Set value of the given node based on evidence propagation.

        :param key: key of the node
        :param value: value of the node
        """
        if key < 0:
            self.get_evidence_values()[-key] = self.negate(value)
        else:
            self.get_evidence_values()[key] = value

    def propagate(self, nodeids, current=None):
        """Propagate the value of the given node
          (true if node is positive, false if node is negative)
        The propagation algorithm is not complete.

        :param nodeids: evidence nodes to set (> 0 means true, < 0 means false)
        :param current: current set of nodes with deterministic value
        :return: dictionary of nodes with deterministic value
        """

        # Initialize current in case nothing is known yet.
        if current is None:
            current = {}

        # Provide easy access to values
        values = {True: self.TRUE, False: self.FALSE}

        # Reverse mapping between a node and its parents
        atoms_in_rules = defaultdict(set)

        # Queue of nodes that need to be handled.
        # INVARIANT: elements in queue have deterministic value
        # INVARIANT: elements in queue are not yet listed in current
        queue = set(nodeids)

        while queue:
            nid = queue.pop()

            # Get information about the node
            n = self.get_node(abs(nid))
            t = type(n).__name__

            if abs(nid) not in current:
                # This is the first time we process this node.
                # We should process its parents again.
                for at in atoms_in_rules[abs(nid)]:
                    if at in current:
                        # Parent has a truth value.
                        # Try to propagate parent again.
                        if current[abs(at)] == self.TRUE:
                            queue.add(abs(at))
                        else:
                            queue.add(-abs(at))

            # Record node value in current
            current[abs(nid)] = values[nid > 0]

            # Process node and propagate to children
            if t == 'atom':
                # Nothing to do.
                pass
            else:
                # Get the list of children with their actual (propagated) values.
                children = []
                for c in n.children:
                    ch = current.get(abs(c), abs(c))
                    if c < 0:
                        ch = self.negate(ch)
                    children.append(ch)

                # Handle trivial cases:
                # Node should be true, but is a conjunction with a false child
                if t == 'conj' and self.FALSE in children and nid > 0:
                    raise InconsistentEvidenceError()
                # Node should be false, but is a disjunction with a true child
                elif t == 'disj' and self.TRUE in children and nid < 0:
                    raise InconsistentEvidenceError()
                # Node should be false, and is a conjunction with a false child
                elif t == 'conj' and self.FALSE in children and nid < 0:
                    # Already satisfied, nothing else to do
                    pass
                # Node should be true and is a disjunction with a true child
                elif t == 'disj' and self.TRUE in children and nid > 0:
                    # Already satisfied, nothing else to do
                    pass
                else:
                    # Filter out deterministic children
                    children = list(filter(lambda x: x != 0 and x is not None, children))
                    if len(children) == 1:
                        # One child left: propagate value to the child
                        if abs(children[0]) not in current:
                            if nid < 0:
                                queue.add(-children[0])
                            else:
                                queue.add(children[0])
                            atoms_in_rules[abs(children[0])].discard(abs(nid))
                    elif nid > 0 and t == 'conj':
                        # Conjunction is true => all children are true
                        for c in children:
                            if abs(c) not in current:
                                queue.add(c)
                            atoms_in_rules[abs(c)].discard(abs(nid))
                    elif nid < 0 and t == 'disj':
                        # Disjunction is false => all children are false
                        for c in children:
                            if abs(c) not in current:
                                queue.add(-c)
                            atoms_in_rules[abs(c)].discard(abs(nid))
                    else:
                        # We can't propagate yet. Mark current rule as parent of its children.
                        for c in children:
                            atoms_in_rules[abs(c)].add(abs(nid))
        return current

    # def propagate(self, nodeids, current=None):
    #     if current is None:
    #         current = {}
    #
    #     values = {True: self.TRUE, False: self.FALSE}
    #     atoms_in_rules = defaultdict(set)
    #
    #     updated = set()
    #     queue = set(nodeids)
    #     while queue:
    #         nid = queue.pop()
    #
    #         if abs(nid) not in current:
    #             updated.add(abs(nid))
    #             for at in atoms_in_rules[abs(nid)]:
    #                 if at in current:
    #                     if current[abs(at)] == 0:
    #                         queue.add(abs(at))
    #                     else:
    #                         queue.add(-abs(at))
    #             current[abs(nid)] = values[nid > 0]
    #
    #         n = self.get_node(abs(nid))
    #         t = type(n).__name__
    #         if t == 'atom':
    #             pass
    #         else:
    #             children = []
    #             for c in n.children:
    #                 ch = current.get(abs(c), abs(c))
    #                 if c < 0:
    #                     ch = self.negate(ch)
    #                 children.append(ch)
    #             if t == 'conj' and None in children and nid > 0:
    #                 raise InconsistentEvidenceError()
    #             elif t == 'disj' and 0 in children and nid < 0:
    #                 raise InconsistentEvidenceError()
    #             children = list(filter(lambda x: x != 0 and x is not None, children))
    #             if len(children) == 1:  # only one child
    #                 if abs(children[0]) not in current:
    #                     if nid < 0:
    #                         queue.add(-children[0])
    #                     else:
    #                         queue.add(children[0])
    #                     atoms_in_rules[abs(children[0])].discard(abs(nid))
    #             elif nid > 0 and t == 'conj':
    #                 # Conjunction is true
    #                 for c in children:
    #                     if abs(c) not in current:
    #                         queue.add(c)
    #                     atoms_in_rules[abs(c)].discard(abs(nid))
    #             elif nid < 0 and t == 'disj':
    #                 # Disjunction is false
    #                 for c in children:
    #                     if abs(c) not in current:
    #                         queue.add(-c)
    #             else:
    #                 for c in children:
    #                     atoms_in_rules[abs(c)].add(abs(nid))
    #     return current

    # ====================================================================================== #
    # ==========                        EXPORT TO STRING                         =========== #
    # ====================================================================================== #

    def __str__(self):
        s = '\n'.join('%s: %s' % (i, n) for i, n, t in self)
        f = True
        for q in self.labeled():
            if f:
                f = False
                s += '\nQueries : '
            s += '\n* %s : %s [%s]' % q

        f = True
        for q in self.evidence():
            if f:
                f = False
                s += '\nEvidence : '
            s += '\n* %s : %s' % q

        f = True
        for c in self.constraints():
            if c.is_nontrivial():
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

            pl = problog.program.PrologFile(input_file)
            problog.formula.LogicFormula.create_from(avoid_name_clash=True, keep_order=True, \
label_all=True)
            prologfile = gp.to_prolog()

        :return: Prolog program
        :rtype: str
        """

        lines = ['%s.' % c for c in self.enum_clauses()]

        for qn, qi in self.queries():
            if is_ground(qn):
                if qi == self.TRUE:
                    lines.append('%s.' % qn)
                elif qi == self.FALSE:
                    lines.append('%s :- fail.' % qn)
                lines.append('query(%s).' % qn)

        for qn, qi in self.evidence():
            if qi < 0:
                lines.append('evidence(%s).' % -qn)
            else:
                lines.append('evidence(%s).' % qn)

        return '\n'.join(lines)

        # lines = []
        # neg_heads = set()
        # for head, body in self.enumerate_clauses():
        #     head_name = self.get_name(head)
        #     if head_name.is_negated():
        #         pos_name = -head_name
        #         head_name = Term(pos_name.functor + '_aux', *pos_name.args)
        #         if head not in neg_heads:
        #             lines.append('%s :- %s.' % (pos_name, -head_name))
        #             neg_heads.add(head)
        #     if body:    # clause with a body
        #         body = ', '.join(map(str, map(self.get_name, body)))
        #         lines.append('%s :- %s.' % (head_name, body))
        #     else:   # fact
        #         prob = self.get_node(head).probability
        #
        #         if prob is not None:
        #             lines.append('%s::%s.' % (prob, head_name))
        #         else:
        #             lines.append('%s.' % head_name)
        # return '\n'.join(lines)

    def get_name(self, key):
        """Get the name of the given node.

        :param key: key of the node
        :return: name of the node
        :rtype: Term
        """
        if key == 0:
            return Term('true')
        elif key is None:
            return Term('false')
        else:
            node = self.get_node(abs(key))
            name = node.name

            if not self._is_valid_name(name) and type(node).__name__ == 'disj' and node.children:
                if key < 0:
                    name = self.get_name(-node.children[0])
                else:
                    name = self.get_name(node.children[0])

            if name is None:
                name = Term('node_%s' % abs(key))
            if key < 0:
                return -name
            else:
                return name

    def enumerate_clauses(self, relevant_only=True):
        """Enumerate the clauses of this logic formula.
            Clauses are represented as (``head, [body]``).

        :param relevant_only: only list clauses that are part of the ground program for a query or \
         evidence
        :return: iterator of clauses
        """
        enumerated = OrderedSet()

        if relevant_only:
            to_enumerate = OrderedSet(abs(n) for q, n in self.queries() if self.is_probabilistic(n))
            to_enumerate |= OrderedSet(abs(n)
                                       for q, n in self.evidence() if self.is_probabilistic(n))
        else:
            to_enumerate = OrderedSet(range(1, len(self) + 1))

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

    def extract_ads(self, relevant, processed):

        # Collect information about annotated disjunctions
        choices = set([])
        choice_group = {}
        choice_by_group = defaultdict(list)
        choice_parent = {}
        choice_by_parent = {}
        choice_body = {}
        choice_name = {}
        choice_prob = {}

        for i, n, t in self:
            if not relevant[i]:
                continue
            if t == 'atom' and n.group is not None:
                choice_group[i] = n.group
                choice_prob[i] = n.probability
                choice_by_group[n.group].append(i)
                choices.add(i)
                processed[i] = True
            elif t == 'conj' and n.children[-1] in choices:
                choice = n.children[-1]
                choice_parent[choice_group[choice]] = i
                choice_by_parent[i] = choice
                choice_body[choice_group[choice]] = n.children[0]
                if n.name is not None:
                    choice_name[choice] = n.name
                processed[i] = True
                if not self._is_valid_name(self.get_node(abs(n.children[0])).name):
                    processed[n.children[0]] = True

        for i, n, t in self:
            if t == 'disj':
                overlap = set(n.children) & set(choice_by_parent.keys()) | set(n.children) & set(choices)
                for o in overlap:
                    p = choice_by_parent.get(o, o)
                    if self._is_valid_name(n.name) and not self._is_valid_name(choice_name.get(p)):
                        choice_name[p] = n.name

        for group, choices in choice_by_group.items():
            # Construct head

            head = Or.from_list([choice_name[c].with_probability(choice_prob[c]) for c in choices])

            # Construct body
            body = self.get_body(choice_body.get(group, self.TRUE), processed)

            if body is None:
                yield head
            else:
                yield (Clause(head, body))

    def _is_valid_name(self, name):
        return name is not None and \
            not name.functor.startswith('_problog_') and \
            not name.functor == 'choice' and not name.functor == 'body'

    def get_body(self, index, processed=None, parent_name=None):
        if index == self.TRUE:
            return None
        else:
            node = self.get_node(abs(index))
            ntype = type(node).__name__
            if self._is_valid_name(node.name) and str(node.name) != str(parent_name):
                if index < 0:
                    return -node.name
                else:
                    return node.name
            elif ntype == 'atom':
                # Easy case: atom
                if index < 0:
                    return -node.name
                else:
                    return node.name
            elif ntype == 'conj':
                if index < 0:
                    return -node.name
                else:
                    children = self._unroll_conj(node)
                    return And.from_list(list(map(self.get_name, children)))
            elif ntype == 'disj' and len(node.children) == 1 and not self._is_valid_name(node.name):
                if processed:
                    processed[abs(index)] = True
                if index < 0:
                    b = self.get_body(-node.children[0], parent_name=parent_name)
                else:
                    b = self.get_body(node.children[0], parent_name=parent_name)
                return b
            elif ntype == 'disj':
                if index < 0:
                    return -node.name
                else:
                    return node.name
            else:
                print (self)
                print (index, node)
                raise Exception('Unexpected')

    def enum_clauses(self):
        relevant = self.extract_relevant()
        processed = [False] * (len(self) + 1)
        for ad in self.extract_ads(relevant, processed):
            yield ad
        for i, n, t in self:
            if relevant[i] and not processed[i]:
                if t == 'atom':
                    if n.name is not None and n.source not in ('builtin', 'negation'):
                        yield n.name.with_probability(n.probability)
                elif t == 'disj':
                    for c in n.children:
                        if not processed[abs(c)] or self._is_valid_name(self.get_node(abs(c)).name):
                            b = self.get_body(c, parent_name=n.name)
                            if str(n.name) != str(b):   # TODO bit of a hack?
                                yield Clause(n.name, b)
                elif t == 'conj' and n.name is None:
                    pass
                else:
                    yield Clause(n.name, self.get_body(i, parent_name=n.name))

    def extract_relevant(self):
        relevant = [False] * (len(self)+1)
        roots = set(abs(n) for q, n in self.queries() if self.is_probabilistic(n))
        roots |= set(abs(n) for q, n in self.evidence() if self.is_probabilistic(n))
        while roots:
            root = roots.pop()
            if not relevant[root]:
                relevant[root] = True
                node = self.get_node(root)
                ntype = type(node).__name__
                if ntype != 'atom':
                    for c in node.children:
                        if not relevant[abs(c)]:
                            roots.add(abs(c))
        return relevant

    def get_node_multiplicity(self, index):
        if self.is_true(index):
            return 1
        elif self.is_false(index):
            return 0
        else:
            node = self.get_node(abs(index))
            ntype = type(node).__name__
            if ntype == 'atom':
                return 1
            elif index < 0:
                # TODO verify this is correct: negative node has multiplicity 1
                return 1
            else:
                child_multiplicities = [self.get_node_multiplicity(c) for c in node.children]
                if ntype == 'disj':
                    return sum(child_multiplicities)
                else:
                    r = 1
                    for cm in child_multiplicities:
                        r *= cm
                    return r

    def enumerate_branches(self, index):
        if self.is_true(index):
            yield 0, [self.TRUE]
        elif self.is_false(index):
            yield 0, []
        elif index < 0:
            yield index, [index]
        else:
            node = self.get_node(index)
            ntype = type(node).__name__
            if ntype == 'atom':
                yield index, [index]
            elif ntype == 'conj':
                from itertools import product, chain
                for b in product(*(self.enumerate_branches(c) for c in node.children)):
                    c_max, c_br = zip(*b)
                    mx = max(c_max)
                    yield max(index, mx), chain(*c_br)
            else:
                for c in node.children:
                    for mx, b in self.enumerate_branches(c):
                        yield mx, b

    def copy_node(self, target, index):
        if self.is_true(index):
            return target.TRUE
        elif self.is_false(index):
            return target.FALSE
        else:
            node = self.get_node(abs(index))
            ntype = type(node).__name__
            sign = 1 if index > 0 else -1
            if ntype == 'atom':
                return sign * target.add_atom(*node)
            elif ntype == 'conj':
                children = [self.copy_node(target, c) for c in node.children]
                return sign * target.add_and(children)
            elif ntype == 'disj':
                children = [self.copy_node(target, c) for c in node.children]
                return sign * target.add_or(children)

    def _unroll_conj(self, node):
        assert type(node).__name__ == 'conj'

        if len(node.children) == 1:
            return node.children
        elif len(node.children) == 2 and node.children[1] > 0:
            children = [node.children[0]]
            current = node.children[1]
            current_node = self.get_node(current)
            while type(current_node).__name__ == 'conj' and len(current_node.children) == 2 and \
                    not self._is_valid_name(current_node.name):
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
            if name.is_negated():
                pos_name = -name
                name = Term(pos_name.functor + '_aux', *pos_name.args)
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

    def clone(self, destination):
        source = self
        # TODO maintain a translation table
        for i, n, t in source:
            if t == 'atom':
                j = destination.add_atom(n.identifier, n.probability, n.group, source.get_name(i))
            elif t == 'conj':
                j = destination.add_and(n.children, source.get_name(i))
            elif t == 'disj':
                j = destination.add_or(n.children, source.get_name(i))
            else:
                raise TypeError('Unknown node type')
            assert i == j

        for name, node, label in source.get_names_with_label():
            destination.add_name(name, node, label)

        for c in source.constraints():
            if c.is_nontrivial():
                destination.add_constraint(c)

        return destination


class LogicDAG(LogicFormula, Evaluatable):
    """A propositional logic formula without cycles."""

    def __init__(self, auto_compact=True, **kwdargs):
        LogicFormula.__init__(self, auto_compact, **kwdargs)

    def _create_evaluator(self, semiring, weights, **kwargs):
        if semiring.is_nsp():
            return FormulaEvaluatorNSP(self, semiring, weights)
        else:
            return FormulaEvaluator(self, semiring, weights)


class DeterministicLogicFormula(LogicFormula):
    """A deterministic logic formula."""

    def __init__(self, **kwdargs):
        LogicFormula.__init__(self, **kwdargs)

    def add_atom(self, identifier, probability, group=None, name=None, source=None):
        return self.TRUE


#
#
# class StringKeyLogicFormula(LogicFormula) :
#     """A propositional logic formula consisting of and, or, not and atoms."""
#
#     TRUE = 'true'
#     FALSE = 'false'
#
#     def __init__(self) :
#         LogicFormula.__init__(self)
#
#         self.__nodes = defaultdict(list)
#
#         self.__constraints_me = {}
#         self.__constraints = []
#
#     def _add( self, node, key=None, reuse=True ) :
#         """Adds a new node, or reuses an existing one.
#
#         :param node: node to add
#         :param reuse: (default True) attempt to map the new node onto an existing one \
#         based on its content
#
#         """
#         self.__nodes[key].append(node)
#         return key
#
#     def _update( self, key, value ) :
#         """Replace the node with the given node."""
#         self.__nodes[ key ] = [value]
#
#     def add_not( self, component ) :
#         """Returns the key to the negation of the node."""
#         if self.isTrue(component) :
#             return self.FALSE
#         elif self.isFalse(component) :
#             return self.TRUE
#         elif component.startswith('-') :
#             return component[1:]
#         else :
#             return '-' + component
#
#     def _addCompound(self, nodetype, content, t, f, key=None, readonly=True, update=None) :
#         """Add a compound term (AND or OR)."""
#         assert( content )   # Content should not be empty
#
#         # #If there is a t node, (true for OR, false for AND)
#         # if t in content : return t
#         #
#         # # Eliminate unneeded node nodes (false for OR, true for AND)
#         # content = filter( lambda x : x != f, content )
#         #
#         # # Put into fixed order and eliminate duplicate nodes
#         # content = tuple(sorted(set(content)))
#         #
#         # # Empty OR node fails, AND node is true
#         # if not content : return f
#
#         # # Contains opposites: return 'TRUE' for or, 'FALSE' for and
#         # if len(set(content)) > len(set(map(abs,content))) : return t
#
#         # If node has only one child, just return the child.
#         # Don't do this for modifiable nodes, we need to keep a separate node.
#         if len(content) == 1 : return self._add(content[0], key=key)
#
#         content = tuple(content)
#
#         if nodetype == 'conj' :
#             node = self._create_conj( content )
#             return self._add( node, key=key )
#         elif nodetype == 'disj' :
#             node = self._create_disj( content )
#             if update != None :
#                 # If an update key is set, update that node
#                 return self._update( update, node )
#             elif readonly :
#                 # If the node is readonly, we can try to reuse an existing node.
#                 return self._add( node, key=key )
#             else :
#                 # If node is modifiable, we shouldn't reuse an existing node.
#                 return self._add( node, key=key, reuse=False )
#         else :
#             raise TypeError("Unexpected node type: '%s'." % nodetype)
#
#     def _resolve(self, key) :
#         if type(key) == str :
#             return key
#
#         res = self.__nodes.get(key,key)
#         if type(res) == list :
#             assert(len(res) == 1)
#             return res[0]
#         else :
#             return res
#
#     def _deref(self, x) :
#         c = x
#         neg = 0
#         while type(c) == str :
#             if c[0] == '-' :
#                 c = c[1:]
#                 neg += 1
#             x = c
#             if not c in self.__nodes : break
#             nn = self.__nodes[c]
#             if len(nn) == 1 :
#                 c = nn[0]
#             else :
#                 break
#         if neg % 2 == 0 :
#             return x
#         else :
#             return '-' + x
#
#     def __iter__(self) :
#         for k in self.__nodes :
#             n = self.__nodes[k]
#             child_names = []
#             children = []
#             for x in n :
#                 if type(x) == str :
#                     x = self._deref(x)
#                     child_names.append(x)
#                 else :
#                     key = '%s_%s' % (k,len(child_names) )
#                     child_names.append( key )
#                     if type(x).__name__ != 'atom' :
#                         x_children = [self._deref(y) for y in x.children]
#                         children.append( (key, self._create_conj(x_children) ) )
#                     else :
#                         children.append( (key, x ))
#             if len(child_names) > 1 :
#                 yield (k, self._create_disj(child_names), 'disj')
#                 for i,c in children :
#                     yield (i, c, type(c).__name__)
#             else :
#                 for i,c in children :
#                     yield (k, c, type(c).__name__)
#
#     def __len__(self) :
#         return len(self.__nodes)
#
#     ##################################################################################
#     ####                            OUTPUT GENERATION                             ####
#     ##################################################################################
#
#     def toLogicFormula(self) :
#         target = LogicFormula(auto_compact=False)
#         translate = {}
#         i = 0
#         for k,n,t in self :
#             i += 1
#             translate[k] = i
#             translate['-' + str(k) ] = -i
#         for k,n,t in self :
#             if t == 'atom' :
#                 i = target.add_atom( n.identifier, n.probability, n.group )
#             elif t == 'disj' :
#                 i = target.add_or( [ translate[x] for x in n.children ] )
#             elif t == 'conj' :
#                 i = target.add_and( [ translate[x] for x in n.children ] )
#             assert(i == translate[k])
#
#         for name, key, label in self.get_names_with_label():
#             key = self._deref(key)
#             target.add_name(name, translate[key], label)
#
#         return target
#
#     @classmethod
#     def loadFrom(cls, lp) :
#         interm = StringKeyLogicFormula()
#         for c in lp :
#             if type(c).__name__ == 'Clause' :
#                 key = str(c.head)
#                 body = []
#                 current = c.body
#                 while type(current).__name__ == 'And' :
#                     if type(current.op1).__name__ == 'Not' :
#                         body.append('-' + str(current.op1.child))
#                     else :
#                         body.append(str(current.op1))
#                     current = current.op2
#                 if type(current).__name__ == 'Not' :
#                     body.append('-' + str(current.child))
#                 else :
#                     body.append(str(current))
#                 interm.add_and( body, key=key )
#                 interm.add_name(key, key, interm.LABEL_NAMED)
#             elif type(c).__name__ == 'Term' :
#                 key = str(c.with_probability())
#                 interm.add_atom( key, c.probability, None )
#                 interm.add_name(key, key, interm.LABEL_NAMED)
#             else :
#                 raise Exception("Unexpected type: '%s'" % type(c).__name__)
#         return interm
#


# Alternative cycle breaking below: loop formula's
#   ASSAT: Computing Answer Sets of A Logic Program By SAT Solvers
#   Fangzhen Lin and Yuting Zhao
#   AAAI'02
# Not in use:
#   - does not work with SDD
#   - added constraints can become extremely large

# #@transform(LogicFormula, LogicDAG)
# def breakCyclesConstraint(source, target) :
#     relevant = [False] * (len(source)+1)
#     cycles = {}
#     for name, node, label in source.getNamesWithLabel() :
#         if label != interm.LABEL_NAMED :
#             for c_in in findCycles( source, node, [], relevant) :
#                 c_in = tuple(sorted(c_in))
#                 if c_in in cycles :
#                     pass
#                 else :
#                     cycles[c_in] = splitCycle(source, c_in)
#
#     for c_in, c_out in cycles.items() :
#         source.addConstraint(ConstraintLoop(c_in, c_out))
#     return source
#
# def splitCycle(src, loop) :
#     cycle_free = []
#     for n in loop :
#         n = src.get_node(n)
#         t = type(n).__name__
#         if t == 'disj' :
#             cycle_free += [ c for c in n.children if not c in loop ]
#         elif t == 'conj' :
#             pass
#         else :
#             raise Exception('?')
#     return cycle_free
#
# def findCycles( src, a, path, relevant=None ) :
#     n = src.get_node(a)
#     t = type(n).__name__
#     if relevant != None : relevant[a] = True
#     try :
#         s = path.index(a)
#         yield path[s:]
#     except ValueError :
#         if t == 'atom' :
#             pass
#         else :
#             for c in n.children :
#                 for p in findCycles( src, c, path + [a], relevant ) :
#                     yield p
#
#
# class ConstraintLoop(Constraint) :
#     """Loop breaking constraint."""
#
#     def __init__(self, cycle_nodes, noncycle_nodes) :
#         self.in_loop = cycle_nodes
#         self.ex_loop = noncycle_nodes
#         self.in_node = None
#
#     def __str__(self) :
#         return 'loop_break(%s, %s)' % (list(self.in_loop), list(self.ex_loop))
#
#     def isTrue(self) :
#         return False
#
#     def isFalse(self) :
#         return False
#
#     def is_nontrivial(self) :
#         return True
#
#     def as_clauses(self) :
#         if self.is_nontrivial() :
#             ex_loop = tuple(self.ex_loop)
#             lines = []
#             for m in self.in_loop :
#                 lines.append( ex_loop + (-m,) )
#             return lines
#         else :
#             return []
#
#     def updateWeights(self, weights, semiring) :
#         """Update the weights of the logic formula accordingly."""
#         pass
#
#     def copy( self, rename={} ) :
#         cycle_nodes = set(rename.get(x,x) for x in self.in_loop)
#         noncycle_nodes = set(rename.get(x,x) for x in self.ex_loop)
#         result = ConstraintLoop( cycle_nodes, noncycle_nodes )
#         return result

#
# def copyFormula(source, target) :
#     for i, n, t in source :
#         if t == 'atom' :
#             target.add_atom( n.identifier, n.probability, n.group )
#         elif t == 'conj' :
#             target.add_and( n.children )
#         elif t == 'disj' :
#             target.add_or( n.children )
#         else :
#             raise TypeError("Unknown node type '%s'" % t)
#
#     for name, node, label in source.get_names_with_label():
#         target.add_name(name, node, label)
#
# def breakCycles_lp(source, target=None) :
#
#     if target != None :
#         copyFormula(source,target)
#     else :
#         target = source
#
#     tmp_file = tempfile.mkstemp('.lp')[1]
#     with open(tmp_file, 'w') as f :
#         lf_to_smodels(target, f)
#     output = subprocess.check_output(['lp2acyc', tmp_file])
#     smodels_to_lf( target, output )
#
#     try :
#         os.remove(tmp_file)
#     except OSError :
#         pass
#
#     return target
#
#
#
# def expand_node( formula, node_id ) :
#     """Expand conjunctions by their body until a disjunction or atom is encountered.
#     This method assumes that all cycles go through a disjunctive node.
#     """
#     node = formula.get_node(abs(node_id))
#     nodetype = type(node).__name__
#     conjuncts = []
#     if nodetype == 'disj' :
#         return [node_id]
#     elif nodetype == 'atom' :
#         return [node_id]
#     elif node_id < 0 :
#         return [node_id]
#     else : # conj
#         for c in node.children :
#             conjuncts += expand_node(formula,c)
#         return conjuncts
#
# def lf_to_smodels( formula, out ) :
#
#     # '1' is an internal atom => false
#
#     #print (formula)
#     # Write rules
#     # Basic rule:
#     #   1 head #lits #neglits [ body literals with negative first ]
#     for i,n,t in formula :
#         if t == 'disj' :
#             for c in n.children :
#                 body = expand_node(formula, c)
#                 l = len(body)
#                 nl = len([ b for b in body if b < 0 ])
#                 body = [ abs(b)+1 for b in sorted(body) ]
#                 print('1 %s %s %s %s' % (i+1, l, nl, ' '.join(map(str,sorted(body))) ), file=out)
#     print (0, file=out)
#
#     for i,n,t in formula :
#         if t == 'atom' or t == 'disj' :
#             print (i+1, i, file=out)
#
#     # Symbol table => must contain all (otherwise hidden in output)
#     # Facts and disjunctions
#     #   2 a
#     #   3 b
#
#     print (0, file=out)
#     print ('B+', file=out)
#     # B+  positive evidence?
#
#     print (0, file=out)
#     print ('B-', file=out)
#     # B-  negative evidence?
#
#     print (0, file=out)
#
#     # Number of models
#     print (1, file=out)
#
# def smodels_to_lf( formula, acyclic ) :
#
#     section = 0
#     rules = defaultdict(list)
#     data = [[]]
#     for line in acyclic.split('\n') :
#         if line == '0' :
#             section += 1
#             data.append([])
#         else :
#             data[-1].append( line )
#
#     acyc_nodes = frozenset([ int(x.split()[0]) for x in data[1] if '_acyc_' in x ])
#     given_nodes = frozenset([ int(x.split()[0]) for x in data[1] if not '_acyc_' in x ])
#     if len(data[3]) > 1 :
#         root_node = int(data[3][1])
#     else :
#         root_node = None
#
#     for line in data[0] :
#         line = line.split()
#         line_head = int(line[1])
#         line_neg = int(line[3])
#         line_body_neg = frozenset(map(int,line[4:4+line_neg]))
#         line_body_pos = frozenset(map(int,line[4+line_neg:]))
#
#         # acyc_nodes are true
#         if acyc_nodes & line_body_neg : continue
#         # part of original program
#         if line_head in given_nodes : continue
#         # acyc_nodes are true
#         line_body_pos -= acyc_nodes
#         body = sorted([ -a for a in line_body_neg ] + list(line_body_pos))
#         rules[line_head].append(body)
#
#     translate = {}
#     for head in rules :
#         acyc_insert(formula, rules, head, given_nodes, translate)
#
#     if root_node != None :
#         formula.addConstraint(TrueConstraint(translate[root_node]))
#
#     #print (formula)
#     # print (translate[root_node])
#
# def acyc_insert( formula, rules, head, given, translate ) :
#     if head < 0 :
#         f = -1
#     else :
#         f = 1
#
#     if abs(head) in translate :
#         return f * translate[abs(head)]
#     elif abs(head) in given :
#         return f * (abs(head)-1)
#     else :
#         disjuncts = []
#         for body in rules[abs(head)] :
#             new_body = [ acyc_insert(formula, rules, x, given, translate ) for x in body ]
#             disjuncts.append(formula.add_and( new_body ))
#         new_node = formula.add_or( disjuncts )
#         translate[abs(head)] = new_node
#         return f*new_node
#
# import collections
#
# class OrderedSet(collections.MutableSet):
#
#     def __init__(self, iterable=None):
#         self.end = end = []
#         end += [None, end, end]         # sentinel node for doubly linked list
#         self.map = {}                   # key --> [key, prev, next]
#         if iterable is not None:
#             self |= iterable
#
#     def __len__(self):
#         return len(self.map)
#
#     def __contains__(self, key):
#         return key in self.map
#
#     def add(self, key):
#         if key not in self.map:
#             end = self.end
#             curr = end[1]
#             curr[2] = end[1] = self.map[key] = [key, curr, end]
#
#     def discard(self, key):
#         if key in self.map:
#             key, prev, next = self.map.pop(key)
#             prev[2] = next
#             next[1] = prev
#
#     def __iter__(self):
#         end = self.end
#         curr = end[2]
#         while curr is not end:
#             yield curr[0]
#             curr = curr[2]
#
#     def __reversed__(self):
#         end = self.end
#         curr = end[1]
#         while curr is not end:
#             yield curr[0]
#             curr = curr[1]
#
#     def pop(self, last=True):
#         if not self:
#             raise KeyError('set is empty')
#         key = self.end[1][0] if last else self.end[2][0]
#         self.discard(key)
#         return key
#
#     def __repr__(self):
#         if not self:
#             return '%s()' % (self.__class__.__name__,)
#         return '%s(%r)' % (self.__class__.__name__, list(self))
#
#     def __eq__(self, other):
#         if isinstance(other, OrderedSet):
#             return len(self) == len(other) and list(self) == list(other)
#         return set(self) == set(other)
