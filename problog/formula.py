from __future__ import print_function

from collections import namedtuple, defaultdict
import warnings

from .core import transform, ProbLogObject, ProbLogError, deprecated_function, deprecated

from .util import Timer
from .logic import Term

import logging, tempfile, subprocess, os


class LogicFormula(ProbLogObject):
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
    
    
    TRUE = 0
    FALSE = None
    
    LABEL_QUERY = "query"
    LABEL_EVIDENCE_POS = "evidence+"
    LABEL_EVIDENCE_NEG = "evidence-"
    LABEL_EVIDENCE_MAYBE = "evidence?" 
    LABEL_NAMED = "named"
        
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
    
    def __init__(self, auto_compact=True, avoid_name_clash=False, keep_order=False, use_string_names=False, **kwdargs):
        ProbLogObject.__init__(self)
        
        # List of nodes
        self.__nodes = []
        # Lookup index for 'atom' nodes, key is identifier passed to addAtom()
        self.__index_atom = {}
        # Lookup index for 'and' nodes, key is tuple of sorted children 
        self.__index_conj = {}
        # Lookup index for 'or' nodes, key is tuple of sorted children
        self.__index_disj = {}
        
        # Node names (for nodes of interest)
        self.__names = defaultdict(dict)
        self.__names_order = []
        self.__names_reverse = {}
        
        self.__atom_count = 0
        
        self.__auto_compact = auto_compact
        self.__avoid_name_clash = avoid_name_clash
        self.__keep_order = keep_order
        
        self.__constraints_me = {}
        self.__constraints = []
        self.__constraints_for_node = defaultdict(list)
        
        self.__use_string_names = use_string_names
        
    def constraintsForNode(self, node) :
        return self.__constraints_for_node[node]
        
    def addConstraintOnNode(self, node, constraint) :
        self.__constraints_for_node[node].append(constraint)
        
    def getAtomCount(self) :
        return self.__atom_count
        
    def isTrivial(self) :
        return self.getAtomCount() == len(self)
        
    ######################################################################
    ###                          Manage labels                         ###
    ######################################################################
    
    def addName(self, name, node_id, label=None) :
        """Associates a name to the given node identifier.
        
            :param name: name of the node
            :param node_id: id of the node
            :param label: type of node (see LogicFormula.LABEL_*)
        """
        if self.isProbabilistic(node_id):
            node = self.get_node(abs(node_id))
            node = type(node)(*(node[:-1] + (name,)))
            self._update(abs(node_id), node)

        if self.__use_string_names :
            name = str(name)
        if not label in self.__names or not name in self.__names[label] :
            self.__names_order.append( (label,name) )
        self.__names[label][name] = node_id
        if self.__avoid_name_clash :
            self.__names_reverse[node_id] = name
            
    def addQuery(self, name, node_id) :
        """Associates a name to the given node identifier and marks it as a query.
            This redirects to :func:`addName` (name, node_id, LABEL_QUERY)
        """
        self.addName(name, node_id, label=self.LABEL_QUERY)
        
    def addEvidence(self, name, node_id, value) :
        """Associates a name to the given node identifier and marks it as evidence.
            See :func:`addName`.
            
            :param value: True (positive evidence) / False (negative evidence) / other (unknown evidence)
            
        """
        if value==True :
            self.addName(name, node_id, self.LABEL_EVIDENCE_POS )
        elif value==False :
            self.addName(name, node_id, self.LABEL_EVIDENCE_NEG)
        else :
            self.addName(name, node_id, self.LABEL_EVIDENCE_MAYBE)

    def getNames(self, label=None):
        warnings.warn('getNames() is deprecated', FutureWarning)
        return self.get_names(label)

    def get_names(self, label=None) :
        """Get a list of named nodes.
            :param label: restrict names to the given label (default: all labels)
            :return: list of tuples (name, node_id)
        """
        if label is None :
            result = []
            for label, name in self.__names_order :
                result.append( ( name, self.__names[label][name]) )
        else :
            result = self.__names.get( label, {} ).items()
        return result
        
    def getNamesWithLabel(self) :
        """Get a list of named nodes with label type.
            :return: list of tuples (name, node_id, label)
        """
        result = []
        for label, name in self.__names_order :
            result.append( ( name, self.__names[label][name], label))
        return result
        
    def getNodeByName(self, name) :
        """Get a node by name.
            :param name: name of the node
            :return: node_id of the node
            :raise KeyError: if name does not exist
        """
        for label in self.__names :
            res = self.__names[label].get(name,'#ERROR#')
            if res != '#ERROR#' :
                return res
        raise KeyError()
        
    def queries(self) :
        """Get query nodes."""
        return self.get_names(self.LABEL_QUERY)

    def evidence(self) :
        """Get evidence nodes."""
        evidence_true = self.get_names(self.LABEL_EVIDENCE_POS)
        evidence_false = self.get_names(self.LABEL_EVIDENCE_NEG)
        return list(evidence_true) + [ (a,self.addNot(b)) for a,b in evidence_false ]
        
    def named(self) :
        """Get named nodes."""
        return self.get_names(self.LABEL_NAMED)
    
        
    ######################################################################
    ###                          Manage logic                          ###
    ######################################################################
    
        
    def _add( self, node, key=None, reuse=True ) :
        """Adds a new node, or reuses an existing one.
        
        :param node: node to add
        :param reuse: (default True) attempt to map the new node onto an existing one based on its content
        
        """
        if reuse :            
            # Determine the node's key and lookup identifier base on node type.
            ntype = type(node).__name__
            if ntype == 'atom' :
                key = node.identifier
                collection = self.__index_atom
            elif ntype == 'conj' :
                key = node.children
                collection = self.__index_conj
            elif ntype == 'disj' :
                key = node.children
                collection = self.__index_disj
            else :
                raise TypeError("Unexpected node type: '%s'." % ntype)
        
            if not key in collection :
                # Create a new entry, starting from 1
                index = len(self.__nodes) + 1
                # Add the entry to the collection
                collection[key] = index
                # Add entry to the set of nodes
                self.__nodes.append( node )
            else :
                # Retrieve the entry from collection
                index = collection[key]
        else :
            # Don't reuse, just add node.
            index = len(self.__nodes) + 1
            self.__nodes.append( node )
            
        # Return the entry
        return index
        
    def _update( self, key, value ) :
        """Replace the node with the given node."""
        assert(self.isProbabilistic(key))
        assert(key > 0)            
        self.__nodes[ key - 1 ] = value
    
    def _addConstraintME(self, group, node) :
        if group is None : return
        constraint = self.__constraints_me.get(group)
        if constraint is None :
            constraint = ConstraintAD(group)
            self.__constraints_me[group] = constraint
        constraint.add(node, self)
        self.addConstraintOnNode(constraint, node)
        
    def addAtom(self, identifier, probability, group=None, name=None):
        """Add an atom to the formula.
        
        :param identifier: a unique identifier for the atom
        :param probability: probability of the atom
        :param group: a group identifier that identifies mutually exclusive atoms (or None if no constraint)
        :returns: the identifiers of the node in the formula (returns self.TRUE for deterministic atoms)
        
        This function has the following behavior :
        
        * If ``probability`` is set to ``None`` then the node is considered to be deterministically true and the function will return :attr:`TRUE`.
        * If a node already exists with the given ``identifier``, the id of that node is returned.
        * If ``group`` is given, a mutual exclusivity constraint is added for all nodes sharing the same group.
        * To add an explicitly present deterministic node you can set the probability to ``True``.
        """
        if probability is None:
            return self.TRUE
        else:
            atom = self._create_atom(identifier, probability, group, name)
            node_id = self._add(atom, key=identifier)
            self.__atom_count += 1  # TODO doesn't take reuse into account
            self._addConstraintME(group, node_id)
            return node_id
    
    def addAnd(self, components, key=None, name=None):
        """Add a conjunction to the logic formula.
        
        :param components: a list of node identifiers that already exist in the logic formula.
        :returns: the key of the node in the formula (returns 0 for deterministic atoms)
        """
        return self._addCompound('conj', components, self.FALSE, self.TRUE, key=key, name=name)
        
    def addOr(self, components, key=None, readonly=True, name=None):
        """Add a disjunction to the logic formula.
        
        :param components: a list of node identifiers that already exist in the logic formula.
        :param readonly: indicates whether the node should be modifiable. This will allow additional disjunct to be added without changing the node key. Modifiable nodes are less optimizable.
        :returns: the key of the node in the formula (returns 0 for deterministic atoms)
        :rtype: :class:`int`
        
        By default, all nodes in the data structure are immutable (i.e. readonly).
        This allows the data structure to optimize nodes, but it also means that cyclic formula can not be stored because the identifiers of all descendants must be known add creation time.
        
        By setting `readonly` to False, the node is made mutable and will allow adding disjunct later using the :func:`addDisjunct` method.
        This may cause the data structure to contain superfluous nodes.
        """
        return self._addCompound('disj', components, self.TRUE, self.FALSE, key=key, readonly=readonly, name=name)
        
    def addDisjunct(self, key, component):
        """Add a component to the node with the given key.
        
        :param key: id of the node to update
        :param component: the component to add
        :return: key
        :raises: :class:`ValueError` if ``key`` points to an invalid node
        
        This may only be called with a key that points to a disjunctive node or :attr:`TRUE`.
        """
        if self.isTrue(key) :
            return key
        elif self.isFalse(key) :
            raise ValueError("Cannot update failing node")
        else :
            node = self.getNode(key)
            if type(node).__name__ != 'disj' :
                raise ValueError("Can only update disjunctive node")
                
            if component is None :
                # Don't do anything
                pass
            elif component == 0:
                return self._update(key, self._create_disj((0,), name=node.name))
            else:
                return self._update(key, self._create_disj(node.children + (component,), name=node.name))
            return key
    
    def addNot( self, component ) :
        """Returns the key to the negation of the node."""
        if self.isTrue(component) :
            return self.FALSE
        elif self.isFalse(component) :
            return self.TRUE
        else :
            return -component
    
    def isTrue( self, key ) :
        """Indicates whether the given node is deterministically True."""
        return key == self.TRUE
        
    def isFalse( self, key ) :
        """Indicates whether the given node is deterministically False."""
        return key == self.FALSE
        
    def isProbabilistic(self, key) :
        """Indicates whether the given node is probabilistic."""    
        return not self.isTrue(key) and not self.isFalse(key)
        
    def isAtom(self, key) :
        return self._getNodeType(key) == 'atom'
        
    def isAnd(self, key) :
        return self._getNodeType(key) == 'conj'
        
    def isOr(self, key) :
        return self._getNodeType(key) == 'disj'
        
    def isCompound(self, key) :
        return not self.isAtom(key)

    def get_node(self, key):
        assert(self.isProbabilistic(key))
        assert(key > 0)
        return self.__nodes[ key - 1 ]

    def getNode(self, key) :
        """Get the node with the given key."""
        assert(self.isProbabilistic(key))
        assert(key > 0)            
        return self.__nodes[ key - 1 ]
        
    def _getNode(self, key) :
        """Get the content of the given node."""
        warnings.warn('LogicFormula._getNode(key) is deprecated. Use LogicFormula.getNode(key) instead.', FutureWarning)
        assert(self.isProbabilistic(key))
        assert(key > 0)            
        return self.__nodes[ key - 1 ]
        
    def _getNodeType(self, key) :
        """Get the type of the given node (fact, disj, conj)."""
        return type(self.getNode(key)).__name__
                                
    def _addCompound(self, nodetype, content, t, f, key=None, readonly=True, update=None, name=None):
        """Add a compound term (AND or OR)."""
        assert( content )   # Content should not be empty
        
        if self.__auto_compact :
            # If there is a t node, (true for OR, false for AND)
            if t in content : return t
        
            # Eliminate unneeded node nodes (false for OR, true for AND)
            content = filter( lambda x : x != f, content )
        
            # Put into fixed order and eliminate duplicate nodes
            if self.__keep_order :
                content = tuple(content)
            else :
                content = tuple(sorted(set(content)))
        
            # Empty OR node fails, AND node is true
            if not content : return f
                
            # Contains opposites: return 'TRUE' for or, 'FALSE' for and
            if len(set(content)) > len(set(map(abs,content))) : return t
            
            # If node has only one child, just return the child.
            # Don't do this for modifiable nodes, we need to keep a separate node.
            if (readonly and update is None) and len(content) == 1:
                if self.__avoid_name_clash:
                    name_old = self.get_node(abs(content[0])).name
                    if name is None or name_old is None or name == name_old:
                        return content[0]
                else:
                    return content[0]
        else :
            content = tuple(content)

        if nodetype == 'conj':
            node = self._create_conj(content, name)
            return self._add( node, reuse=self.__auto_compact )
        elif nodetype == 'disj':
            node = self._create_disj(content, name)
            if update is not None:
                # If an update key is set, update that node
                return self._update(update, node)
            elif readonly:
                # If the node is readonly, we can try to reuse an existing node.
                return self._add(node, reuse=self.__auto_compact)
            else:
                # If node is modifiable, we shouldn't reuse an existing node.
                return self._add(node, reuse=False)
        else:
            raise TypeError("Unexpected node type: '%s'." % nodetype) 
    
    def __iter__(self):
        """Iterate over the nodes in the formula.
        
            :returns: iterator over tuples ( key, node, type )
        """
        for i, n in enumerate(self.__nodes):
            yield (i+1, n, type(n).__name__)
        
        
    def iterNodes(self) :
        """Iterate over the nodes in the formula.
            
            :returns: iterator over tuples ( key, node, type )
        """
        warnings.warn('LogicFormula.iterNodes() is deprecated. Please update your code to use default iteration (__iter__).', FutureWarning)
        return self.__iter__()

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
            to_enumerate = OrderedSet(abs(n) for q, n in self.queries() if self.isProbabilistic(n))
            to_enumerate |= OrderedSet(abs(n) for q, n in self.evidence() if self.isProbabilistic(n))
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
                    c_n = self.get_node(c_i)
                    c_t = type(c_n).__name__
                    if c_t == 'atom':
                        yield i, [c_i]
                        if abs(c_i) not in enumerated:
                            to_enumerate.add(abs(c_i))
                    elif c_t == 'conj':
                        body = self._unroll_conj(c_n)
                        to_enumerate |= (OrderedSet(map(abs, body)) - enumerated)
                        yield i, body
                    else:
                        yield i, [c_i]
                        if abs(c_i) not in enumerated:
                            to_enumerate.add(abs(c_i))


    def __len__(self) :
        """Returns the number of nodes in the formula."""
        return len(self.__nodes)        
            
    ######################################################################
    ###                         Manage weights                         ###
    ######################################################################
        
    def getWeights(self) :
        """Get the weights for all atoms in the data structure.
            Atoms with weight `True` are not returned.
            :returns: dictionary { key: weight }
        """
        weights = {}
        for i, n, t in self :
            if t == 'atom' :
                if n.probability != True :
                    weights[i] = n.probability
        return weights
        
    def extractWeights(self, semiring, weights=None) :
        """Extracts the positive and negative weights for all atoms in the data structure.
            
            :param semiring: semiring that determines the interpretation of the weights
            :param weights: dictionary of { node name : weight } that overrides the builtin weights
            :returns: dictionary { key: (positive weight, negative weight) }
            
            Atoms with weight set to ``True`` will get weight ``( semiring.one(), semiring.one() )``.
            
            All constraints are applied to the weights.
        """
        
        
        if weights != None :
            weights = { self.getNodeByName(n) : v for n,v in weights.items() }
        
        result = {}
        for i, n, t in self :
            if t == 'atom' :
                if weights != None : 
                    p = weights.get( i, n.probability )
                else :
                    p = n.probability
                if p != True :
                    result[i] = semiring.pos_value(p), semiring.neg_value(p)
                else :
                    result[i] = semiring.one(), semiring.one()
                    
        for c in self.constraints() :
            c.updateWeights( result, semiring )
        
        return result
        
    ######################################################################
    ###                       Manage constraints                       ###
    ######################################################################
        
    def constraints(self) :
        """Returns a list of all constraints."""
        return list(self.__constraints_me.values()) + self.__constraints
        
    def add_constraint(self, c) :
        """Adds a constraint to the model."""
        self.__constraints.append(c)
        for node in c.getNodes() :
            self.addConstraintOnNode(node,c)


    ##################################################################################
    ####                            OUTPUT GENERATION                             ####
    ##################################################################################
    
    def __str__(self) :
        s =  '\n'.join('%s: %s' % (i,n) for i, n, t in self)   
        f = True
        for q in self.queries() :
            if f :
                f = False
                s += '\nQueries : '
            s += '\n* %s : %s' % q

        f = True
        for q in self.evidence() :
            if f :
                f = False
                s += '\nEvidence : '
            s += '\n* %s : %s' % q

        f = True
        for q in self.named() :
            if f :
                f = False
                s += '\nNamed : '
            s += '\n* %s : %s' % q

        f = True
        for c in self.constraints () :
            if c.isActive() :
                if f :
                    f = False
                    s += '\nConstraints : '
                s += '\n* ' + str(c)
        return s + '\n'

    def _unroll_conj(self, node):
        assert type(node).__name__ == 'conj'

        if len(node.children) == 1:
            return node.children
        elif len(node.children) == 2 and node.children[1] > 0:
            children = [node.children[0]]
            current = node.children[1]
            current_node = self.getNode(current)
            while type(current_node).__name__ == 'conj' and len(current_node.children) == 2:
                children.append(current_node.children[0])
                current = current_node.children[1]
                if current > 0:
                    current_node = self.getNode(current)
                else:
                    current_node = None
            children.append(current)
        else:
            children = node.children
        return children

    def to_prolog(self, yap_style=False):
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

        :param yap_style: use Yap-style negation (i.e. ``\+ (atom)``) instead of ``\+atom``.
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

    def to_dot(self, not_as_node=True) :
        
        not_as_edge = not not_as_node
        
        # Keep track of mutually disjunctive nodes.
        clusters = defaultdict(list)
        
        queries = self.get_names()
        
        # Keep a list of introduced not nodes to prevent duplicates.
        negative = set([])
        
        s = 'digraph GP {\n'
        for index, node, nodetype in self :
            
            if nodetype == 'conj' :
                s += '%s [label="AND", shape="box", style="filled", fillcolor="white"];\n' % (index)
                for c in node.children :
                    opt = ''
                    if c < 0 and not c in negative and not_as_node :
                        s += '%s [label="NOT"];\n' % (c)
                        s += '%s -> %s;\n' % (c,-c)
                        negative.add(c)
                        
                    if c < 0 and not_as_edge :    
                        opt = '[arrowhead="odotnormal"]'
                        c = -c
                    if c != 0 :
                        s += '%s -> %s%s;\n' % (index,c, opt)
            elif nodetype == 'disj' :
                s += '%s [label="OR", shape="diamond", style="filled", fillcolor="white"];\n' % (index)
                for c in node.children :
                    opt = ''
                    if c < 0 and not c in negative and not_as_node :
                        s += '%s [label="NOT"];\n' % (c)
                        s += '%s -> %s;\n' % (c,-c)
                        negative.add(c)
                    if c < 0 and not_as_edge :    
                        opt = '[arrowhead="odotnormal"]'
                        c = -c
                    if c != 0 :
                        s += '%s -> %s%s;\n' % (index,c, opt)
            elif nodetype == 'atom' :
                if node.probability == True :
                    pass
                elif node.group is None :                
                    s += '%s [label="%s", shape="ellipse", style="filled", fillcolor="white"];\n' % (index, node.probability)
                            #, node.functor, ', '.join(map(str,node.args)))
                else :
                    clusters[node.group].append('%s [ shape="ellipse", label="%s", style="filled", fillcolor="white" ];\n' % (index, node.probability))
            else :
                raise TypeError("Unexpected node type: '%s'" % nodetype)
        
        c = 0
        for cluster, text in clusters.items() :
            if len(text) > 1 :
                s += 'subgraph cluster_%s { style="dotted"; color="red"; \n\t%s\n }\n' % (c,'\n\t'.join(text))
            else :
                s += text[0]
            c += 1 
            
        q = 0
        for name, index in set(queries) :
            opt = ''
            if index is None :
                index = 'false'
                if not_as_node :
                    s += '%s [label="NOT"];\n' % (index)
                    s += '%s -> %s;\n' % (index,0)
                elif not_as_edge :
                    opt = ', arrowhead="odotnormal"'
                if not 0 in negative :
                    s += '%s [label="true"];\n' % (0)
                    negative.add(0)
            elif index < 0 : # and not index in negative :
                if not_as_node :
                    s += '%s [label="NOT"];\n' % (index)
                    s += '%s -> %s;\n' % (index,-index)
                    negative.add(index)
                elif not_as_edge :
                    index = -index
                    opt = ', arrowhead="odotnormal"'
            elif index == 0 and not index in negative :
                s += '%s [label="true"];\n' % (index)
                negative.add(0)
            s += 'q_%s [ label="%s", shape="plaintext" ];\n'   % (q, name)
            s += 'q_%s -> %s [style="dotted" %s];\n'  % (q, index, opt)
            q += 1
        return s + '}'

    toProlog = deprecated_function('toProlog', to_prolog)
    toDot = deprecated_function('toDot', to_dot)

class LogicDAG(LogicFormula):
    
    def __init__(self, auto_compact=True, **kwdargs):
        LogicFormula.__init__(self, auto_compact, **kwdargs)


class DeterministicLogicFormula(LogicFormula):

    def __init__(self, **kwdargs):
        LogicFormula.__init__(self, **kwdargs)

    def addAtom(self, identifier, probability, group=None, name=None):
        return self.TRUE


@transform(LogicFormula, LogicDAG)
def break_cycles(source, target, **kwdargs):
    logger = logging.getLogger('problog')
    with Timer('Cycle breaking'):
        cycles_broken = set()
        content = set()
        translation = defaultdict(list)

        for q, n in source.queries():
            if source.isProbabilistic(n):
                newnode = _break_cycles(source, target, n, [], cycles_broken, content, translation)
            else:
                newnode = n
            target.addName(q, newnode, target.LABEL_QUERY)

        for q, n in source.evidence():
            newnode = _break_cycles(source, target, abs(n), [], cycles_broken, content, translation)
            if n < 0:
                target.addName(q, newnode, target.LABEL_EVIDENCE_NEG)
            else:
                target.addName(q, newnode, target.LABEL_EVIDENCE_POS)

        logger.debug("Ground program size: %s", len(target))
        return target

def _break_cycles(source, target, nodeid, ancestors, cycles_broken, content, translation):
    # TODO reuse from translation
    negative_node = nodeid < 0
    nodeid = abs(nodeid)

    if nodeid in ancestors:
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
                if negative_node:
                    return target.addNot(newnode)
                else:
                    return newnode
    child_cycles_broken = set()
    child_content = set()

    content.add(nodeid)
    node = source.get_node(nodeid)
    nodetype = type(node).__name__
    if nodetype == 'atom':
        newnode = target.addAtom(node.identifier, node.probability, node.group, node.name)
    else:
        children = [_break_cycles(source, target, child, ancestors + [nodeid], child_cycles_broken, child_content, translation) for child in node.children]
        newname = node.name
        if newname is not None and child_cycles_broken:
            newfunc = newname.functor + '_cb_' + str(len(translation[nodeid]))
            newname = Term(newfunc, *newname.args)
        if nodetype == 'conj':
            newnode = target.addAnd(children, name=newname)
        else:
            newnode = target.addOr(children, name=newname)
    translation[nodeid].append((newnode, child_cycles_broken, child_content-child_cycles_broken))
    content |= child_content
    cycles_broken |= child_cycles_broken

    if negative_node:
        return target.addNot(newnode)
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
            
    def addNot( self, component ) :
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
                i = target.addAtom( n.identifier, n.probability, n.group )
            elif t == 'disj' :
                i = target.addOr( [ translate[x] for x in n.children ] )
            elif t == 'conj' :
                i = target.addAnd( [ translate[x] for x in n.children ] )
            assert(i == translate[k])
            
        for name, key, label in self.getNamesWithLabel() :
            key = self._deref(key)
            target.addName( name, translate[key], label )
        
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
                interm.addAnd( body, key=key )
                interm.addName( key, key, interm.LABEL_NAMED )
            elif type(c).__name__ == 'Term' :
                key = str(c.withProbability())
                interm.addAtom( key, c.probability, None )
                interm.addName( key, key, interm.LABEL_NAMED )
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
    
    def add(self, node, formula) :
        self.nodes.add(node)
        if len(self.nodes) > 1 and self.extra_node is None :
            # If there are two or more choices -> add extra choice node
            self.updateLogic( formula )
    
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
            self.extra_node = formula.addAtom( ('%s_extra' % (self.group,)), True, None )
            formula.addConstraintOnNode(self, self.extra_node)
    
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
            target.addAtom( n.identifier, n.probability, n.group )
        elif t == 'conj' :
            target.addAnd( n.children )
        elif t == 'disj' :
            target.addOr( n.children )
        else :
            raise TypeError("Unknown node type '%s'" % t)
            
    for name, node, label in source.getNamesWithLabel() :
        target.addName(name, node, label)

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
            disjuncts.append(formula.addAnd( new_body ))
        new_node = formula.addOr( disjuncts )
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
