from __future__ import print_function

from collections import namedtuple, defaultdict

from .core import transform, ProbLogObject

from .core import LABEL_QUERY, LABEL_EVIDENCE_POS, LABEL_EVIDENCE_NEG, LABEL_EVIDENCE_MAYBE, LABEL_NAMED
from .util import Timer

import logging, tempfile, subprocess, os

class LogicFormulaBase(ProbLogObject) :
    
    def getAtomCount(self) :
        pass
        
    def __len__(self) :
        pass
    
    def addName(self, name, node_id, label=None) :
        pass
                
    def getNames(self, label=None) :
        pass
        
    def getNamesWithLabel(self) :
        pass
        
    def addQuery(self, name, key) :
        pass
    
    def queries(self) :
        pass
        
    def addEvidence(self, name, key, value) :
        pass
        
    def evidence(self) :
        pass
    
    def __iter__(self) :
        pass
        
    def getWeights(self) :
        pass
        
    def extractWeights(self, semiring) :
        pass

    def addConstraint(self, constraint) :
        pass
        
    def constraints(self) :
        pass
        
    def addAtom(self, identifier, weight, group) :
        pass
        
    def addAnd(self, children) :
        pass
        
    def addOr(self, children) :
        pass
        
    def addNot(self, children) :
        pass
        
    def getNodeType(self, key) :
        pass
        
    def iterNodes(self) :
        # return key, node, nodetype
        pass
                
    def ready(self) :
        # mark end of initialization
        pass

class LogicFormula(ProbLogObject) :
    """A propositional logic formula consisting of and, or, not and atoms."""
    
    TRUE = 0
    FALSE = None
        
    _atom = namedtuple('atom', ('identifier', 'probability', 'group') )
    _conj = namedtuple('conj', ('children') )
    _disj = namedtuple('disj', ('children') )
    # negation is encoded by using a negative number for the key
    
    def _create_atom( self, identifier, probability, group ) :
        return self._atom( identifier, probability, group )
    
    def _create_conj( self, children ) :
        return self._conj(children)
        
    def _create_disj( self, children ) :
        return self._disj(children)
    
    def __init__(self, auto_compact=True) :
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
        
        self.__atom_count = 0
        
        self.__auto_compact = auto_compact
        
        self.__constraints_me = {}
        self.__constraints = []
        
    def getAtomCount(self) :
        return self.__atom_count
        
    def isTrivial(self) :
        return self.getAtomCount() == len(self)
        
    def addQuery(self, name, node_id) :
        self.addName(name, node_id, label=LABEL_QUERY)
        
    def addEvidence(self, name, node_id, value) :
        if value==True :
            self.addName(name, node_id, LABEL_EVIDENCE_POS )
        elif value==False :
            self.addName(name, node_id, LABEL_EVIDENCE_NEG)
        else :
            self.addName(name, node_id, LABEL_EVIDENCE_MAYBE)
        
    def addName(self, name, node_id, label=None) :
        """Associates a name to the given node identifier."""
        self.__names[label][str(name)] = node_id
                
    def getNames(self, label=None) :
        if label == None :
            result = set()
            for forLabel in self.__names.values() :
                result |= set( forLabel.items() )
        else :
            result = self.__names.get( label, {} ).items()
        return result
        
    def getNamesWithLabel(self) :
        result = []
        for label in self.__names :
            for name, node in self.__names[label].items() :
                result.append( ( name, node, label ) )
        return result
        
    def getNodeByName(self, name) :
        for label in self.__names :
            res = self.__names[label].get(name)
            if res != None :
                return res
        raise KeyError()
        
    def _add( self, node, reuse=True ) :
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
        if group == None : return
        if not group in self.__constraints_me :
            self.__constraints_me[group] = ConstraintME(group)
        self.__constraints_me[group].add(node, self) 
        
    def addAtom( self, identifier, probability, group=None ) :
        """Add an atom to the formula.
        
        :param identifier: a unique identifier for the atom
        :type identifier: any
        :param probability: probability of the atom (or None if it is deterministic)
        :type probability: :class:`problog.logic.basic.Term`
        :param group: a group identifier that identifies mutually exclusive atoms (or None if no constraint)
        :type group: :class:`str`
        :returns: the identifiers of the node in the formula (returns self.TRUE for deterministic atoms)
        """
        if probability == None :
            return 0
        else :
            atom = self._create_atom( identifier, probability, group )
            node_id = self._add( atom )
            self.__atom_count += 1  # TODO doesn't take reuse into account
            self._addConstraintME(group, node_id)
            return node_id
    
    def addAnd( self, components ) :
        """Add a conjunction to the logic formula.
        
        :param components: a list of node identifiers that already exist in the logic formula.
        :returns: the key of the node in the formula (returns 0 for deterministic atoms)
        """
        return self._addCompound('conj', components, self.FALSE, self.TRUE)
        
    def addOr( self, components, readonly=True ) :
        """Add a disjunction to the logic formula.
        
        :param components: a list of node identifiers that already exist in the logic formula.
        :param readonly: indicates whether the node should be modifiable. This will allow additional disjunct to be added without changing the node key. Modifiable nodes are less optimizable.
        :returns: the key of the node in the formula (returns 0 for deterministic atoms)
        :rtype: :class:`int`
        """
        return self._addCompound('disj', components, self.TRUE, self.FALSE, readonly=readonly)
        
    def addDisjunct( self, key, component ) :
        """Add a component to the node with the given key."""
        if self.isTrue(key) :
            return key
        elif self.isFalse(key) :
            raise ValueError("Cannot update failing node")
        else :
            node = self._getNode(key)
            if type(node).__name__ != 'disj' :
                raise ValueError("Can only update disjunctive node")
                
            if component == None :
                # Don't do anything
                pass
            elif component == 0 :
                return self._update( key, self._disj( (0,) ) )
            else :
                return self._update( key, self._disj( node.children + (component,) ) )
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

    def _getNode(self, key) :
        """Get the content of the given node."""
        assert(self.isProbabilistic(key))
        assert(key > 0)            
        return self.__nodes[ key - 1 ]
        
    def _getNodeType(self, key) :
        """Get the type of the given node (fact, disj, conj)."""
        return type(self._getNode(key)).__name__
                                
    def _addCompound(self, nodetype, content, t, f, readonly=True, update=None) :
        """Add a compound term (AND or OR)."""
        assert( content )   # Content should not be empty
        
        if self.__auto_compact :
            # If there is a t node, (true for OR, false for AND)
            if t in content : return t
        
            # Eliminate unneeded node nodes (false for OR, true for AND)
            content = filter( lambda x : x != f, content )
        
            # Put into fixed order and eliminate duplicate nodes
            content = tuple(sorted(set(content)))
        
            # Empty OR node fails, AND node is true
            if not content : return f
                
            # Contains opposites: return 'TRUE' for or, 'FALSE' for and
            if len(set(content)) > len(set(map(abs,content))) : return t
            
            # If node has only one child, just return the child.
            # Don't do this for modifiable nodes, we need to keep a separate node.
            if (readonly and update == None) and len(content) == 1 : return content[0]
        else :
            content = tuple(content)
        
        if nodetype == 'conj' :
            node = self._create_conj( content )
            return self._add( node, reuse=self.__auto_compact )
        elif nodetype == 'disj' :
            node = self._create_disj( content )
            if update != None :
                # If an update key is set, update that node
                return self._update( update, node )
            elif readonly :
                # If the node is readonly, we can try to reuse an existing node.
                return self._add( node, reuse=self.__auto_compact )
            else :
                # If node is modifiable, we shouldn't reuse an existing node.
                return self._add( node, reuse=False )
        else :
            raise TypeError("Unexpected node type: '%s'." % nodetype) 
    
    def __iter__(self) :
        return iter(self.__nodes)
        
    def iterNodes(self) :
        for i, n in enumerate(self) :
            yield (i+1, n, type(n).__name__)
            
        
        
    def getWeights(self) :
        weights = {}
        for i, n in enumerate(self) :
            if type(n).__name__ == 'atom' :
                i = i + 1
                if n.probability != True :
                    weights[i] = n.probability
        return weights
        
    def extractWeights(self, semiring, weights=None) :
        if weights != None :
            weights = { self.getNodeByName(n) : v for n,v in weights.items() }
        
        result = {}
        for i, n in enumerate(self) :
            if type(n).__name__ == 'atom' :
                i = i + 1
                
                if weights != None : 
                    p = weights.get( i, n.probability )
                else :
                    p = n.probability
                if p != True :
                    value = semiring.value(p)
                    result[i] = value, semiring.negate(value)
                else :
                    result[i] = semiring.one(), semiring.one()

        for c in self.constraints() :
            c.updateWeights( result, semiring )
        
        return result
        
    def constraints(self) :
        return list(self.__constraints_me.values()) + self.__constraints
        
    def addConstraint(self, c) :
        self.__constraints.append(c)

    ##################################################################################
    ####                       LOOP BREAKING AND COMPACTION                       ####
    ##################################################################################

    def makeAcyclic(self, preserve_tables=False, output=None) :
        """Break cycles."""
        
        assert(not preserve_tables)                    
        
        # TODO implement preserve tables
        #   This copies the table information from self._def_nodes and translates all result nodes
        #   This requires all result nodes to be maintained separately (add them to protected).
        #   Problem: how to do this without knowledge about internal structure of the engine. 
        
        with Timer('Cycle breaking'):
          # Output formula
          if output == None : output = LogicDAG()
          
          # Protected nodes (these have to exist separately)
          protected = set( [ y for x,y in self.getNames() ] )
                  
          # Translation table from old to new.
          translate = {}
          
          # Handle the given nodes one-by-one
          for name, node, label in self.getNamesWithLabel() :
              if label != LABEL_NAMED :
                  new_node, cycles = self._extract( output, node, protected, translate )
                  translate[node] = new_node
                  output.addName(name, new_node, label)
        
        return output

            
    def _expand( self, index, children, protected, nodetype=None, anc=None ) :
        """Determine the list of all children of the node by combining the given node with its children of the same type, recursively."""
        
        if anc == None : anc = []
        
        if index in children :
            pass
        elif index in anc :
            children.add(index)
        elif index == 0 or index == None :
            children.add(index)
        elif nodetype != None and abs(index) in protected :
            children.add(index)
        elif index < 0 :
            # Combine OR with NOT AND (or vice versa)
            if nodetype != None :
                node = self._getNode(-index)
                ntype = type(node).__name__
                if not ntype in ('conj', 'disj') :
                    children.add(index)
                else :
                    if ntype != nodetype :
                        for c in node.children :
                            self._expand( -c, children, protected, nodetype, anc+[index])
                    else :
                        children.add(index)
            else :
                children.add(index)
        else :  # index > 0
            node = self._getNode(index)
            ntype = type(node).__name__
            if not ntype in ('conj', 'disj') :
                children.add(index)
            elif nodetype == None :
                nodetype = ntype
            
            if ntype == nodetype :
                for c in node.children :
                    self._expand( c, children, protected, nodetype, anc+[index])
            else :
                children.add(index)
        return children
        
    def _extract( self, gp, index, protected, translate, anc=None ) :
        """Copy the given node to a new formula, while breaking loops and node combining."""
        if anc == None : anc = []
        
        if index == 0 or index == None :
            return index, set()
        elif index in anc :
            return None, {index}
        elif abs(index) in translate :
            if index < 0 :
                return self.addNot(translate[abs(index)]), set()
            else :
                return translate[abs(index)], set()
        else :
            node = self._getNode( abs(index) )
            ntype = type(node).__name__
        
        if ntype == 'disj' or ntype == 'conj' :
            # Get all the children while squashing together nodes.
            children = self._expand( abs(index), set(), protected )
            cycles = set()
            new_children = set()
            for child in children :
                new_child, cycle = self._extract( gp, child, protected, translate, anc+[index] )
                new_children.add( new_child )
                cycles |= cycle
            if ntype == 'conj' :
                ncc = set()
                for nc in new_children :
                    gp._expand( nc, ncc, set(), 'conj')
                new_node = gp.addAnd( ncc )
            else :
                ncc = set()
                for nc in new_children :
                    gp._expand( nc, ncc, set(), 'disj')
                new_node = gp.addOr( ncc )

            if index in cycles and new_node != None and new_node != 0 :
                cycles.remove(index)
            if not cycles :
                translate[abs(index)] = new_node
            if index < 0 :
                return self.addNot(new_node), cycles
            else :
                return new_node, cycles
        else :
            res = translate.get(abs(index))
            if res == None :
                res = gp.addAtom( node.identifier, node.probability, node.group )
                translate[abs(index)] = res 
            if index < 0 :
                return self.addNot(res), set()
            else :
                return res, set()
        
        node = self.getNode(index)
    
    
    def __len__(self) :
        return len(self.__nodes)
        
    
    ##################################################################################
    ####                            OUTPUT GENERATION                             ####
    ##################################################################################
    
    def __str__(self) :
        s =  '\n'.join('%s: %s' % (i+1,n) for i, n in enumerate(self.__nodes))   
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
        
    def queries(self) :
        return self.getNames(LABEL_QUERY)

    def evidence(self) :
        evidence_true = self.getNames(LABEL_EVIDENCE_POS)
        evidence_false = self.getNames(LABEL_EVIDENCE_NEG)
        return list(evidence_true) + [ (a,-b) for a,b in evidence_false ]
        
    def named(self) :
        return self.getNames(LABEL_NAMED)
        
    def toDot(self, not_as_node=True) :
        
        not_as_edge = not not_as_node
        
        # Keep track of mutually disjunctive nodes.
        clusters = defaultdict(list)
        
        queries = self.getNames()
        
        # Keep a list of introduced not nodes to prevent duplicates.
        negative = set([])
        
        s = 'digraph GP {\n'
        for index, node in enumerate(self.__nodes) :
            index += 1
            nodetype = type(node).__name__
            
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
                elif node.group == None :                
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
        for name, index in queries :
            opt = ''
            if index == None :
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

class LogicDAG(LogicFormula) : 
    
    def __init__(self, auto_compact=True) :
        LogicFormula.__init__(self, auto_compact)
        
        
        
@transform(LogicFormula, LogicDAG)
def breakCycles(source, target) :
    logger = logging.getLogger('problog')
    result = source.makeAcyclic(preserve_tables=False, output=target)
    logger.debug("Ground program size: %s", len(result))
    return result
        

class StringKeyLogicFormula(ProbLogObject) :
    """A propositional logic formula consisting of and, or, not and atoms."""
    
    TRUE = 'true'
    FALSE = 'false'
    
    _atom = namedtuple('atom', ('identifier', 'probability', 'group') )
    _conj = namedtuple('conj', ('children') )
    _disj = namedtuple('disj', ('children') )
    # negation is encoded by using a negative number for the key
    
    def _create_atom( self, identifier, probability, group ) :
        return self._atom( identifier, probability, group )
    
    def _create_conj( self, children ) :
        return self._conj(children)
        
    def _create_disj( self, children ) :
        return self._disj(children)
    
    def __init__(self) :
        ProbLogObject.__init__(self)
        
        self.__nodes = defaultdict(list)
        
        # Node names (for nodes of interest)
        self.__names = defaultdict(dict)
        
        self.__atom_count = 0
        
        self.__constraints_me = {}
        self.__constraints = []
        
    def getAtomCount(self) :
        return self.__atom_count
        
    def isTrivial(self) :
        return self.getAtomCount() == len(self)
        
    def addQuery(self, name, node_id) :
        self.addName(name, node_id, label=LABEL_QUERY)
        
    def addEvidence(self, name, node_id, value) :
        if value==True :
            self.addName(name, node_id, LABEL_EVIDENCE_POS )
        elif value==False :
            self.addName(name, node_id, LABEL_EVIDENCE_NEG)
        else :
            self.addName(name, node_id, LABEL_EVIDENCE_MAYBE)
        
    def addName(self, name, node_id, label=None) :
        """Associates a name to the given node identifier."""
        self.__names[label][str(name)] = node_id
                
    def getNames(self, label=None) :
        if label == None :
            result = set()
            for forLabel in self.__names.values() :
                result |= set( forLabel.items() )
        else :
            result = self.__names.get( label, {} ).items()
        return result
        
    def getNamesWithLabel(self) :
        result = []
        for label in self.__names :
            for name, node in self.__names[label].items() :
                result.append( ( name, node, label ) )
        return result
        
    def getNodeByName(self, name) :
        for label in self.__names :
            res = self.__names[label].get(name)
            if res != None :
                return res
        raise KeyError()
        
    def _add( self, key, node, reuse=True ) :
        """Adds a new node, or reuses an existing one.
        
        :param node: node to add
        :param reuse: (default True) attempt to map the new node onto an existing one based on its content
        
        """        
        self.__nodes[key].append(node)
        return key
        
    def _update( self, key, value ) :
        """Replace the node with the given node."""
        self.__nodes[ key ] = [value]
    
    def _addConstraintME(self, group, node) :
        if group == None : return
        if not group in self.__constraints_me :
            self.__constraints_me[group] = ConstraintME(group)
        self.__constraints_me[group].add(node, self) 
        
    def addAtom( self, identifier, probability, group=None ) :
        """Add an atom to the formula.
        
        :param identifier: a unique identifier for the atom
        :type identifier: any
        :param probability: probability of the atom (or None if it is deterministic)
        :type probability: :class:`problog.logic.basic.Term`
        :param group: a group identifier that identifies mutually exclusive atoms (or None if no constraint)
        :type group: :class:`str`
        :returns: the identifiers of the node in the formula (returns self.TRUE for deterministic atoms)
        """
        if probability == None :
            return TRUE
        else :
            atom = self._create_atom( identifier, probability, group )
            node_id = self._add( identifier, atom )
            self.__atom_count += 1  # TODO doesn't take reuse into account
            self._addConstraintME(group, node_id)
            return node_id
    
    def addAnd( self, key, components ) :
        """Add a conjunction to the logic formula.
        
        :param components: a list of node identifiers that already exist in the logic formula.
        :returns: the key of the node in the formula (returns 0 for deterministic atoms)
        """
        return self._addCompound(key, 'conj', components, self.FALSE, self.TRUE)
        
    def addOr( self, key, components, readonly=True ) :
        """Add a disjunction to the logic formula.
        
        :param components: a list of node identifiers that already exist in the logic formula.
        :param readonly: indicates whether the node should be modifiable. This will allow additional disjunct to be added without changing the node key. Modifiable nodes are less optimizable.
        :returns: the key of the node in the formula (returns 0 for deterministic atoms)
        :rtype: :class:`int`
        """
        return self._addCompound(key, 'disj', components, self.TRUE, self.FALSE, readonly=readonly)
        
    def addDisjunct( self, key, component ) :
        """Add a component to the node with the given key."""
        raise ValueError('Formula does not support updates.')
    
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
    
    def isTrue( self, key ) :
        """Indicates whether the given node is deterministically True."""
        return key == self.TRUE
        
    def isFalse( self, key ) :
        """Indicates whether the given node is deterministically False."""
        return key == self.FALSE
        
    def isProbabilistic(self, key) :
        """Indicates whether the given node is probabilistic."""    
        return not self.isTrue(key) and not self.isFalse(key)

    def _getNode(self, key) :
        """Get the content of the given node."""
        n = self.__nodes[key]
        if len(n) > 1 :
            return self._create_disj(n)
        else :
            return n[0]
    
    def _getNodeType(self, key) :
        """Get the type of the given node (fact, disj, conj)."""
        return type(self._getNode(key)).__name__
                                
    def _addCompound(self, key, nodetype, content, t, f, readonly=True, update=None) :
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
        if len(content) == 1 : return self._add(key, content[0])
        
        content = tuple(content)
        
        if nodetype == 'conj' :
            node = self._create_conj( content )
            return self._add( key, node )
        elif nodetype == 'disj' :
            node = self._create_disj( content )
            if update != None :
                # If an update key is set, update that node
                return self._update( update, node )
            elif readonly :
                # If the node is readonly, we can try to reuse an existing node.
                return self._add( key, node )
            else :
                # If node is modifiable, we shouldn't reuse an existing node.
                return self._add( key, node, reuse=False )
        else :
            raise TypeError("Unexpected node type: '%s'." % nodetype) 
    
    def __iter__(self) :
        return iter(self.__nodes.values())
    
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
        
        
    def iterNodes(self) :
        for k in self.__nodes :
#            print ('R',k, self.__nodes[k])
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
                
        
    def getWeights(self) :
        weights = {}
        for i, n, t in self.iterNodes() :
            if t == 'atom' :
                if n.probability != True :
                    weights[i] = n.probability
        return weights
        
    def extractWeights(self, semiring, weights=None) :
        if weights != None :
            weights = { self.getNodeByName(n) : v for n,v in weights.items() }
        
        result = {}
        for i, n, t in self.iterNodes() :
            if t == 'atom' :                
                if weights != None : 
                    p = weights.get( i, n.probability )
                else :
                    p = n.probability
                if p != True :
                    value = semiring.value(p)
                    result[i] = value, semiring.negate(value)
                else :
                    result[i] = semiring.one(), semiring.one()

        for c in self.constraints() :
            c.updateWeights( result, semiring )
        
        return result
        
    def constraints(self) :
        return list(self.__constraints_me.values()) + self.__constraints
        
    def addConstraint(self, c) :
        self.__constraints.append(c)
        
    def __len__(self) :
        return len(self.__nodes)
        
    
    ##################################################################################
    ####                            OUTPUT GENERATION                             ####
    ##################################################################################
    
    def __str__(self) :
        s =  '\n'.join('%s: %s' % (i,n) for i, n, t in self.iterNodes())
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
        
    def queries(self) :
        return self.getNames(LABEL_QUERY)

    def evidence(self) :
        evidence_true = self.getNames(LABEL_EVIDENCE_POS)
        evidence_false = self.getNames(LABEL_EVIDENCE_NEG)
        return list(evidence_true) + [ (a,-b) for a,b in evidence_false ]
        
    def named(self) :
        return self.getNames(LABEL_NAMED)
        
    def toLogicFormula(self) :
        target = LogicFormula(auto_compact=False)
        translate = {}
        i = 0
        for k,n,t in self.iterNodes() :
            i += 1
            translate[k] = i
            translate['-' + str(k) ] = -i
        for k,n,t in self.iterNodes() :
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
                interm.addAnd( key, body )
                interm.addName( key, key, LABEL_NAMED )
            elif type(c).__name__ == 'Term' :
                key = str(c.withProbability())
                interm.addAtom( key, c.probability, None )
                interm.addName( key, key, LABEL_NAMED )
            else :
                raise Exception("Unexpected type: '%s'" % type(c).__name__)
        return interm

    


 
class Constraint(object) : 
    pass
    
class ConstraintME(Constraint) :
    """Mutually exclusive."""
    
    def __init__(self, group) :
        self.nodes = set()
        self.group = group
        self.extra_node = None
    
    def __str__(self) :
        return 'mutually_exclusive(%s, %s)' % (list(self.nodes), self.extra_node)
    
    def isTrue(self) :
        return len(self.nodes) <= 1
        
    def isFalse(self) :
        return False
        
    def isActive(self) :
        return not self.isTrue() and not self.isFalse()
    
    def add(self, node, formula) :
        self.nodes.add(node)
        if len(self.nodes) > 1 and self.extra_node == None :
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
        result = ConstraintME( self.group )
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
        if label != LABEL_NAMED :
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
        n = src._getNode(n)
        t = type(n).__name__
        if t == 'disj' :
            cycle_free += [ c for c in n.children if not c in loop ]
        elif t == 'conj' :
            pass
        else :
            raise Exception('?')
    return cycle_free
    
def findCycles( src, a, path, relevant=None ) :
    n = src._getNode(a)
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
        
    def updateWeights(self, weights, semiring) :
        pass
        
    def __str__(self) :
        return '%s is true' % self.node

def copyFormula(source, target) :
    for i, n, t in source.iterNodes() :
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
    node = formula._getNode(abs(node_id))
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
    for i,n,t in formula.iterNodes() :
        if t == 'disj' :
            for c in n.children :
                body = expand_node(formula, c)
                l = len(body)
                nl = len([ b for b in body if b < 0 ])
                body = [ abs(b)+1 for b in sorted(body) ]
                print('1 %s %s %s %s' % (i+1, l, nl, ' '.join(map(str,sorted(body))) ), file=out)
    print (0, file=out)

    for i,n,t in formula.iterNodes() :
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
