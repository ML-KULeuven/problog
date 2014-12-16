from __future__ import print_function

from collections import namedtuple, defaultdict

from .core import transform, ProbLogObject


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
        self.addName(name, node_id, label='query')
        
    def addEvidence(self, name, node_id, value) :
        if value :
            self.addName(name, node_id, 'evidence')
        else :
            self.addName(name, node_id, '-evidence')
        
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
        
    def extractWeights(self, semiring) :
        weights = {}
        for i, n in enumerate(self) :
            if type(n).__name__ == 'atom' :
                i = i + 1
                if n.probability != True :
                    weights[i] = semiring.value(n.probability), semiring.negate(semiring.value(n.probability))
                else :
                    weights[i] = semiring.one(), semiring.one()

        for c in self.constraints() :
            c.updateWeights( weights, semiring )
        
        return weights
        
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
        
        
        # Output formula
        if output == None : output = LogicDAG()
        
        # Protected nodes (these have to exist separately)
        protected = set( [ y for x,y in self.getNames() ] )
                
        # Translation table from old to new.
        translate = {}
        
        # Handle the given nodes one-by-one
        for name, node, label in self.getNamesWithLabel() :
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
        for c in self.constraints () :
            if c.isActive() :
                if f :
                    f = False
                    s += '\nConstraints : '
                s += '\n* ' + str(c)
        return s   
        
    def queries(self) :
        return self.getNames('query')

    def evidence(self) :
        evidence_true = self.getNames('evidence') 
        evidence_false = self.getNames('-evidence') 
        return evidence_true + [ (a,-b) for a,b in evidence_false ]

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
                if node.group == None :                
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
                index = 0
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
    return source.makeAcyclic(preserve_tables=False, output=target)
 
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

