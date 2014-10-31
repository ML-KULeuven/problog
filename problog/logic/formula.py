from __future__ import print_function

from collections import namedtuple, defaultdict

class LogicFormula(object) :
    """A propositional logic formula consisting of and, or, not and atoms."""
    
    TRUE = 0
    FALSE = None
    
    _atom = namedtuple('atom', ('identifier', 'probability', 'group') )
    _conj = namedtuple('conj', ('children') )
    _disj = namedtuple('disj', ('children') )
    
    def __init__(self) :
        self.__nodes = []
        self.__index_atom = {}
        self.__index_conj = {}
        self.__index_disj = {}
        
    def _add( self, key, collection, node ) :
        """Adds a new node, or reuses an existing one."""
        if collection == None :
            index = len(self.__nodes) + 1
            self.__nodes.append( node )
        elif not key in collection :
            # Create a new entry, starting from 1
            index = len(self.__nodes) + 1
            # Add the entry to the collection
            collection[key] = index
            # Add entry to the set of nodes
            self.__nodes.append( node )
        else :
            # Retrieve the entry from collection
            index = collection[key]
        # Return the entry
        return index
        
    def _update( self, key, value ) :
        assert(self.isProbabilistic(key))
        assert(key > 0)            
        self.__nodes[ key - 1 ] = value
    
        
    def addAtom( self, identifier, probability, group=None ) :
        """Add an atom to the formula.
        
        :param identifier: a unique identifier for the atom
        :type identifier: any
        :param probability: probability of the atom (or None if it is deterministic)
        :type probability: :class:`problog.logic.basic.Term`
        :param group: a group identifier that identifies mutually exclusive atoms (or None if no constraint)
        :type group: :class:`str`
        :returns: the identifiers of the node in the formula (returns 0 for deterministic atoms)
        :rtype: :class:`int`
        """
        if probability == None :
            return 0
        else :
            return self._add( identifier, self.__index_atom, self._atom( identifier, probability, group ) )
    
    def addAnd( self, components ) :
        """Add a conjunction to the logic formula.
        
        :param components: a list of node identifiers that already exist in the logic formula.
        :returns: the key of the node in the formula (returns 0 for deterministic atoms)
        :rtype: :class:`int`
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
            self._addCompound('disj', node.children + (component,), self.TRUE, self.FALSE, readonly=False, update=key )
            return key
    
    def negate( self, key ) :
        """Returns the key to the negation of the node."""
        if self.isTrue(key) :
            return self.FALSE
        elif self.isFalse(key) :
            return self.TRUE
        else :
            return -key
    
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
                
    def makeAcyclic(self, nodes) :
        """Break cycles."""
        # Output formula
        output = LogicFormula()
        
        # Protected nodes (these have to exist separately)
        protected = set(nodes)
        
        # Translation table from old to new.
        translate = {}
        
        # Handle the given nodes one-by-one
        output_nodes = []
        for node in nodes :
            new_node, cycles = self._extract( output, node, protected, translate )
            translate[node] = new_node
            output_nodes.append(new_node)
        
        return output, output_nodes
        
    def makeNNF(self) :
        """Transform into NNF form."""
        pass
        
    def makeCNF(self, completion=True) :
        """Transform into CNF form."""
        pass
        
    def _addCompound(self, nodetype, content, t, f, readonly=True, update=None) :
        assert( content )   # Content should not be empty
        
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
        
        if nodetype == 'conj' :
            node = self._conj( content )
            return self._add( content, self.__index_conj, node )
        elif nodetype == 'disj' :
            node = self._disj( content )
            if update != None :
                # If an update key is set, update that node
                return self._update( update, node )
            elif readonly :
                # If the node is readonly, we can try to reuse an existing node.
                return self._add( content, self.__index_disj, node )
            else :
                # If node is modifiable, we shouldn't reuse an existing node.
                return self._add( content, None, node )
        else :
            raise ValueError("Unexpected node type: '%s'" % nodetype) 
    
    

    ##################################################################################
    ####                       LOOP BREAKING AND COMPACTION                       ####
    ##################################################################################
            
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
                return self.negate(translate[abs(index)]), set()
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
                    return new_node, cycles
            if index < 0 :
                return self.negate(new_node), cycles
            else :
                return new_node, cycles
        else :
            res = translate.get(abs(index))
            if res == None :
                res = gp.addAtom( node.identifier, node.probability, node.group )
                translate[abs(index)] = res 
            if index < 0 :
                return self.negate(res), set()
            else :
                return res, set()
        
        node = self.getNode(index)
    
        
    
    ##################################################################################
    ####                            OUTPUT GENERATION                             ####
    ##################################################################################
    
    def __str__(self) :
        s =  '\n'.join('%s: %s' % (i+1,n) for i, n in enumerate(self.__nodes))   
        return s   

    def toDot(self, queries) :
        # Keep track of mutually disjunctive nodes.
        clusters = defaultdict(list)
        
        # Keep a list of introduced not nodes to prevent duplicates.
        negative = set([])
        
        s = 'digraph GP {\n'
        for index, node in enumerate(self.__nodes) :
            index += 1
            nodetype = type(node).__name__
            
            if nodetype == 'conj' :
                s += '%s [label="AND", shape="box", style="filled", fillcolor="white"];\n' % (index)
                for c in node.children :
                    #c = self._deref(c)
                    if c < 0 and not c in negative :
                        s += '%s [label="NOT"];\n' % (c)
                        s += '%s -> %s;\n' % (c,-c)
                        negative.add(c)
                    if c != 0 :
                        s += '%s -> %s;\n' % (index,c)
            elif nodetype == 'disj' :
                s += '%s [label="OR", shape="diamond", style="filled", fillcolor="white"];\n' % (index)
                for c in node.children :
                    #c = self._deref(c)
                    if c < 0 and not c in negative :
                        s += '%s [label="NOT"];\n' % (c)
                        s += '%s -> %s;\n' % (c,-c)
                        negative.add(c)
                    if c != 0 :
                        s += '%s -> %s;\n' % (index,c)
            elif nodetype == 'atom' :
                if node.group == None :                
                    s += '%s [label="%s", shape="circle", style="filled", fillcolor="white"];\n' % (index, node.probability)
                            #, node.functor, ', '.join(map(str,node.args)))
                else :
                    clusters[node.group].append('%s [ shape="circle", label="%s", style="filled", fillcolor="white" ];\n' % (index, node.probability))
            else :
                raise ValueError("Unexpected node type: '%s'" % nodetype)
        
        c = 0
        for cluster, text in clusters.items() :
            if len(text) > 1 :
                s += 'subgraph cluster_%s { style="dotted"; color="red"; \n\t%s\n }\n' % (c,'\n\t'.join(text))
            else :
                s += text[0]
            c += 1 
            
        q = 0
        for index, name in queries :
            if index < 0 and not index in negative :
                s += '%s [label="NOT"];\n' % (index)
                s += '%s -> %s;\n' % (index,-index)
                negative.add(index)
            elif index == 0 and not index in negative :
                s += '%s [label="true"];\n' % (index)
            s += 'q_%s [ label="%s", shape="plaintext" ];\n'   % (q, name)
            s += 'q_%s -> %s [style="dotted"];\n'  % (q, index)
            q += 1

        return s + '}'


    ##################################################################################
    ####                             CNF CONVERSION                               ####
    ##################################################################################
    
    def toCNF(self) :
        # Keep track of mutually disjunctive nodes.
        choices = defaultdict(list)
        sums = defaultdict(float)
        
        lines = []
        facts = {}
        for index, node in enumerate(self.__nodes) :
            index += 1
            nodetype = type(node).__name__
            
            if nodetype == 'conj' :
                line = str(index) + ' ' + ' '.join( map( lambda x : str(-(x)), node.children ) ) + ' 0'
                lines.append(line)
                for x in node.children  :
                    lines.append( "%s %s 0" % (-index, x) )
            elif nodetype == 'disj' :
                line = str(-index) + ' ' + ' '.join( map( lambda x : str(x), node.children ) ) + ' 0'
                lines.append(line)
                for x in node.children  :
                    lines.append( "%s %s 0" % (index, -x) )
            elif nodetype == 'atom' :
                if node.group == None :                
                    facts[index] = (node.probability.value, 1.0-node.probability.value)
                else :
                    choices[node.group].append(index)
                    sums[node.group] += node.probability.value
                    facts[index] = (node.probability.value, 1.0)                    
            else :
                raise ValueError("Unexpected node type: '%s'" % nodetype)
            
        atom_count = len(facts)
        for cluster, nodes in choices.items() :
            if len(nodes) > 1 :
                if sums[cluster] < 1.0-1e-6 :
                    facts[atom_count+1] = (1.0 - sums[cluster], 1.0)
                    nodes.append(atom_count+1)
                    atom_count += 1
                for i, a in enumerate(nodes) :
                    for b in nodes[i+1:] :
                         lines.append('-%s -%s 0' % (a,b))
                lines.append(' '.join(map(str,nodes + [0]))) 
        clause_count = len(lines)
    
        return [ 'p cnf %s %s' % (atom_count, clause_count) ] + lines, facts
