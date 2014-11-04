from __future__ import print_function

from collections import namedtuple, defaultdict

class LogicFormula(object) :
    """A propositional logic formula consisting of and, or, not and atoms."""
    
    TRUE = 0
    FALSE = None
    
    _atom = namedtuple('atom', ('identifier', 'probability', 'group') )
    _conj = namedtuple('conj', ('children') )
    _disj = namedtuple('disj', ('children') )
    # negation is encoded by using a negative number for the key
    
    def __init__(self) :
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
            node_id = self._add( self._atom( identifier, probability, group ) )
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
                
    def makeAcyclic(self, preserve_tables=False) :
        """Break cycles."""
        
        assert(not preserve_tables)                    
        
        # TODO implement preserve tables
        #   This copies the table information from self._def_nodes and translates all result nodes
        #   This requires all result nodes to be maintained separately (add them to protected).
        #   Problem: how to do this without knowledge about internal structure of the engine. 
        
        
        # Output formula
        output = LogicFormula()
        
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
        
    def makeNNF(self) :
        """Transform into NNF form."""
        pass
        
    def makeCNF(self, completion=True) :
        """Transform into CNF form."""
        pass
        
    def _addCompound(self, nodetype, content, t, f, readonly=True, update=None) :
        """Add a compound term (AND or OR)."""
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
            return self._add( node )
        elif nodetype == 'disj' :
            node = self._disj( content )
            if update != None :
                # If an update key is set, update that node
                return self._update( update, node )
            elif readonly :
                # If the node is readonly, we can try to reuse an existing node.
                return self._add( node )
            else :
                # If node is modifiable, we shouldn't reuse an existing node.
                return self._add( node, reuse=False )
        else :
            raise TypeError("Unexpected node type: '%s'." % nodetype) 
    
    

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
        s += '\n' + str(self.__names)
        return s   

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
        

    ##################################################################################
    ####                             CNF CONVERSION                               ####
    ##################################################################################
    
    def toCNF(self) :
        # TODO: does not work with non-constant probabilities
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
            
        atom_count = len(self.__nodes)
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
            else :
                p = facts[nodes[0]][0]
                facts[nodes[0]] = (p, 1.0-p)
        clause_count = len(lines)
    
        return [ 'p cnf %s %s' % (atom_count, clause_count) ] + lines, facts


class CNF(object) :
    
    def __init__(self, formula) :
        self.lines, self.facts = formula.toCNF()  
        self.names = formula.getNamesWithLabel()
    
    def toDimacs(self) : 
        return '\n'.join( self.lines )
    
class NNFFile(LogicFormula) :
    
    def __init__(self, filename, facts, names) :
        LogicFormula.__init__(self)
        self.filename = filename
        self.facts = facts
        self.__probs = {}
        self.__original_probs = {}
        self.load(filename, names)
        self.__probs = dict(self.__original_probs)
    
    def load(self, filename, names) :
        names_inv = defaultdict(list)
        for name,node,label in names :
            names_inv[node].append((name,label))
        
        with open(filename) as f :
            line2node = {}
            lnum = 0
            for line in f :
                line = line.strip().split()
                if line[0] == 'nnf' :
                    pass
                elif line[0] == 'L' :
                    name = int(line[1])
                    probs = (1.0,1.0)
                    probs = self.facts.get(abs(name), probs)
                    
                    # if name in qn :
                    #     prob = str(prob) + '::' + str(qn[name])
                    # elif -name in qn :
                    #     prob = str(prob) + '::-' + str(qn[-name])
                    node = self.addAtom( abs(name), probs  ) #
                    if name < 0 : node = -node
                    line2node[lnum] = node
                    if abs(name) in names_inv :
                        for actual_name, label in names_inv[abs(name)] :
                            if name < 0 :
                                self.addName(actual_name, -node, label)
                            else :
                                self.addName(actual_name, node, label)
                    self.__original_probs[abs(node)] = probs
                    lnum += 1
                elif line[0] == 'A' :
                    children = map(lambda x : line2node[int(x)] , line[2:])
                    line2node[lnum] = self.addAnd( children )
                    lnum += 1
                elif line[0] == 'O' :
                    children = map(lambda x : line2node[int(x)], line[3:])
                    line2node[lnum] = self.addOr( children )        
                    lnum += 1
                else :
                    print ('Unknown line type')

    def _calculateProbability(self, key) :
        assert(key != 0)
        assert(key != None)
        assert(key > 0) 
        
        node = self._getNode(key)
        ntype = type(node).__name__
        assert(ntype != 'atom')
        
        childprobs = [ self.getProbability(c) for c in node.children ]
        if ntype == 'conj' :
            p = 1.0
            for c in childprobs :
                p *= c
            return p
        elif ntype == 'disj' :
            return sum(childprobs)
        else :
            raise TypeError("Unexpected node type: '%s'." % nodetype)
            
    def getProbability(self, key) :
        if key == 0 :
            return 1.0
        elif key == None :
            return 0.0
        else :
            probs = self.__probs.get(abs(key))
            if probs == None :
                p = self._calculateProbability( abs(key) )
                probs = (p, 1.0-p)
                self.__probs[abs(key)] = probs
            if key < 0 :
                return probs[1]
            else :
                return probs[0]
                                
    def setTrue(self, key) :
        pos = self.getProbability(key)
        self.setProbability( key, pos, 0.0 )

    def setFalse(self, key) :
        neg = self.getProbability(-key)
        self.setProbability( key, 0.0, neg )

    def setRTrue(self, key) :
        pos = 1.0
        self.setProbability( key, pos, 0.0 )

    def setRFalse(self, key) :
        neg = 1.0
        self.setProbability( key, 0.0, neg )

                                        
    def setProbability(self, key, pos, neg=None) :
        if neg == None : neg = 1.0 - pos
        self.__probs[key] = (pos,neg)
    
    def resetProbabilities(self) :
        self.__probs = dict(self.__original_probs)
    
#class 


# class LogicGraph(object) :
#   
#
#

# class LogicProgram(object ):
#     pass
#
#
# class AndOrDAG(object) :
#     # Cycle free
#
#     pass
#
# class CNF(object) :
#     # CNF format
#     pass
#
# class sdDNNF(object) :
#     # DDNNF format
#     pass
#


