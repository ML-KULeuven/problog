
# NNF == and-or graph
# OBDD : 
# d-DNNF : deterministic decomposable NNF
# sd-DNNF : smooth deterministic decomposable NNF
# SDD : structured decomposable + strong determinism

# OBDD < SDD < d-DNNF

# cycle-free: and-or-graph is a DAG 
# deterministic: disjuncts are logically disjoint
# decomposable: conjuncts do not share variables
# smoothness: all disjuncts mention the same set of variables
#   => smoothen => 
#    for each disjunction A = A1 \/ A2 \/ ... \/ An
#       if atoms(Ai) != atoms(A)
#           replace Ai with Ai /\ /\_A in S (A\/-A) where S = atoms(A)-atoms(Ai)       

# c2d => CNF -> d-DNNF
# dsharp => CNF -> sd-DNNF
# sdd => CNF/DNF -> sDD

# DIMACS format for CNF
#
# Command lines start with 'c'
# Problem line: 'p' 'cnf' #vars #clauses
# Clause: a b -c 0  where a,b,c are indexes of literals
# Clause always ends with 0   
#
#   and: self_lit -child_lit1 -child_lit2
#       -self_lit +child_lit1
#       -self_lit +child_lit2
#           
#   or:  -self_lit child_lit1 child_lit2
#        +self_lit -child_lit1
#        +self_lit -child_lit2
#
#
#   OR 1
#       AND 2
#           LIT3
#           LIT4
#       AND 5
#           LIT6
#           LIT7
#
# AND                   TAKE 3,4 = TRUE / 6,7 = FALSE
#   -1 OR 2 OR 5        TRUE
#       1 OR -2         1 = TRUE
#       1 OR -5         TRUE
#   2 OR -3 OR -4       For 2=TRUE => TRUE
#       -2 OR 3         TRUE
#       -2 OR 4         TRUE
#   5 OR -6 OR -7       TRUE
#       -5 OR 6         FALSE => 5 
#       -5 OR 7         FALSE => 5


class NNF(object) :
    
    def isDAG(self) :
        pass
        
    def isDeterministic(self) :
        pass
        
    def isDecomposable(self) :
        pass
        
    def isSmooth(self) :
        pass
                
    def getNode(self) :
        pass
        
    def getUsedFacts(self, index) :
        pass
        
    
# Taken from ProbFOIL
class GroundProgram(object) :
    
    # Invariant: stored nodes do not have TRUE or FALSE in their content.
    
    TRUE = 0
    FALSE = None
    
    def __init__(self, parent=None) :
        if parent :
            self.__offset = len(parent)
        else :
            self.__offset = 0
        self.__parent = parent
        self.clear()
    
    def clear(self) :
        self.__nodes = []
        self.__fact_names = {}
        self.__nodes_by_content = {}
        self.__probabilities = []
        self.__usedfacts = []
        
    def getFact(self, name) :
        return self.__fact_names.get(name, None)
        
    def _getUsedFacts(self, index) :
        if index < 0 :
            return self.__usedfacts[-index-1]
        else :
            return self.__usedfacts[index-1]
        
    def _setUsedFacts(self, index, value) :
        if index < 0 :
            self.__usedfacts[-index-1] = frozenset(value)
        else :
            self.__usedfacts[index-1] = frozenset(value)
        
    def _negate(self, t) :
        if t == self.TRUE :
            return self.FALSE
        elif t == self.FALSE :
            return self.TRUE
        else :
            return -t
            
    def addChoice(self, rule) :
        return self._addNode('choice', rule)
        
    def addFact(self, name, probability) :
        """Add a named fact to the grounding."""
        assert(not name.startswith('pf_'))
        node_id = self.getFact(name)
        if node_id == None : # Fact doesn't exist yet
            node_id = self._addNode( 'fact', (name, probability) )
            self.__fact_names[name] = node_id
            self.setProbability(node_id, probability)
            self._setUsedFacts(node_id,[abs(node_id)])
        return node_id
        
    def addNode(self, nodetype, content) :
        if nodetype == 'or' :
            return self.addOrNode(content)
        elif nodetype == 'and' :
            return self.addAndNode(content)
        else :
            raise Exception("Unknown node type '%s'" % nodetype)
        
    def addOrNode(self, content) :
        """Add an OR node."""
        return self._addCompoundNode('or', content, self.TRUE, self.FALSE)
        
    def addAndNode(self, content) :
        """Add an AND node."""
        return self._addCompoundNode('and', content, self.FALSE, self.TRUE)
        
    def _addCompoundNode(self, nodetype, content, t, f) :
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
        if len(content) == 1 : return content[0]
        
        # Lookup node for reuse
        key = (nodetype, content)
        node_id = self.__nodes_by_content.get(key, None)
        
        if node_id == None :    
            # Node doesn't exist yet
            node_id = self._addNode( *key )
        return node_id
        
    def _addNode(self, nodetype, content) :
        node_id = len(self) + 1
        self.__nodes.append( (nodetype, content) )
        self.__probabilities.append(None)
        self.__usedfacts.append(frozenset([]))
        return node_id
        
    def getNode(self, index) :
        assert (index != None and index > 0)
        if index <= self.__offset :
            return self.__parent.getNode(index)
        else :
            return self.__nodes[index-self.__offset-1]
    
    def calculateProbability(self, nodetype, content) :
        if nodetype == 'or' :
            f = lambda a, b : a*(1-b)
            p = 1
        elif nodetype == 'and' :
            f = lambda a, b : a*b
            p = 1
        for child in content :
            p_c = self.getProbability(child)
            if p_c == None :
                p = None
                break
            else :
                p = f(p,p_c)
        if p != None and nodetype == 'or' :
            p = 1 - p
        return p
        
    def getProbability(self, index) :
        if index == 0 :
            return 1
        elif index == None :
            return 0
        elif index < 0 :
            p = self.getProbability(-index)
            if p == None :
                return None
            else :
                return 1 - p
        else :
            return self.__probabilities[index-1]
    
    def setProbability(self, index, p) :
        #print ('SP', index, p, self.getNode(index))
        if index == 0 or index == None :
            pass
        elif index < 0 :
            self.__probabilities[-index-1] = 1 - p
        else :
            self.__probabilities[index-1] = p
                    
    def integrate(self, lines, rules=None) :
    
        # Dictionary query_name => node_id
        result = {}
        
        ln_to_ni = ['?'] * (len(lines) + 1)   # line number to node id
        line_num = 0
        for line_type, line_content, line_alias in lines[1:] :
            line_num += 1
            node_id = self._integrate_line(line_num, line_type, line_content, line_alias, lines, ln_to_ni, rules)
            if node_id != None :
                result[line_alias] = node_id
        return result
        
    def _integrate_line(self, line_num, line_type, line_content, line_alias, lines, ln_to_ni, rules) :
        # TODO make it work for cycles
        
        debg = False
        if line_num != None :
            node_id = ln_to_ni[line_num]
            if node_id != '?' : return node_id
        
        if line_type == 'fact' :
            if line_content > 1.0 - 1e-10 :
                node_id = 0
            else :
                node_id = self.addFact(line_alias, line_content)
        else :
            # Compound node => process content recursively
            subnodes = []
            for subnode in line_content :
                if type(subnode) == tuple :
                    subnodes.append(self._integrate_line(None, subnode[0], subnode[1], None, lines, ln_to_ni, rules))
                else :
                    subnode_id = int(subnode)
                    neg = subnode_id < 0
                    subnode_id = abs(subnode_id)
                    subnode = lines[subnode_id]
                    tr_subnode = self._integrate_line(subnode_id, subnode[0], subnode[1], subnode[2], lines, ln_to_ni, rules)
                    if neg :
                        tr_subnode = self._negate(tr_subnode)
                    subnodes.append(tr_subnode)
                    
        if line_type == 'or' :
            node_id = self.addOrNode(tuple(subnodes))    
        elif line_type == 'and' :
            node_id = self.addAndNode(tuple(subnodes))    
            
        # Store in translation table
        if line_num != None : ln_to_ni[line_num] = node_id
        
        return node_id
        
    def _selectNodes(self, queries, node_selection) :
        for q in queries :
            node_id = q
            if node_id :
                self._selectNode(abs(node_id), node_selection)
        
    def _selectNode(self, node_id, node_selection) :
        assert(node_id != 0)
        if not node_selection[node_id-1] :
            node_selection[node_id-1] = True
            nodetype, content = self.getNode(node_id)
            
            if nodetype in ('and','or') :
                for subnode in content :
                    if subnode :
                        self._selectNode(abs(subnode), node_selection)
        
    def __len__(self) :
        return len(self.__nodes) + self.__offset
        
    def toCNF(self, queries=None) :
        # if self.hasCycle :
        #     raise NotImplementedError('The dependency graph contains a cycle!')
        
        if queries != None :
            node_selection = [False] * len(self)    # selection table
            self._selectNodes(queries, node_selection)
        else :
            node_selection = [True] * len(self)    # selection table
            
        lines = []
        facts = {}
        for k, sel in enumerate( node_selection ) :
          if sel :
            k += 1
            v = self.getNode(k)
            nodetype, content = v
            
            if nodetype == 'fact' :
                facts[k] = content[1]
            elif nodetype == 'and' :
                line = str(k) + ' ' + ' '.join( map( lambda x : str(-(x)), content ) ) + ' 0'
                lines.append(line)
                for x in content :
                    lines.append( "%s %s 0" % (-k, x) )
            elif nodetype == 'or' :
                line = str(-k) + ' ' + ' '.join( map( lambda x : str(x), content ) ) + ' 0'
                lines.append(line)
                for x in content :
                    lines.append( "%s %s 0" % (k, -x) )
                # lines.append('')
            elif nodetype == 'choice' :
                if content.hasScore() :
                    facts[k] = content.probability
                else :
                    facts[k] = 1.0
            else :
                raise ValueError("Unknown node type!")
                
        atom_count = len(self)
        clause_count = len(lines)
        return [ 'p cnf %s %s' % (atom_count, clause_count) ] + lines, facts
        
    def stats(self) :
        return namedtuple('IndexStats', ('atom_count', 'name_count', 'fact_count' ) )(len(self), 0, len(self.__fact_names))
        
    def __str__(self) :
        return '\n'.join('%s: %s (p=%s)' % (i+1,n, self.__probabilities[i]) for i, n in enumerate(self.__nodes))   
    
