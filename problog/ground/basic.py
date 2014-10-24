from __future__ import print_function

from ..utils import TemporaryDirectory, local_path

from ..logic.program import PrologFile




class Grounder(object) :
    
    def __init__(self, env=None) :
        self.env = env        
        
    def ground(self, lp, update=None) :
        """Ground the given logic program.
            :param lp: logic program
            :type lp: :class:`.LogicProgram`
            :param update: ground program to update (incremental grounding)
            :type update: :class:`.GroundProgram`
            :returns: grounded version of the given logic program
            :rtype: :class:`.GroundProgram`
        """
        raise NotImplementedError("Grounder.ground is an abstract method." )
        
        
# Taken and modified from ProbFOIL
class YapGrounder(Grounder) :
    
    def __init__(self, env=None) :
        Grounder.__init__(self, env)
    
    def ground(self, lp, update=None) :
        # This grounder requires a physical Prolog file.
        lp = PrologFile.createFrom(lp)
        
        # Call the grounder.
        grounder_result, qr = self._call_grounder(lp.filename)
        
        # qr are queries (list)
        
        # Initialize a new ground program if none was given.
        if update == None : update = GroundProgram()
        
        # Integrate result into the ground program
        #res = update.integrate(grounder_result)

        # Return the result
        return update
    
    def _call_grounder(self, in_file) : 
        PROBLOG_GROUNDER = local_path('ground/ground_compact.pl')
        
        # Create a temporary directory that is removed when the block exits.
        with TemporaryDirectory(tmpdir='/tmp/problog/') as tmp :
                
            # 2) Call yap to do the actual grounding
            ground_program = tmp.abspath('problog.ground')
        
            queries = tmp.abspath('problog.queries')
        
            evidence = tmp.abspath('problog.evidence')
                
            import subprocess
        
            try :
                output = subprocess.check_output(['yap', "-L", PROBLOG_GROUNDER , '--', in_file, ground_program, evidence, queries ])
                with open(queries) as f :
                    qr = [ line.strip() for line in f ]
                    
                self._build_grounding(ground_program)
                
                return None, None    
               # return self._read_grounding(ground_program), qr
            except subprocess.CalledProcessError :
                print ('Error during grounding', file=sys.stderr)
                return [], []
    
    def _read_grounding(self, filename) :
        lines = []
        with open(filename,'r') as f :
            for line in f :
                line = line.strip().split('|', 1)
                name = None
            
                if len(line) > 1 : name = line[1].strip()
                line = line[0].split()
                line_id = int(line[0])
                line_type = line[1].lower()
            
                while line_id >= len(lines) :
                    lines.append( (None,[],None) )
                if line_type == 'fact' :
                    line_content = float(line[2])
                else :
                    line_content = lines[line_id][1] + [(line_type, line[2:])]
                    line_type = 'or'
                
                lines[line_id] = ( line_type, line_content, name )
        return lines
        
    def _build_grounding(self, filename) :
        # line = 'line_type', 'line_content', 'name'
        
        
        for line in open(filename) :
            content, name = line.strip().split(' | ')
            content = content.split()
            
            line_index = content[0]
            line_type = content[1]
            line_content = content[2:]
        
            
            
        
        pass
        
        

# Grounder with constraints


# Taken and modified from from ProbFOIL
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
        
    def getFact(self, name) :
        return self.__fact_names.get(name, None)
                
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
            #self.__nodes_by_content[ key ] = node_id
        return node_id
        
    def _addNode(self, nodetype, content) :
        node_id = len(self) + 1
        self.__nodes.append( (nodetype, content) )
        return node_id
        
    def getNode(self, index) :
        assert (index != None and index > 0)
        if index <= self.__offset :
            return self.__parent.getNode(index)
        else :
            return self.__nodes[index-self.__offset-1]
                    
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
        
        print (ln_to_ni)
        
        debg = False
        if line_num != None :
            node_id = ln_to_ni[line_num]
            if node_id != '?' : return node_id
            ln_to_ni[line_num] = 'X'
        
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
        return '\n'.join('%s: %s' % (i+1,n) for i, n in enumerate(self.__nodes))   
    
