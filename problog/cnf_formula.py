class CNF(object) :
    """A logic formula in Conjunctive Normal Form.
    
    This class does not derive from LogicFormula. (Although it could.)
    
    """
    
    def __init__(self) :
        self.__names = []
        self.__constraints = []
        self.__weights = []
        
        self.__lines = []

    def getNamesWithLabel(self) :
        return self.__names

    def constraints(self) :
        return self.__constraints
        
    def getWeights(self) :
        return self.__weights
                
    @classmethod
    def createFrom(cls, formula, **extra) :
        cnf = CNF()
        
        lines = []
        for index, node in enumerate(formula) :
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
                pass
            else :
                raise ValueError("Unexpected node type: '%s'" % nodetype)
            
        for c in formula.constraints() :
            for l in c.encodeCNF() :
                lines.append(' '.join(map(str,l)) + ' 0')
        
        clause_count = len(lines)
        atom_count = len(formula)
        cnf.__lines = [ 'p cnf %s %s' % (atom_count, clause_count) ] + lines
        cnf.__names = formula.getNamesWithLabel()
        cnf.__constraints = formula.constraints()
        cnf.__weights = formula.getWeights()
        return cnf
        
    def toDimacs(self) :
        return '\n'.join( self.__lines )
        
# CNFFile -> read CNF
# CNF -> CNFFile write toDimacs

class CNFFormula(LogicDAG) :
    
    def __init__(self) :
        LogicDAG.__init__(auto_compact=False)
        
    def __iter__(self) :
        for n in LogicDAG.__iter__(self) :
            yield n
        yield self._create_conj( tuple(range(self.getAtomCount()+1, len(self)+1) ) )
    
    def addAnd(self, content) :
        raise TypeError('This data structure does not support conjunctions.')

class CNFFile(LogicBase) :
    
    # TODO add read functionality???
    
    def __init__(filename=None, readonly=True) :
        self.filename = filename
        self.readonly = readonly
        
        if filename == None :
            self.filename = tempfile.mkstemp('.cnf')[1] :
            self.readonly = False
        else :
            self.filename = filename
            self.readonly = readonly
            
        self.__atom_count = 0
        self.__lines = []
        self.__constraints = []
                        
    def addAtom(self, *args) :
        self.__atom_count += 1
    
    def addAnd(self, content) :
        raise TypeError('This data structure does not support conjunctions.')
                
    def addOr(self, content) :
        self.__lines.append( ' '.join(map(str, content)) + ' 0\n' )
        
    def addNot(self, content) :
        return -content
        
    def addConstraint(self, constraint) :
        for l in constraint.encodeCNF() :
            lines.append(' '.join(map(str,l)) + ' 0')
        
    def constraints(self) :        
        return self.__constraints
            
    def ready(self) :
        if self.readonly :
            raise TypeError('This data structure is read only.')
        with open(self.filename, 'w') as f :
            f.write('p cnf %s %s\n' % (self.__atom_count, len(self.__lines)))
            f.write('\n'.join(self.__lines))

        
def clarks_completion( source, destination ) :
    # Source if an acyclic formula (LogicDAG)
    # Destination is another LogicDAG (in CNF form)
    
    # Every node in original gets a literal
    num_atoms = len(source)
    
    # Add atoms
    for i in range(0, num_atoms) :
        destination.addAtom( (i+1), True, (i+1) )

    # Complete other nodes
    for index, node, nodetype in source.iterNodes() :
        if nodetype == 'conj' :
            destination.addOr( (index,) + tuple( map( lambda x : destination.addNot(x), node.children ) ) )
            for x in node.children  :
                destination.addOr( destination.addNot(index), x )
        elif nodetype == 'disj' :
            destination.addOr( (destination.addNot(index),) + tuple( node.children ) )
            for x in node.children  :
                destination.addOr( index, destination.addNot(x) )
        elif nodetype == 'atom' :
            pass
        else :
            raise ValueError("Unexpected node type: '%s'" % nodetype)
        
    for c in formula.constraints() :
        destination.addConstraint(c)
    
    # TODO copy names and weights
    # cnf.__names = formula.getNamesWithLabel()
    # cnf.__weights = formula.getWeights()
        
    destination.ready()
    return destination
    