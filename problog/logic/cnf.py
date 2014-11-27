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