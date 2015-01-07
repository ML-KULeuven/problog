from __future__ import print_function

from .formula import LogicDAG

from .core import ProbLogObject, transform
from .util import Timer

import tempfile

class CNF(ProbLogObject) :
    """A logic formula in Conjunctive Normal Form.
    
    This class does not derive from LogicFormula. (Although it could.)
    
    """
    
    def __init__(self) :
        self.__atom_count = 0
        self.__lines = []
        self.__constraints = []
        
        self.__names = []
        self.__weights = []
                        
    def addAtom(self, *args) :
        self.__atom_count += 1
    
    def addAnd(self, content) :
        raise TypeError('This data structure does not support conjunctions.')
                
    def addOr(self, content) :
        self.__lines.append( ' '.join(map(str, content)) + ' 0' )
        
    def addNot(self, content) :
        return -content
        
    def addConstraint(self, constraint) :
        self.__constraints.append(constraint)        
        for l in constraint.encodeCNF() :
            self.__lines.append(' '.join(map(str,l)) + ' 0')
        
    def constraints(self) :        
        return self.__constraints

    def getNamesWithLabel(self) :
        return self.__names
        
    def setNamesWithLabel(self, names) :
        self.__names = names

    def getWeights(self) :
        return self.__weights
        
    def setWeights(self, weights) :
        self.__weights = weights
            
    def ready(self) :
        pass

    def toDimacs(self) :
        return 'p cnf %s %s\n' % (self.__atom_count, len(self.__lines)) + '\n'.join( self.__lines )
        
    def getAtomCount(self) :
        return self.__atom_count
        
    def isTrivial(self) :
        return len(self.__lines) == 0
        
# CNFFile -> read CNF
# CNF -> CNFFile write toDimacs

class CNFFormula(LogicDAG) :
    """A CNF stored in memory."""
    
    def __init__(self) :
        LogicDAG.__init__(auto_compact=False)
        
    def __iter__(self) :
        for n in LogicDAG.__iter__(self) :
            yield n
        yield self._create_conj( tuple(range(self.getAtomCount()+1, len(self)+1) ) )
    
    def addAnd(self, content) :
        raise TypeError('This data structure does not support conjunctions.')

class CNFFile(CNF) :
    """A CNF stored in a file."""
    
    # TODO add read functionality???
    
    def __init__(self, filename=None, readonly=True) :
        self.filename = filename
        self.readonly = readonly
        
        if filename == None :
            self.filename = tempfile.mkstemp('.cnf')[1]
            self.readonly = False
        else :
            self.filename = filename
            self.readonly = readonly
            
    def ready(self) :
        if self.readonly :
            raise TypeError('This data structure is read only.')
        with open(self.filename, 'w') as f :
            f.write('p cnf %s %s\n' % (self.__atom_count, len(self.__lines)))
            f.write('\n'.join(self.__lines))

@transform(LogicDAG, CNF) 
def clarks_completion( source, destination ) :    
    with Timer('Clark\'s completion'):
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
                    destination.addOr( (destination.addNot(index), x) )
            elif nodetype == 'disj' :
                destination.addOr( (destination.addNot(index),) + tuple( node.children ) )
                for x in node.children  :
                    destination.addOr( (index, destination.addNot(x)) )
            elif nodetype == 'atom' :
                pass
            else :
                raise ValueError("Unexpected node type: '%s'" % nodetype)
            
        for c in source.constraints() :
            destination.addConstraint(c)
        
        destination.setNamesWithLabel(source.getNamesWithLabel())
        destination.setWeights(source.getWeights())
                
        destination.ready()
        return destination

