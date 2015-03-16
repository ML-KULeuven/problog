from __future__ import print_function

from collections import namedtuple, defaultdict
from .formula import LogicDAG, LogicFormula, breakCycles
from .cnf_formula import CNF
from .logic import LogicProgram
from .evaluator import Evaluator, SemiringProbability, Evaluatable, InconsistentEvidenceError
from .core import transform
from .util import Timer

from .sdd_formula import SDDEvaluator

import warnings

try :
    import sdd
except Exception :
    sdd = None
#    warnings.warn('The SDD library could not be found!', RuntimeWarning)


class SDDtp(LogicFormula, Evaluatable) :
    """Alternative SDD-based LogicFormula for forward reasoning."""
    
    def __init__(self, var_count=None) :
        LogicFormula.__init__(self, auto_compact=False)
        
        self.sdd_nodes = None
        self.sdd_updates = None
        
        if sdd is None :
            raise RuntimeError('The SDD library is not available. Please run the installer.')
        
        self.sdd_manager = None
        self.var_count = var_count
        if var_count != None and var_count != 0 :
            self.sdd_manager = sdd.sdd_manager_create(var_count + 1, 0) # auto-gc & auto-min
            
        self.sdd_nodes_prv = None
    
    def setVarCount(self, var_count) :
        self.var_count = var_count
        if var_count != 0 :
            self.sdd_manager = sdd.sdd_manager_create(var_count + 1, 0) # auto-gc & auto-min
            
    def getVarCount(self) :
        return self.var_count
        
    def __del__(self) :
        if self.sdd_manager != None :
            sdd.sdd_manager_free(self.sdd_manager)
            
    ##################################################################################
    ####                        CREATE SDD SPECIFIC NODES                         ####
    ##################################################################################
    
    def _getSDDNode(self, key) :
        node = self.get_sdd_node(key)
        assert(node!=None)
        return node
    
    def get_sdd_node( self, key ) :
        assert(key != 0)
        assert(key != None)
        
        if key < 0 :
            index = abs(key)-1
            if self.sdd_nodes_prv is None :
                return sdd.sdd_manager_false(self.sdd_manager)
            else :
                return sdd.sdd_negate( self.sdd_nodes_prv[index], self.sdd_manager )
        else :
            index = key-1
            if self.sdd_nodes is None :
                return None
            else : 
                return self.sdd_nodes[index]
        
    def set_sdd_node( self, key, node ) :
        index = key - 1
        if self.sdd_nodes is None :
            self.sdd_nodes = [None] * len(self)
            self.sdd_updates = [False] * len(self)
        self.sdd_nodes[index] = node
        
    def get_sdd_updated(self, key) :
        # TODO if key < 0 : get from previous strata
        assert(key != 0)
        if key < 0 :
            return True
        else :
            index = key-1
            if self.sdd_updates is None :
                return False
            else :
                return self.sdd_updates[index]
        
    def set_sdd_updated(self, key, value) :
        index = key - 1
        if self.sdd_updates is None :
            self.sdd_nodes = [None] * len(self)
            self.sdd_updates = [False] * len(self)
        self.sdd_updates[index] = value
    
    def _sdd_equiv( self, n1, n2 ) :
        m = self.sdd_manager
        i1 = sdd.sdd_disjoin( sdd.sdd_negate(n1,m), n2, m )
        i2 = sdd.sdd_disjoin( sdd.sdd_negate(n2,m), n1, m )
        return sdd.sdd_conjoin( i1, i2, m)
    
    def _sdd_changed(self, old_sdd, new_sdd) :
        if old_sdd == new_sdd :
        #if sdd.sdd_node_is_true(self._sdd_equiv( old_sdd, new_sdd )) : 
            return False
        else :
            return True
        
    def update_sdd(self, index) :
        """Recompute SDD for nodes."""
        result = self._update_sdd(index)
        self.set_sdd_updated(index, result)
        return result
        
    def at_fixpoint(self) :
        return len(self) == 0 or (not self.sdd_updates is None and not ( True in self.sdd_updates ))
        
    def build_sdd(self) :
        stop = False
        while not stop :
            # Run to fixpoint on the current strata
            while not self.at_fixpoint() :
                for k in range(0,len(self)) :
                    self.update_sdd(k+1)
                    
            # Check whether we need to start another strata
            if self.sdd_nodes_prv == self.sdd_nodes :
                stop = True
            else :
                self.sdd_nodes_prv = self.sdd_nodes[:]
                self.sdd_updates = [True] * len(self)
    
    def _update_sdd(self, key) :
        """Recompute SDD for nodes."""
        assert(key > 0)
        
        node = self.getNode(key)
        old_sdd = self.get_sdd_node(key)
        if type(node).__name__ == 'atom' :
            if old_sdd is None :
                new_sdd = sdd.sdd_manager_literal( key, self.sdd_manager )
                self.set_sdd_node( key, new_sdd )
                return True
            else :
                return False
        else :
            children = [ self.get_sdd_node(c) for c in node.children ]
            child_updates = [ self.get_sdd_updated(c) for c in node.children ]
            if type(node).__name__ == 'conj' :
                if None in children :
                    # At least one of the children is not available yet.
                    return False
                elif True in child_updates :
                    # At least one of the children has been modified in the previous iteration.
                    # Q? => can we be smarter if we know that the child has been added instead of been modified
                    new_sdd = children[0]
                    for c in children[1:] :
                        new_sdd = sdd.sdd_conjoin( new_sdd, c, self.sdd_manager )
                    self.set_sdd_node( key, new_sdd )
                    if old_sdd is None :
                        return True
                    else :
                        return self._sdd_changed(old_sdd, new_sdd)
                else :
                    # None of the children have been modified.
                    return False
            elif type(node).__name__ == 'disj' :
                if True in child_updates :
                    # One of the children was updated
                    children = list(filter(None, children)) # Eliminate undefined nodes from list of children
                    new_sdd = children[0]
                    for c in children[1:] :
                        new_sdd = sdd.sdd_disjoin( new_sdd, c, self.sdd_manager )
                    self.set_sdd_node( key, new_sdd )
                    return self._sdd_changed(old_sdd, new_sdd)
                else :
                    # None of the children was updated
                    return False
    
    ##################################################################################
    ####                         GET SDD SPECIFIC INFO                            ####
    ##################################################################################                
        
    def saveSDDToDot( self, filename, index=None ) :
        if index is None :
            sdd.sdd_shared_save_as_dot(filename, self.sdd_manager)
        else :
            sdd.sdd_save_as_dot(filename, self._getSDDNode(index))
        
    ##################################################################################
    ####                          UNSUPPORTED METHODS                             ####
    ##################################################################################                
        
    def _update( self, key, value ) :
        """Replace the node with the given node."""
        raise NotImplementedError('SDD formula does not support node updates.')
        
    def addDisjunct( self, key, component ) :
        """Add a component to the node with the given key."""
        raise NotImplementedError('SDD formula does not support node updates.')        
        
    ##################################################################################
    ####                               EVALUATION                                 ####
    ##################################################################################          
    
    def _createEvaluator(self, semiring, weights) :
        if not isinstance(semiring,SemiringProbability) :
            raise ValueError('SDD evaluation currently only supports probabilities!')
        return SDDEvaluator(self, semiring, weights)
        
    @classmethod
    def is_available(cls) :
        return sdd != None
        
@transform(LogicFormula, SDDtp)
def buildSDD( source, destination ) :
    with Timer('Compiling SDD'):
        size = len(source)
        destination.setVarCount(size)
        for i, n, t in source :
            if t == 'atom' :
                destination.addAtom( n.identifier, n.probability, n.group )
            elif t == 'conj' :
                destination.addAnd( n.children )
            elif t == 'disj' :
                destination.addOr( n.children )
            else :
                raise TypeError('Unknown node type')
                
        for name, node, label in source.getNamesWithLabel() :
            destination.addName(name, node, label)
        
        for c in source.constraints() :
            destination.addConstraint(c)
    destination.build_sdd()
    return destination
    
