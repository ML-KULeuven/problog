from __future__ import print_function

import sys, os

from collections import namedtuple, defaultdict
from .formula import LogicDAG, LogicFormula, breakCycles
from .cnf_formula import CNF
from .logic import LogicProgram
from .interface import ground
from .evaluator import Evaluator, SemiringProbability
from .core import transform

import warnings

try :
    import sdd
except ImportError :
    sdd = None
    warnings.warn('The SDD library could not be found!', RuntimeWarning)

class SDD(LogicDAG) :
    """A propositional logic formula consisting of and, or, not and atoms represented as an SDD.

    This class has two restrictions with respect to the default LogicFormula:

        * The number of atoms in the SDD should be known at construction time.
        * It does not support updatable nodes.

    This means that this class can not be used directly during grounding.
    It can be used as a target for the ``makeAcyclic`` method.
    """

    _atom = namedtuple('atom', ('identifier', 'probability', 'group', 'sddlit') )
    _conj = namedtuple('conj', ('children', 'sddnode') )
    _disj = namedtuple('disj', ('children', 'sddnode') )
    # negation is encoded by using a negative number for the key

    def __init__(self, var_count=None) :        
        LogicDAG.__init__(self, auto_compact=False)
        self.sdd_manager = None
        if var_count != None :
            self.sdd_manager = sdd.sdd_manager_create(var_count, 0) # auto-gc & auto-min
    
    def setVarCount(self, var_count) :
        self.sdd_manager = sdd.sdd_manager_create(var_count, 0) # auto-gc & auto-min

    def __del__(self) :
        sdd.sdd_manager_free(self.sdd_manager)

    ##################################################################################
    ####                        CREATE SDD SPECIFIC NODES                         ####
    ##################################################################################

    def _create_atom( self, identifier, probability, group ) :
        new_lit = self.getAtomCount()+1
        return self._atom( identifier, probability, group, new_lit )

    def _create_conj( self, children ) :
        children_original = children[:]
        c1 = self._getSDDNode( children[0] )
        children = children[1:]
        while children :
            c2 = self._getSDDNode( children[0] )
            children = children[1:]
            c1 = sdd.sdd_conjoin(c1, c2, self.sdd_manager)
        return self._conj( children_original, c1 )

    def _create_disj( self, children ) :
        children_original = children[:]
        c1 = self._getSDDNode( children[0] )
        children = children[1:]
        while children :
            c2 = self._getSDDNode( children[0] )
            children = children[1:]
            c1 = sdd.sdd_disjoin(c1, c2, self.sdd_manager)
        return self._disj( children_original, c1 )

    ##################################################################################
    ####                         GET SDD SPECIFIC INFO                            ####
    ##################################################################################                
        
    def _getSDDNode( self, index ) :
        negate = False
        if index < 0 :
            index = -index
            negate = True 
        node = self._getNode(index)
        if type(node).__name__ == 'atom' :
            # was node.sddlit
            result = sdd.sdd_manager_literal( index, self.sdd_manager )
        else :
            result = node.sddnode
        if negate :
            return sdd.sdd_negate( result, self.sdd_manager )
        else :
            return result

    def saveSDDToDot( self, filename, index=None ) :
        if index == None :
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


    # @classmethod
    # def createFrom(cls, formula, **extra) :
    #     # TODO support formula CNF
    #     assert( isinstance(formula, LogicProgram) or isinstance(formula, LogicFormula) or isinstance(formula, CNF) )
    #     if isinstance(formula, LogicProgram) :
    #         formula = ground(formula)
    #
    #     # Invariant: formula is CNF or LogicFormula
    #     if not isinstance(formula, SDD) :
    #         size = len(formula)
    #         # size = formula.getAtomCount()
    #         sdd = SDD(size)
    #         formula = formula.makeAcyclic(output=sdd)
    #
    #         for c in formula.constraints() :
    #             sdd.addConstraint(c)
    #         return sdd
    #
    #     else :
    #         # TODO force_copy??
    #         return formula


    ##################################################################################
    ####                               EVALUATION                                 ####
    ##################################################################################                

    def getEvaluator(self, semiring=None) :
        if semiring == None :
            semiring = SemiringProbability()
    
        if (type(semiring) != SemiringProbability) :
            raise ValueError('SDD evaluation currently only supports probabilities!')
    
        evaluator = SDDEvaluator(self, semiring )

        for n_ev, node_ev in evaluator.getNames('evidence') :
            evaluator.addEvidence( node_ev )
    
        for n_ev, node_ev in evaluator.getNames('-evidence') :
            evaluator.addEvidence( -node_ev )

        evaluator.propagate()
        return evaluator
    
    def evaluate(self, index=None, semiring=None) :
        evaluator = self.getEvaluator(semiring)
    
        if index == None :
            result = {}
            # Probability of query given evidence
            for name, node in evaluator.getNames('query') :
                w = evaluator.evaluate(node)    
                if w < 1e-6 : 
                    result[name] = 0.0
                else :
                    result[name] = w
            return result
        else :
            return evaluator.evaluate(node)
            
@transform(LogicDAG, SDD)
def buildSDD( source, destination ) :
    size = len(source)
    destination.setVarCount(size)
    for i, n, t in source.iterNodes() :
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
    return destination
        

class SDDEvaluator(Evaluator) :

    def __init__(self, formula, semiring) :
        Evaluator.__init__(self, formula, semiring)
        self.__sdd = formula
        self.sdd_manager = formula.sdd_manager
        self.__probs = {}

    def getNames(self, label=None) :
        return self.__sdd.getNames(label)
    
    def initialize(self) :
        self.__probs.clear()
    
        self.__probs.update(self.__sdd.extractWeights(self.semiring))
                    
        for ev in self.iterEvidence() :
            self.setEvidence( abs(ev), ev > 0 )
        
        # evidence sdd => conjoin evidence nodes 
            
    def propagate(self) :
        self.initialize()
            
    def _sdd_disjoin( self, r1, *r ) :
        if not r :
            return self.__sdd._getSDDNode(r1)
        else :
            rec = self._sdd_disjoin(*r)
            return sdd.sdd_disjoin( self.__sdd._getSDDNode(r1), rec, self.sdd_manager )
    
    def _sdd_equiv( self, n1, n2 ) :
        m = self.sdd_manager
        i1 = sdd.sdd_disjoin( sdd.sdd_negate(n1,m), n2, m )
        i2 = sdd.sdd_disjoin( sdd.sdd_negate(n2,m), n1, m )
        return sdd.sdd_conjoin( i1, i2, m)
    
    def evaluate(self, node) :
        # TODO make sure this works for negative query nodes        
        m = self.sdd_manager 

        # TODO build evidence and constraints before
        evidence_sdd = sdd.sdd_manager_true( m )
        for ev in self.iterEvidence() :
            evidence_sdd = sdd.sdd_conjoin( evidence_sdd, self.__sdd._getSDDNode(ev), m )
        
        for c in self.__sdd.constraints() :
            for rule in c.encodeCNF() :
                evidence_sdd = sdd.sdd_conjoin( evidence_sdd, self._sdd_disjoin( *rule ), m )
        
    
        query_sdd = self._sdd_equiv( sdd.sdd_manager_literal(node, self.sdd_manager), self.__sdd._getSDDNode(node))

        query_sdd = sdd.sdd_conjoin( query_sdd, evidence_sdd, self.sdd_manager )

    
        # TODO this is probably not always correct:
        if sdd.sdd_node_is_true( query_sdd ) :
            return self.__probs[ node ][0]
        else :
            wmc_manager = sdd.wmc_manager_new( query_sdd , 0, self.sdd_manager )

            for i, n in enumerate(sorted(self.__probs)) :
                i = i + 1
                pos, neg = self.__probs[n]
                sdd.wmc_set_literal_weight( n, pos, wmc_manager )   # Set positive literal weight
                sdd.wmc_set_literal_weight( -n, neg, wmc_manager )  # Set negative literal weight
            Z = sdd.wmc_propagate( wmc_manager )
            result = sdd.wmc_literal_pr( node, wmc_manager )
            sdd.wmc_manager_free(wmc_manager)
            return result
            
    def setEvidence(self, index, value ) :
        pos = self.semiring.one()
        neg = self.semiring.zero()
        if value :
            self.setWeight( index, pos, neg )
        else :
            self.setWeight( index, neg, pos )


    # def getWeight(self, index) :
    #     if index == 0 :
    #         return self.semiring.one()
    #     elif index == None :
    #         return self.semiring.zero()
    #     else :
    #         pos_neg = self.__probs.get(abs(index))
    #         if pos_neg == None :
    #             p = self._calculateWeight( abs(index) )
    #             pos, neg = (p, self.semiring.negate(p))
    #         else :
    #             pos, neg = pos_neg
    #         if index < 0 :
    #             return neg
    #         else :
    #             return pos
            
    def setWeight(self, index, pos, neg) :
        self.__probs[index] = (pos, neg)

    
