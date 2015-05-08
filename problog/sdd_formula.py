from __future__ import print_function

import sys, os

from collections import namedtuple, defaultdict
from .formula import LogicDAG, LogicFormula, breakCycles
from .cnf_formula import CNF
from .logic import LogicProgram
from .evaluator import Evaluator, SemiringProbability, Evaluatable, InconsistentEvidenceError
from .core import transform
from .util import Timer

import warnings

try :
    import sdd
except Exception :
    sdd = None
#    warnings.warn('The SDD library could not be found!', RuntimeWarning)


class SDDManager(object):
    """
    Manager for SDDs.
    It wraps around the SDD library and offers some additional methods.
    """

    def __init__(self, varcount=0, auto_gc=True):
        """
        Create a new SDD manager.
        :param varcount: number of initial variables
        :type varcount: int
        :param auto_gc: use automatic garbage collection and minimization
        :type auto_gc: bool
        """
        if varcount is None or varcount == 0:
            varcount = 1
        self.__manager = sdd.sdd_manager_create(varcount, auto_gc)
        self.varcount = varcount

    def get_manager(self):
        """
        Get the underlying sdd manager.
        :return:
        """
        return self.__manager

    def add_variable(self, label=0):
        """
        Add a variable to the manager and return its label.
        :param label: suggested label of the variable
        :type label: int
        :return: label of the new variable
        :rtype: int
        """
        if label == 0 or label > self.varcount:
            sdd.sdd_manager_add_var_after_last(self.__manager)
            self.varcount += 1
            return self.varcount
        else:
            return label

    def literal(self, label):
        """
        Return an SDD node representing a literal.
        :param label: label of the literal
        :type label: int
        :return: SDD node representing the literal
        :rtype: SDDNode
        """
        self.add_variable(label)
        return sdd.sdd_manager_literal(label, self.__manager)

    def is_true(self, node):
        """
        Checks whether the SDD node represents True
        :param node: node to verify
        :type node: SDDNode
        :return: True if the node represents True
        :rtype: bool
        """
        return sdd.sdd_node_is_true(node)

    def is_false(self, node):
        """
        Checks whether the SDD node represents False
        :param node: node to verify
        :type node: SDDNode
        :return: False if the node represents False
        :rtype: bool
        """
        return sdd.sdd_node_is_false(node)

    def conjoin(self, *nodes):
        """
        Create the conjunction of the given nodes.
        :param nodes: nodes to conjoin
        :type: SDDNode
        :return: conjunction of the given nodes
        :rtype: SDDNode
        """
        r = sdd.sdd_manager_true(self.__manager)
        for s in nodes:
            r1 = sdd.sdd_conjoin(r, s, self.__manager)
            sdd.sdd_ref(r1, self.__manager)
            sdd.sdd_deref(r, self.__manager)
            r = r1
        return r

    def disjoin(self, *nodes):
        """
        Create the disjunction of the given nodes.
        :param nodes: nodes to conjoin
        :type: SDDNode
        :return: disjunction of the given nodes
        :rtype: SDDNode
        """
        r = sdd.sdd_manager_false(self.__manager)
        for s in nodes:
            r1 = sdd.sdd_disjoin(r, s, self.__manager)
            sdd.sdd_ref(r1, self.__manager)
            sdd.sdd_deref(r, self.__manager)
            r = r1
        return r

    def negate(self, node):
        """
        Create the negation of the given node.
        :param node: negation of the given node
        :type node: SDDNode
        :return: negation of the given node
        :rtype: SDDNode
        """
        new_sdd = sdd.sdd_negate(node, self.__manager)
        sdd.sdd_ref(new_sdd, self.__manager)
        return new_sdd

    def same(self, node1, node2):
        """
        Checks whether two SDD nodes are equivalent.
        :param node1: first node
        :type: SDDNode
        :param node2: second node
        :type: SDDNode
        :return: True if the given nodes are equivalent, False otherwise.
        :rtype: bool
        """
        # Assumes SDD library always reuses equivalent nodes.
        return int(node1) == int(node2)

    def equiv(self, node1, node2):
        """
        Enforce the equivalence between node1 and node2 in the SDD.
        :param node1:
        :param node2:
        :return:
        """
        not1 = self.negate(node1)
        not2 = self.negate(node2)
        i1 = self.disjoin(not1, node2)
        sdd.sdd_deref(not1, self.__manager)
        i2 = self.disjoin(node1, not2)
        sdd.sdd_deref(not2, self.__manager)
        r = self.conjoin(i1, i2)
        sdd.sdd_deref(i1, self.__manager)
        sdd.sdd_deref(i2, self.__manager)
        return r

    def deref(self, *nodes):
        for node in nodes:
            sdd.sdd_deref(node, self.__manager)

    def to_dot(self, node, filename):
        sdd.sdd_save_as_dot(filename, node)


    def __del__(self):
        sdd.sdd_manager_free(self.__manager)
        self.__manager = None


class SDD(LogicDAG, Evaluatable):
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

    def __init__(self, var_count=None, auto_gc=False, **kwdargs):
        LogicDAG.__init__(self, auto_compact=False)
        if sdd is None:
            raise RuntimeError('The SDD library is not available. Please run the installer.')

        self.auto_gc = auto_gc
        self.sdd_manager = SDDManager(var_count, auto_gc=auto_gc)

    def set_varcount(self, varcount):
        self.sdd_manager = SDDManager(varcount=varcount+1, auto_gc=self.auto_gc)

    ##################################################################################
    ####                        CREATE SDD SPECIFIC NODES                         ####
    ##################################################################################

    def _create_atom(self, identifier, probability, group):
        new_lit = self.getAtomCount()+1
        return self._atom(identifier, probability, group, new_lit)

    def _create_conj(self, children):
        new_sdd = self.sdd_manager.conjoin(*[self._get_sddnode(c) for c in children])
        return self._conj(children, new_sdd)

    def _create_disj(self, children):
        new_sdd = self.sdd_manager.disjoin(*[self._get_sddnode(c) for c in children])
        return self._disj(children, new_sdd)

    ##################################################################################
    ####                         GET SDD SPECIFIC INFO                            ####
    ##################################################################################                
        
    def _get_sddnode(self, index):
        negate = False
        if index < 0:
            index = -index
            negate = True 
        node = self.getNode(index)
        if type(node).__name__ == 'atom':
            # was node.sddlit
            result = self.sdd_manager.literal(index)
        else:
            result = node.sddnode
        if negate:
            return self.sdd_manager.negate(result)
        else:
            return result

    def saveSDDToDot( self, filename, index=None ) :
        if self.sdd_manager is None:
            raise ValueError('The SDD manager is not instantiated.')
        else:
            if index is None :
                sdd.sdd_shared_save_as_dot(filename, self.sdd_manager.get_manager())
            else:
                sdd.sdd_save_as_dot(filename, self._get_sddnode(index))
        
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
    
    def _createEvaluator(self, semiring, weights) :
        if not isinstance(semiring,SemiringProbability) :
            raise ValueError('SDD evaluation currently only supports probabilities!')
        return SDDEvaluator(self, semiring, weights)
        
    @classmethod
    def is_available(cls) :
        return sdd != None
    
            
@transform(LogicDAG, SDD)
def buildSDD( source, destination, **kwdargs):
    with Timer('Compiling SDD'):
        size = len(source)
        destination.set_varcount(size)
        for i, n, t in source:
            if t == 'atom':
                destination.addAtom(n.identifier, n.probability, n.group)
            elif t == 'conj':
                destination.addAnd(n.children)
            elif t == 'disj':
                destination.addOr(n.children)
            else:
                raise TypeError('Unknown node type')
                
        for name, node, label in source.getNamesWithLabel():
            destination.addName(name, node, label)
        
        for c in source.constraints():
            destination.addConstraint(c)

    return destination
        

class SDDEvaluator(Evaluator) :

    def __init__(self, formula, semiring, weights=None) :
        Evaluator.__init__(self, formula, semiring)
        self.__sdd = formula
        self.sdd_manager = formula.sdd_manager
        self.__probs = {}
        self.__given_weights = weights

    def getNames(self, label=None) :
        return self.__sdd.getNames(label)
    
    def initialize(self, with_evidence=True) :
        self.__probs.clear()
    
        self.__probs.update(self.__sdd.extractWeights(self.semiring, self.__given_weights))
                            
        if with_evidence :
            for ev in self.iterEvidence() :
                self.setEvidence( abs(ev), ev > 0 )
        
        # evidence sdd => conjoin evidence nodes 
            
    def propagate(self) :
        self.initialize()
        
    def evaluate(self, node):
        if node == 0:
            return self.semiring.one()
        elif node is None:
            return self.semiring.zero()

        query_node_sdd = self.sdd_manager.equiv(self.sdd_manager.literal(node), self.__sdd._get_sddnode(node))
        evidence_sdd = self.sdd_manager.conjoin(*[self.__sdd._get_sddnode(ev) for ev in self.iterEvidence() if self.__sdd.isCompound(abs(ev))])
        rule_sdds = []
        for c in self.__sdd.constraints():
            for rule in c.encodeCNF():
                rule_sdds.append(self.sdd_manager.disjoin(*[self.__sdd._get_sddnode(r) for r in rule]))
        query_sdd = self.sdd_manager.conjoin(query_node_sdd, evidence_sdd, *rule_sdds)

        self.sdd_manager.deref(query_node_sdd)
        self.sdd_manager.deref(evidence_sdd)
        self.sdd_manager.deref(*rule_sdds)

        if self.sdd_manager.is_false(query_sdd):
            raise InconsistentEvidenceError()

        if self.sdd_manager.is_true(query_sdd):
            if node < 0:
                return self.__probs[-node][1]
            else:
                return self.__probs[node][0]
                
        else :
            logspace = 0
            if self.semiring.isLogspace():
                logspace = 1
            wmc_manager = sdd.wmc_manager_new(query_sdd, logspace, self.sdd_manager.get_manager())
            for i, n in enumerate(sorted(self.__probs)):
                i += 1
                pos, neg = self.__probs[n]
                sdd.wmc_set_literal_weight(n, pos, wmc_manager)   # Set positive literal weight
                sdd.wmc_set_literal_weight(-n, neg, wmc_manager)  # Set negative literal weight
            Z = sdd.wmc_propagate(wmc_manager)
            result = sdd.wmc_literal_pr(node, wmc_manager)
            sdd.wmc_manager_free(wmc_manager)
            return result
            
    def setEvidence(self, index, value ) :
        pos = self.semiring.one()
        neg = self.semiring.zero()
        if value :
            self.setWeight( index, pos, neg )
        else :
            self.setWeight( index, neg, pos )
            
    def setWeight(self, index, pos, neg) :
        self.__probs[index] = (pos, neg)

#from .sdd_formula_alt import SDDtp as SDD
