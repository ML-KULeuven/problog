from __future__ import print_function

import tempfile, os, sys, subprocess
from collections import defaultdict

from .evaluator import Evaluator, SemiringProbability
from .formula import LogicFormula
from .logic import LogicProgram
from .cnf_formula import CNF
from .interface import ground

class NNF(LogicFormula) :
    
    def __init__(self) :
        LogicFormula.__init__(self, auto_compact=False)

    @classmethod
    def createFrom(cls, formula, **extra) :
        assert( isinstance(formula, LogicProgram) or isinstance(formula, LogicFormula) or isinstance(formula, CNF) )
        if isinstance(formula, LogicProgram) :
            formula = ground(formula)

        # Invariant: formula is CNF or LogicFormula
        if not isinstance(formula, NNF) :
            if not isinstance(formula, CNF) :
                formula = CNF.createFrom(formula)
            # Invariant: formula is CNF
            return cls._compile(formula)
        else :
            # TODO force_copy??
            return formula
                        
    @classmethod
    def _compile(cls, cnf) :
        names = cnf.getNamesWithLabel()
        
        # TODO add alternative compiler support
        cnf_file = tempfile.mkstemp('.cnf')[1]
        with open(cnf_file, 'w') as f :
            f.write(cnf.toDimacs())

        nnf_file = tempfile.mkstemp('.nnf')[1]
        cmd = ['dsharp', '-Fnnf', nnf_file, '-smoothNNF','-disableAllLits', cnf_file ] #

        OUT_NULL = open(os.devnull, 'w')

        attempts_left = 10
        success = False
        while attempts_left and not success :
            try :
                subprocess.check_call(cmd, stdout=OUT_NULL)
                success = True
            except subprocess.CalledProcessError as err :
                #print (err)
                #print ("dsharp crashed, retrying", file=sys.stderr)
                attempts_left -= 1
                if attempts_left == 0 :
                    raise err
        OUT_NULL.close()
        return cls._load_nnf( nnf_file, cnf)
    
    @classmethod
    def _load_nnf(cls, filename, cnf) :
        nnf = NNF()

        weights = cnf.getWeights()
        
        names_inv = defaultdict(list)
        for name,node,label in cnf.getNamesWithLabel() :
            names_inv[node].append((name,label))
        
        with open(filename) as f :
            line2node = {}
            rename = {}
            lnum = 0
            for line in f :
                line = line.strip().split()
                if line[0] == 'nnf' :
                    pass
                elif line[0] == 'L' :
                    name = int(line[1])
                    prob = weights.get(abs(name), True)
                    node = nnf.addAtom( abs(name), prob )
                    rename[abs(name)] = node
                    if name < 0 : node = -node
                    line2node[lnum] = node
                    if abs(name) in names_inv :
                        for actual_name, label in names_inv[abs(name)] :
                            if name < 0 :
                                nnf.addName(actual_name, -node, label)
                            else :
                                nnf.addName(actual_name, node, label)
                    lnum += 1
                elif line[0] == 'A' :
                    children = map(lambda x : line2node[int(x)] , line[2:])
                    line2node[lnum] = nnf.addAnd( children )
                    lnum += 1
                elif line[0] == 'O' :
                    children = map(lambda x : line2node[int(x)], line[3:])
                    line2node[lnum] = nnf.addOr( children )        
                    lnum += 1
                else :
                    print ('Unknown line type')
                    
        for c in cnf.constraints() :
            nnf.addConstraint(c.copy(rename))
                    
        return nnf
        
    def getEvaluator(self, semiring=None) :
        if semiring == None :
            semiring = SemiringProbability()
        
        evaluator = SimpleNNFEvaluator(self, semiring )

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




class SimpleNNFEvaluator(Evaluator) :
    
    def __init__(self, formula, semiring) :
        Evaluator.__init__(self, formula, semiring)
        self.__nnf = formula        
        self.__probs = {}
        
        self.Z = 0
    
    def getNames(self, label=None) :
        return self.__nnf.getNames(label)
        
    def initialize(self) :
        self.__probs.clear()
        
        self.__probs.update(self.__nnf.extractWeights(self.semiring))
                        
        for ev in self.iterEvidence() :
            self.setEvidence( abs(ev), ev > 0 )
            
        self.Z = self.getZ()
                
    def propagate(self) :
        self.initialize()
        
    def getZ(self) :
        result = self.getWeight( len(self.__nnf) )
        return result
        
    def evaluate(self, node) :
        p = self.getWeight(abs(node))
        n = self.getWeight(-abs(node))
        self.setValue(abs(node), (node > 0) )
        result = self.getWeight( len(self.__nnf) )
        self.resetValue(abs(node),p,n)
        return self.semiring.normalize(result,self.Z)
        
    def resetValue(self, index, pos, neg) :
        self.setWeight( index, pos, neg)
            
    def getWeight(self, index) :
        if index == 0 :
            return self.semiring.one()
        elif index == None :
            return self.semiring.zero()
        else :
            pos_neg = self.__probs.get(abs(index))
            if pos_neg == None :
                p = self._calculateWeight( abs(index) )
                pos, neg = (p, self.semiring.negate(p))
            else :
                pos, neg = pos_neg
            if index < 0 :
                return neg
            else :
                return pos
                
    def setWeight(self, index, pos, neg) :
        self.__probs[index] = (pos, neg)
        
    def setEvidence(self, index, value ) :
        pos = self.semiring.one()
        neg = self.semiring.zero()
        if value :
            self.setWeight( index, pos, neg )
        else :
            self.setWeight( index, neg, pos )
            
    def setValue(self, index, value ) :
        if value :
            pos = self.getWeight(index)
            self.setWeight( index, pos, self.semiring.zero() )
        else :
            neg = self.getWeight(-index)
            self.setWeight( index, self.semiring.zero(), neg )

    def _calculateWeight(self, key) :
        assert(key != 0)
        assert(key != None)
        assert(key > 0) 
        
        node = self.__nnf._getNode(key)
        ntype = type(node).__name__
        assert(ntype != 'atom')
        
        childprobs = [ self.getWeight(c) for c in node.children ]
        if ntype == 'conj' :
            p = self.semiring.one()
            for c in childprobs :
                p = self.semiring.times(p,c)
            return p
        elif ntype == 'disj' :
            p = self.semiring.zero()
            for c in childprobs :
                p = self.semiring.plus(p,c)
            return p
        else :
            raise TypeError("Unexpected node type: '%s'." % nodetype)    



    
