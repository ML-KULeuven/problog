from __future__ import print_function

import tempfile, os, sys, subprocess
from collections import defaultdict

from . import system_info
from .evaluator import Evaluator, SemiringProbability
from .formula import LogicDAG
from .logic import LogicProgram, LogicBase
from .cnf_formula import CNF
from .interface import ground
from .core import transform

class NNF(LogicDAG) :
    
    def __init__(self) :
        LogicDAG.__init__(self, auto_compact=False)
                                    
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


class NNFFile(LogicBase) :
    
    def __init__(self, filename=None, readonly=True) :
        """Create a new NNF file, or read an existing one."""
        
        if filename == None :
            self.filename = tempfile.mkstemp('.nnf')[1]
            self.readonly = False
        else :
            self.filename = filename
            self.readonly = readonly
        
    def load(self) :
        pass

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


class Compiler(object) :
    
    __compilers = {}
    
    @classmethod
    def getDefault(cls) :
        if system_info.get('c2d', False) :
            return _compile_with_c2d
        else :
            return _compile_with_dsharp
    
    @classmethod
    def get(cls, name) :
        result = cls.__compilers.get(name)
        if result == None : result = cls.getDefault()
        return result
        
    @classmethod
    def add(cls, name, func) :
        cls.__compilers[name] = func


if system_info.get('c2d', False) :
    @transform(CNF, NNF)
    def _compile_with_c2d( cnf, nnf=None ) :
        cnf_file = tempfile.mkstemp('.cnf')[1]
        nnf_file = cnf_file + '.nnf'
        cmd = ['cnf2dDNNF', '-dt_method', '0', '-smooth_all', '-reduce', '-visualize', '-in', cnf_file ]
        return _compile( cnf, cmd, cnf_file, nnf_file )
    Compiler.add( 'c2d', _compile_with_c2d )


@transform(CNF, NNF)
def _compile_with_dsharp( cnf, nnf=None ) :
    cnf_file = tempfile.mkstemp('.cnf')[1]
    nnf_file = tempfile.mkstemp('.nnf')[1]    
    cmd = ['dsharp', '-Fnnf', nnf_file, '-smoothNNF','-disableAllLits', cnf_file ] #
    return _compile( cnf, cmd, cnf_file, nnf_file )
Compiler.add( 'dsharp', _compile_with_dsharp )


def _compile(cnf, cmd, cnf_file, nnf_file) :
    names = cnf.getNamesWithLabel()
    
    with open(cnf_file, 'w') as f :
        f.write(cnf.toDimacs())

    attempts_left = 10
    success = False
    while attempts_left and not success :
        try :
            with open(os.devnull, 'w') as OUT_NULL :
                subprocess.check_call(cmd, stdout=OUT_NULL)
            success = True
        except subprocess.CalledProcessError as err :
            attempts_left -= 1
            if attempts_left == 0 :
                raise err
    return _load_nnf( nnf_file, cnf)


def _load_nnf(filename, cnf) :
    
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
