from __future__ import print_function

from .logic import Constant, Term
from .engine import computeFunction

from collections import defaultdict
import subprocess
import sys, os, tempfile

class Semiring(object) :
    
    def one(self) :
        raise NotImplementedError()
    
    def zero(self) :
        raise NotImplementedError()
        
    def plus(self, a, b) :
        raise NotImplementedError()

    def times(self, a, b) :
        raise NotImplementedError()

    def negate(self, a) :
        raise NotImplementedError()

    def value(self, a) :
        raise NotImplementedError()

    def normalize(self, a, Z) :
        raise NotImplementedError()
    
class SemiringProbability(Semiring) :

    def one(self) :
        return 1.0

    def zero(self) :
        return 0.0
        
    def plus(self, a, b) :
        return a + b
        
    def times(self, a, b) :
        return a * b

    def negate(self, a) :
        return 1.0 - a
                
    def value(self, a) :
        if isinstance(a, Constant) :
            return a.value
        elif isinstance(a, Term) :
            return computeFunction(a.functor, a.args, None).value
        else :
            return a

    def normalize(self, a, Z) :
        return a/Z

class SemiringSymbolic(Semiring) :
    
    def one(self) :
        return "1"
    
    def zero(self) :
        return "0"
        
    def plus(self, a, b) :
        if a == "0" :
            return b
        elif b == "0" :
            return a
        else :
            return "(%s + %s)" % (a,b)

    def times(self, a, b) :
        if a == "0" or b == "0" :
            return "0"
        elif a == "1" :
            return b
        elif b == "1" :
            return a
        else :
            return "%s*%s" % (a,b)

    def negate(self, a) :
        if a == "0" :
            return "1"
        elif a == "1" :
            return "0"
        else :
            return "(1-%s)" % a 

    def value(self, a) :
        return str(a)
        
    def normalize(self, a, Z) :
        if Z == "1" :
            return a
        else :
            return "%s / %s" % (a,Z)


class Evaluator(object) :

    def __init__(self, formula, semiring) :
        self.formula = formula
        self.__semiring = semiring
        
        self.__evidence = []
        
    def _get_semiring(self) : return self.__semiring
    semiring = property(_get_semiring)
        
    def initialize(self) :
        raise NotImplementedError('Evaluator.initialize() is an abstract method.')
        
    def propagate(self) :
        raise NotImplementedError('Evaluator.propagate() is an abstract method.')
        
    def evaluate(self, index) :
        """Compute the value of the given node."""
        raise NotImplementedError('Evaluator.evaluate() is an abstract method.')
        
    def getZ(self) :
        """Get the normalization constant."""
        raise NotImplementedError('Evaluator.getZ() is an abstract method.')
        
    def addEvidence(self, node) :
        """Add evidence"""
        self.__evidence.append(node)
        
    def clearEvidence(self) :
        self.__evidence = []
        
    def iterEvidence(self) :
        return iter(self.__evidence)            
            

