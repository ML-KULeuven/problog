#! /usr/bin/env python
import sys, os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../' ) )

from problog_cli import main as problog_main

from problog.evaluator import Semiring
from problog.logic import Constant, Term


def is_float(a) :
    return type(a) == Constant or type(a) == float
    
def is_poisson(a) :
    return type(a) == Term and a.functor == 'poisson'
    
class SemiringWithPoisson(Semiring) :
    """Semiring that allows mixing of discrete probability distributions with Poisson distributions.
        
        We currently do not support multiplying Poisson distributions.
        The operations are implemented as follows (f=float, p(a)=Poisson with parameter a, x=either)
            
        Plus:
            0 + x = x
            1 + x = 1
            f + f = f+f
            p(a) + p(b) = p(a+b)
            f + p(a) = Error
        
        Times:
            0 * x = 0
            1 * x = x
            f * f = f*f
            f * p(a) = p(f*a) 
            p(a) * p(b) = Error
         
        Negate:
            -f = 1-f
            -p(a) = 1   This means effects of multiple clauses are additive => score(c1\/c2\/c3) == score(c1)+score(c2)+score(c3)
        
    """
    

    def one(self) :
        return 1.0

    def zero(self) :
        return 0.0
        
    def plus(self, a, b) :
        if is_float(a) and is_float(b):
            return Constant(float(a)+float(b))
        elif is_float(a) and float(a) == 0.0 :
            return b
        elif is_float(b) and float(b) == 0.0 :
            return a
        elif is_float(a) and float(a) == 1.0 :
            return 1.0
        elif is_float(b) and float(b) == 1.0 :
            return 1.0
        elif is_poisson(a) and is_poisson(b) :
            return Term('poisson', float(a.args[0]) + float(b.args[0]))
        else :
            raise RuntimeError("Can't compute this: '%s+%s'" % (a,b) )
            
    def times(self, a, b) :
        if is_float(a) and is_float(b) :
            return Constant(float(a)*float(b))
        elif is_float(a) and float(a) == 0.0 :
            return 0.0
        elif is_float(b) and float(b) == 0.0 :
            return 0.0
        elif is_float(a) and float(a) == 1.0 :
            return b
        elif is_float(b) and float(b) == 1.0 :
            return a
        elif is_float(a) and is_poisson(b) :
            return Term('poisson', float(a)*float(b.args[0]))
        elif is_float(b) and is_poisson(a) :
            return Term('poisson', float(b)*float(a.args[0]))
        else :
            raise RuntimeError("Can't compute this: '%s*%s'" % (a,b) )

    def negate(self, a) :
        if is_float(a) :
            return 1.0 - float(a)
        return 1.0
                
    def value(self, a) :
        return a

    def normalize(self, a, Z) :
        if is_float(Z) :
            if is_float(a) :
                return float(a) / float(Z)
            elif is_poisson(a) :
                return Term('poisson', float(a.args[0])*float(Z))
        raise RuntimeError("Can't compute this: '%s / %s'" % (a,b) )
                

def run(filename) :
    
    semiring = SemiringWithPoisson()
    
    problog_main( filename, semiring=semiring )
    


if __name__ == '__main__' :
    if len(sys.argv) <= 1 :
        filename = os.path.join(os.path.dirname(__file__), 'example_poisson.pl' )
    else :
        filename = sys.argv[1]
    
    run(filename)