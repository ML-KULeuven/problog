from problog.program import DefaultPrologFactory
from problog.logic import Term

from .logic import distributions, Distribution


class DCPrologFactory(DefaultPrologFactory):
    def __init__(self, identifier=0):
        DefaultPrologFactory.__init__(self, identifier=identifier)

    def build_function(self, functor, arguments, location=None, **extra):
        if functor=="'~'":
            #Maybe add arithmetic for distributional head here
            rv, distribution = arguments
            print(distribution.functor)
            assert distribution.functor in distributions
            distribution = Distribution(distribution.functor, *distribution.args)
            return Term(functor, rv, distribution, location=(self.loc_id, location), **extra)
        else:
            return Term(functor, *arguments, location=(self.loc_id, location), **extra)
