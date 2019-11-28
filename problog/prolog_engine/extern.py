import sys
from pyswip import registerForeign, Term as SWITerm, getFloat
from problog.logic import Term
from swip import pyswip_to_term, term_to_pyswip


class swi_problog_export(object):

    def __init__(self, *args, **kwdargs):
        self.arity = len(args)
        self.input_arguments = [a[1:] for a in args if a[0] == '+']
        self.output_arguments = [a[1:] for a in args if a[0] == '-']

        self.input_ids = {i for i in range(len(args)) if args[i][0] == '+'}
        self.output_ids = {i for i in range(len(args)) if args[i][0] == '-'}

    def __call__(self, func, *args, **kwargs):
        def wrapper(*args):
            inputs = self.__convert_inputs(*args)
            result = func(*inputs)
            self.__convert_outputs(result, *args)
            return True

        registerForeign(wrapper, name=func.__name__, arity=self.arity)
        return wrapper

    def __convert_inputs(self, *args):
        # TODO: Implement a cleaner version of the casting of a SWI-predicate to a Problog Term
        return [pyswip_to_term(args[i]) for i in self.input_ids]

    def __convert_outputs(self, result, *args):
        x = 0
        for i in self.output_ids:
            args[i].unify(term_to_pyswip(result[x]))
            x += 1
