import sys
from my_pyswip import registerForeign, Term as SWITerm, Functor as SWIFunctor, Variable as SWIVariable, getFloat, \
    PL_FIRST_CALL, PL_REDO, PL_PRUNED, PL_FA_NONDETERMINISTIC, Atom
from problog.logic import Term
from problog.extern import problog_export
from swip import pyswip_to_term, term_to_pyswip, pyswip_to_number


class swi_problog_export(object):
    """
    Decorator for the problog (based on swi engine) foreign functions

    The arguments are the input and output types
    e.g., ('+term', '+list', '-term') means that the arity of the function is 3,
    two arguments are inputs, the first one is a term and the second one a list,
    and the output will be a term.

    Three symbols are expected: + for the input, - for the output and ? for both
    List of possible types:
        - term
        - list
        - int
        - float
        - str
    """
    prolog = None
    database = None

    def __init__(self, *args, **kwdargs):
        self.problog = swi_problog_export.prolog
        self.database = swi_problog_export.database
        self.arity = len(args)
        self.input_arguments = [a[1:] for a in args if a[0] in ('+', '?')]
        self.output_arguments = [a[1:] for a in args if a[0] in ('-', '?')]

        self.input_ids = {i for i in range(len(args)) if args[i][0] in ('+', '?')}
        self.output_ids = {i for i in range(len(args)) if args[i][0] in ('-', '?')}

    def __call__(self, func, *args, **kwargs):
        def wrapper(*args):
            inputs = self._convert_inputs(*args)
            result = func(*inputs, database=self.database)
            if result:
                return self._convert_outputs(result, *args)
            return False

        if swi_problog_export.prolog is not None:
            registerForeign(wrapper, self.prolog.swipl, name=func.__name__, arity=self.arity)
        return wrapper

    def _convert_inputs(self, *args):
        x = 0
        res = []
        for i in self.input_ids:
            if self.input_arguments[x] == "int":
                res.append(pyswip_to_number(args[i], int))
            elif self.input_arguments[x] == "float":
                res.append(pyswip_to_number(args[i], float))
            else:
                rtype = None
                if self.input_arguments[x] == "list":
                    rtype = list
                res.append(pyswip_to_term(args[i], rtype=rtype))
            x += 1
        return res

    def _convert_outputs(self, result, *args):
        if len(self.output_ids) > 1:
            x = 0
            for i in self.output_ids:
                res = self._convert_output(x, result[x])
                if not self.unify(args[i], res):
                    return False
                x += 1
        else:
            for i in self.output_ids:
                result = self._convert_output(0, result)
                if not self.unify(args[i], result):
                    return False
        return True

    def _convert_output(self, x, result):
        if self.output_arguments[x] in ("int", "float", "str"):
            return result
        rtype = None
        if self.output_arguments[x] == "term":
            rtype = Term
        return term_to_pyswip(result, swi_problog_export.prolog.swipl, rtype)

    @staticmethod
    def unify(term, value):
        if type(term) is SWIVariable:
            term.unify(value)
            return True
        elif type(term) is SWIFunctor:
            if term.arity == value.arity:
                for i in range(term.arity):
                    r = swi_problog_export.unify(term.args[i], value.args[i])
                    if not r:
                        return False
                return True
            else:
                return False
        else:
            return term == value


class swi_problog_export_nondet(swi_problog_export):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_result = None

    def __call__(self, func, *args, **kwargs):
        def wrapper(*args):
            context = args[-1]
            control = self.prolog.swipl.PL_foreign_control(context)
            context = self.prolog.swipl.PL_foreign_context(context)

            if control == PL_FIRST_CALL:
                context = 0
                inputs = self._convert_inputs(*args[:-1])
                self._cached_result = func(*inputs, database=self.database)

            if control != PL_PRUNED:
                return self._unification(context, *args[:-1])

        if swi_problog_export.prolog is not None:
            registerForeign(wrapper, self.prolog.swipl,
                            name=func.__name__, arity=self.arity, flags=PL_FA_NONDETERMINISTIC)
        return wrapper

    def _unification(self, context, *args):
        if self._cached_result and len(self._cached_result) > context:
            if not self._convert_outputs(self._cached_result[context], *args):
                return self._unification(context+1, *args)
            context += 1
            return self.prolog.swipl.PL_retry(context)
        return False

