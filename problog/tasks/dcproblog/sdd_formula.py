from problog.core import transform, transform_create_as
from problog.dd_formula import build_dd
from problog.formula import LogicFormula
from problog.sdd_formula import SDD
from .dd_formula import DDEvaluatorHAL
from .formula import LogicFormulaHAL


class SDDHAL(SDD):
    def __init__(self, **kwdargs):
        SDD.__init__(self, **kwdargs)
        self.density_values = {}

    def _create_evaluator(self, semiring, weights, **kwargs):
        return DDEvaluatorHAL(self, semiring, weights, **kwargs)

    def to_formula(self, sdds):
        """Extracts a LogicFormula from the SDD."""
        formula = LogicFormulaHAL(keep_order=True)
        formula.density_values = self.density_values
        for n, q, l in self.labeled():
            node = self.get_inode(q)
            node = sdds[n]
            constraints = self.get_constraint_inode()
            nodec = self.get_manager().conjoin(node, constraints)
            i = self._to_formula(formula, nodec, {})
            formula.add_name(n, i, l)
        return formula

    def sdd_functions_to_dot(self, *args, sdds=None, **kwargs):
        if kwargs.get('use_internal'):
            for qn, qi in self.queries():
                filename = mktempfile('.dot')
                self.get_manager().write_to_dot(self.get_inode(qi), filename)
                with open(filename) as f:
                    return f.read()
        else:
            return self.to_formula(sdds).functions_to_dot(*args, **kwargs)


    def _to_formula(self, formula, current_node, cache=None):
        if cache is not None and current_node.id in cache:
            return cache[current_node.id]
        if self.get_manager().is_true(current_node):
            retval = formula.TRUE
        elif self.get_manager().is_false(current_node):
            retval = formula.FALSE
        elif current_node.is_literal():  # it's a literal
            lit = current_node.literal
            at = self.var2atom[abs(lit)]
            node = self.get_node(at)
            if lit < 0:
                retval = -formula.add_atom(-lit, probability=node.probability, \
                name=node.name, group=node.group, cr_extra=False, is_extra=node.is_extra)
            else:
                retval = formula.add_atom(lit, probability=node.probability,\
                name=node.name, group=node.group, cr_extra=False, is_extra=node.is_extra)
        else:  # is decision
            elements = list(current_node.elements())
            primes = [prime for (prime, sub) in elements]
            subs = [sub for (prime, sub) in elements]

            # Formula: (p1^s1) v (p2^s2) v ...
            children = []
            for p, s in zip(primes, subs):
                p_n = self._to_formula(formula, p, cache)
                s_n = self._to_formula(formula, s, cache)
                c_n = formula.add_and((p_n, s_n))
                children.append(c_n)
            retval = formula.add_or(children)
        if cache is not None:
            cache[current_node.id] = retval
        return retval

@transform(LogicFormula, SDDHAL)
def build_sdd(source, destination, **kwdargs):
    result = build_dd(source, destination, **kwdargs)
    return result

transform_create_as(SDDHAL, LogicFormula)
