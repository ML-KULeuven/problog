import psipy
from .algebra import Algebra, BaseS

str2distribution = {
    "delta" : psipy.delta_pdf,
    "normal" : psipy.normal_pdf,
    "normalInd" : psipy.normalInd_pdf,
    "uniform" : psipy.uniform_pdf,
    "beta" : psipy.beta_pdf,
    "poisson" : psipy.poisson_pdf
}

class S(BaseS):
    def __init__(self, psi_symbol, variables=set()):
        BaseS.__init__(self, psi_symbol, variables)

    def __add__(self, other):
        s = S(
            psipy.simplify(psipy.add(self.value,other.value)),
            variables = self.variables | other.variables
        )
        return s
    def __sub__(self, other):
        s = S(
            psipy.simplify(psipy.sub(self.value,other.value)),
            variables = self.variables | other.variables
        )
        return s
    def __mul__(self, other):
        s = S(
            psipy.simplify(psipy.mul(self.value,other.value)),
            variables = self.variables | other.variables
        )
        return s
    def __truediv__(self, other):
        s = S(
            psipy.simplify(psipy.div(self.value,other.value)),
            variables = self.variables | other.variables
        )
        return s
    def __pow__(self, other):
        s = S(
            psipy.simplify(psipy.pow(self.value,other.value)),
            variables = self.variables | other.variables
        )
        return s

    def exp(self):
        return S(psipy.exp(self.value), variables=self.variables)
    def sigmoid(self):
        return S(psipy.sig(self.value), variables=self.variables)

    def lt(self, other):
        s = S(
            psipy.less(self.value,other.value),
            variables = self.variables | other.variables
        )
        return s
    def le(self, other):
        s = S(
            psipy.less_equal(self.value,other.value),
            variables = self.variables | other.variables
        )
        return s
    def gt(self, other):
        s = S(
            psipy.greater(self.value,other.value),
            variables = self.variables | other.variables
        )
        return s
    def ge(self, other):
        s = S(
            psipy.greater_equal(self.value,other.value),
            variables = self.variables | other.variables
        )
        return s
    def eq(self, other):
        s = S(
            psipy.equal(self.value,other.value),
            variables = self.variables | other.variables
        )
        return s
    def ne(self, other):
        s = S(
            psipy.not_equal(self.value,other.value),
            variables = self.variables | other.variables
        )
        return s

class PSI(Algebra):
    def __init__(self, values, free_variables=set()):
        Algebra.__init__(self, values)
        self.free_variables = free_variables

    def symbolize(self, expression, variables=set()):
        if isinstance(expression, (int, float)):
            return S(psipy.S(str(expression)))
        elif isinstance(expression, bool):
            return S(psipy.S(str(int(expression))))
        else:
            return S(expression, variables=set(variables))

    def integrate(self, weight, normalization=False):
        integrant = weight.value
        vs = set()
        for rv in weight.variables:
            integrant = psipy.mul(integrant, self.densities[rv[:-1]])
        for rv in weight.variables:
            if normalization or not self.is_free(rv[:-1]):
                integrant = psipy.integrate(self.random_values[rv[:-1]], integrant)
            else:
                vs.add(rv)
        return S(integrant, vs)

    def is_free(self, variable):
        return variable in self.free_variables

    def construct_density(self, name, functor, args):
        args = [a.value for a in args]
        sym_names = self.random_values[name]
        if functor in (
            psipy.delta_pdf,
            psipy.normal_pdf,
            psipy.uniform_pdf,
            psipy.beta_pdf,
            psipy.poisson_pdf
        ):
            #sym_names has only one entry
            return functor(sym_names[0], *args)
        elif functor in (psipy.normalInd_pdf,):
            return functor(sym_names, *args)
        elif functor in (psipy.real_symbol, ):
            return psipy.S("1")
            # return functor(sym_names[0])

    def make_values(self, name, dimension_values, functor, args):
        if name in self.random_values:
            return
        else:
            sym_names = []
            for v in dimension_values:
                s = Algebra.name2str(v.functor)
                s = psipy.S(s)
                sym_names.append(s)
            self.random_values[name] = sym_names
            functor = str2distribution[functor]
            density = self.construct_density(name, functor, args)
            self.densities[name] = density
            return

    def construct_negated_algebraic_expression(self, symbol):
        if psipy.is_iverson(symbol.value):
            neg_value = psipy.negate_condition(symbol.value)
        else:
            neg_value = psipy.sub(psipy.S("1"), symbol.value)
            neg_value = psipy.simplify(neg_value)
        return S(neg_value, symbol.variables)
