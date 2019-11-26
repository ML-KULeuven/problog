from problog.logic import Constant

from ..logic import SymbolicConstant, ValueDimConstant

SUB = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")
M_SUB = "-".translate(SUB)


def get_algebra(abstract_abe, values, free_variables, **kwdargs):
    if abstract_abe.name == "psi":
        from .psi import PSI
        return PSI(values, free_variables)
    elif abstract_abe.name == "pyro":
        from .pyro import Pyro
        return Pyro(values, free_variables, abstract_abe.n_samples, abstract_abe.ttype, abstract_abe.device)


def addS(a, b):
    return a + b


def subS(a, b):
    return a - b


def mulS(a, b):
    return a * b


def divS(a, b):
    return a / b


def powS(a, b):
    return a ** b


def expS(a):
    return a.exp()


def sigmoidS(a):
    return a.sigmoid()


def ltS(a, b):
    return a.lt(b)


def leS(a, b):
    return a.le(b)


def gtS(a, b):
    return a.gt(b)


def geS(a, b):
    return a.ge(b)


def eqS(a, b):
    return a.eq(b)


def neS(a, b):
    return a.ne(b)


def obsS(a, b):
    r = a.obs(b)
    print(r)
    return r


str2algebra = {
    "<": ltS,
    "<=": leS,
    ">": gtS,
    ">=": geS,
    "=": eqS,
    "\=": neS,

    "list": list,

    "add": addS,
    "sub": subS,
    "mul": mulS,
    "div": divS,
    "/": divS,
    "pow": powS,
    "exp": expS,
    "sigmoid": sigmoidS,

    "obs": obsS
}


class BaseS(object):
    def __init__(self, value, variables):
        self.value = value
        self.variables = variables

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)


class Algebra(object):
    random_values = {}
    densities = {}
    normalization = False  # TODO move this to argument of integration functions

    def __init__(self, values, free_variables):
        self.density_values = values
        self.free_variables = free_variables

    @staticmethod
    def name2str(name):
        name = "{term}{no_d}{M_SUB}{no_dim}".format(term=name[0], no_d=str(name[1]).translate(SUB),
                                                    no_dim=str(name[2]).translate(SUB), M_SUB=M_SUB)
        for c, rc in {"(": "__", ")": "", ",": "_"}.items():
            name = name.replace(c, rc)
        return name

    def is_free(self, v):
        for fv in self.free_variables:
            if fv == v[0]:
                return True
        return False

    def one(self):
        return self.symbolize(1)

    def zero(self):
        return self.symbolize(0)

    def times(self, a, b, index=None):
        return a * b

    def plus(self, a, b, index=None):
        return a + b

    def value(self, expression):
        return self.construct_algebraic_expression(expression)

    def negate(self, a):
        return self.construct_negated_algebraic_expression(a)

    def result(self, a, formula=None):
        return self.integrate(a)

    def probability(self, a, z):
        if not a:
            return self.symbolize(0)
        return a / z

    def get_values(self, density_name, dimension):
        if not density_name in self.random_values:
            density = self.density_values[density_name]
            args = [self.construct_algebraic_expression(a) for a in density.args]
            self.make_values(density.name, density.dimension_values, density.functor, args)
        return self.random_values[density_name][dimension]

    def construct_algebraic_expression(self, expression):
        if isinstance(expression, Constant):
            return self.construct_algebraic_expression(expression.functor)
        else:
            assert isinstance(expression, SymbolicConstant)
            if expression.functor == "obs":
                v = expression.args[0]
                density_name = v.density_name
                dimension = v.dimension
                self.construct_algebraic_expression(v)
                obs = self.construct_algebraic_expression(expression.args[1])

                self.random_values[density_name][dimension] = obs.value
                print(self.random_values[density_name][dimension])
                print(v)
                print(obs)
                print(v)

                return obs

            elif isinstance(expression, ValueDimConstant):
                density_name = expression.density_name
                dimension = expression.dimension
                values = self.get_values(density_name, dimension)
                return self.symbolize(values, variables=expression.cvariables)
            else:
                args = [self.construct_algebraic_expression(a) for a in expression.args]
                if isinstance(expression.functor, bool):
                    return self.symbolize(int(expression.functor))
                elif isinstance(expression.functor, (int, float)):
                    return self.symbolize(expression.functor)
                else:
                    functor = str2algebra[expression.functor]
                    if functor == list:
                        return args
                    else:
                        return functor(*args)
