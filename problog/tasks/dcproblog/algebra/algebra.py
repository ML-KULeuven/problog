from problog.logic import Constant


from ..logic import (
    pdfs,
    SymbolicConstant,
    RandomVariableComponentConstant,
    RandomVariableConstant,
)


SUB = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")
M_SUB = "-".translate(SUB)


def get_algebra(abstract_abe, values, **kwdargs):
    if abstract_abe.name == "psi":
        from .psi import PSI

        return PSI(values, **kwdargs)
    elif abstract_abe.name == "pyro":
        from .pyro import Pyro

        return Pyro(
            values, abstract_abe.n_samples, abstract_abe.ttype, abstract_abe.device
        )


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
    "observation": obsS,
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
    def __init__(self, values):
        self.density_values = values
        self.random_values = {}
        self.densities = {}

    @staticmethod
    def name2str(name):
        name = "{term}{no_d}{M_SUB}{no_dim}".format(
            term=name[0],
            no_d=str(name[1]).translate(SUB),
            no_dim=str(name[2]).translate(SUB),
            M_SUB=M_SUB,
        )
        for c, rc in {"(": "__", ")": "", ",": "_"}.items():
            name = name.replace(c, rc)
        return name

    # def is_free(self, v):
    #     for fv in self.free_variables:
    #         if fv == v[0]:
    #             return True
    #     return False

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

    def result(self, a, free_variable=None, normalization=False):
        return self.integrate(
            a, free_variable=free_variable, normalization=normalization
        )

    def probability(self, a, z):
        if not a:
            return self.symbolize(0)
        return a / z

    def get_values(self, density_name, dimension):
        if not density_name in self.random_values:
            density = self.density_values[density_name]
            args = [
                self.construct_algebraic_expression(a)
                for a in density.distribution_args
            ]
            assert density_name == density.name
            self.make_values(
                density.name, density.components, density.distribution_functor, args
            )
        return self.random_values[density_name][dimension]

    def construct_algebraic_expression(self, expression):
        if isinstance(expression, Constant):
            return self.construct_algebraic_expression(expression.functor)
        # elif isinstance(expression, RandomVariableConstant) and expression.functor=="beta":
        elif isinstance(expression, RandomVariableConstant) and (
            expression.distribution_functor in ("beta",)
        ):
            component = self.construct_algebraic_expression(expression.components[0])
            return component
        elif isinstance(expression, RandomVariableConstant):
            raise NotImplementedError
        else:
            assert isinstance(expression, SymbolicConstant)

            if expression.functor == "observation":
                observation_weight = self.make_observation(*expression.args)
                return self.symbolize(
                    observation_weight, variables=expression.args[0].cvariables
                )
            elif isinstance(expression, RandomVariableComponentConstant):
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
