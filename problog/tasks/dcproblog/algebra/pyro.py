import torch
import pyro
import math

from .algebra import Algebra, BaseS, SUB

from ..logic import SymbolicConstant

str2distribution = {
    "delta": pyro.distributions.Delta,
    "normal": pyro.distributions.Normal,
    "normalMV": pyro.distributions.MultivariateNormal,
    "uniform": pyro.distributions.Uniform,
    "beta": pyro.distributions.Beta,
    "poisson": pyro.distributions.Poisson,
}


class MixtureComponent(object):
    def __init__(self, samples, weights, component_index):
        self.samples = samples
        self.weights = weights
        self.component_index = component_index

    def __truediv__(self, other):
        return MixtureComponent(
            self.samples, self.weights / other, self.component_index
        )

    def __str__(self):
        return "MixComp{}".format(str(self.component_index).translate(SUB))


class ObservationWeight(object):
    def __init__(self, value, dmu):
        self.value = value
        self.dmu = dmu

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)


class S(BaseS):
    def __init__(self, tensor, variables=set(), dmu=0):
        self.dmu = dmu
        if isinstance(tensor, torch.Tensor) and bool(torch.all(torch.eq(tensor, 0.0))):
            tensor = 0.0
        BaseS.__init__(self, tensor, variables)

    def __add__(self, other):
        if isinstance(self.value, (int, float)) and self.value == 0.0:
            return other
        elif isinstance(other.value, (int, float)) and other.value == 0.0:
            return self
        if self.dmu > other.dmu:
            return other
        elif self.dmu < other.dmu:
            return self
        else:
            return S(
                self.value + other.value,
                variables=self.variables | other.variables,
                dmu=self.dmu,
            )

    def __sub__(self, other):
        assert self.dmu == other.dmu
        assert self.dmu == 0
        return S(self.value - other.value, variables=self.variables | other.variables)

    def __mul__(self, other):
        dmu = self.dmu + other.dmu
        return S(
            self.value * other.value,
            variables=self.variables | other.variables,
            dmu=dmu,
        )

    def __truediv__(self, other):
        return S(self.value / other.value, variables=self.variables | other.variables)

    def __pow__(self, other):
        return S(self.value ** other.value, variables=self.variables | other.variables)

    def exp(self):
        if isinstance(self, (int, float)):
            return math.exp(self)
        else:
            return S(torch.exp(self.value), variables=self.variables)

    def sigmoid(self):
        if isinstance(self, (int, float)):
            return 1.0 / (1 + math.exp(-self))
        else:
            return S(torch.sigmoid(self.value), variables=self.variables)

    def lt(self, other):
        value = self.value < self.other
        return S(value.float(), variables=self.variables | other.variables)

    def le(self, other):
        value = self.value <= self.other
        return S(value.float(), variables=self.variables | other.variables)

    def gt(self, other):
        value = self.value > other.value
        return S(value.float(), variables=self.variables | other.variables)

    def ge(self, other):
        value = self.value >= other.value
        return S(value.float(), variables=self.variables | other.variables)

    def eq(self, other):
        value = self.value == other.value
        return S(value.float(), variables=self.variables | other.variables)

    def ne(self, other):
        value = self.value == other.value
        return S(value.float(), variables=self.variables | other.variables)


class Pyro(Algebra):
    def __init__(self, values, n_samples, ttype, device):
        Algebra.__init__(self, values)
        self.Tensor = self.setup_tensor(ttype, device)
        torch.set_default_tensor_type(self.Tensor)

        self.n_samples = n_samples
        self.device = torch.device(device)

    def setup_tensor(self, ttype, device):
        if ttype == "float64" and device == "cpu":
            Tensor = torch.DoubleTensor
        elif ttype == "float32" and device == "cpu":
            Tensor = torch.FloatTensor
        elif ttype == "float64":
            Tensor = torch.cuda.DoubleTensor
        elif ttype == "float32":
            Tensor = torch.cuda.FloatTensor
        return Tensor

    def symbolize(self, expression, variables=set(), dmu=0):
        if isinstance(expression, (int, float)):
            return S(float(expression), dmu=dmu)
        elif isinstance(expression, ObservationWeight):
            return S(
                observation_weight.value,
                variables=set(variables),
                dmu=observation_weight.dmu,
            )
        else:
            return S(expression, variables=set(variables), dmu=dmu)

    def integrate(self, weight, free_variable=None, normalization=False):
        if free_variable:
            self.create_values(free_variable)
            values = self.random_values[free_variable]
            # TODO pass on variables
            return S(MixtureComponent(values, weight.value, free_variable[1]))
        else:
            if isinstance(weight.value, (int, float)):
                return S(weight.value, dmu=weight.dmu)
            else:
                return S(torch.mean(weight.value), dmu=weight.dmu)

    def normalize(self, a, z):
        if a.dmu > z.dmu:
            return self.symbolize(0)
        else:
            return a / z

    @staticmethod
    def _format_density(density, dim, n_samples):
        if tuple(density.batch_shape) == (dim, n_samples):
            return density
        elif tuple(density.batch_shape) == (n_samples,):
            return density.expand_by(torch.Size((dim,)))
        else:
            return density.expand_by(torch.Size((dim, n_samples)))

    def construct_density(self, name, dim, functor, args):
        args = [a.value for a in args]
        if functor in (
            pyro.distributions.Normal,
            pyro.distributions.Uniform,
            pyro.distributions.Beta,
            pyro.distributions.Poisson,
        ):
            density = functor(*args)
            return self._format_density(density, dim, self.n_samples)
        elif functor in (pyro.distributions.Delta,):
            if isinstance(args[0], (int, float)):
                v = torch.tensor(args[0])
            else:
                v = args[0]
            density = functor(v)
            return self._format_density(density, dim, self.n_samples)
        # elif functor in (torch.normalInd_pdf,):
        #     return functor(*args)

    def make_values(self, name, components, functor, args):
        if name in self.random_values:
            pass
        else:
            functor = str2distribution[functor]
            density = self.construct_density(name, len(components), functor, args)
            samples = pyro.sample(name, density)
            self.densities[name] = density
            self.random_values[name] = samples

    def construct_negated_algebraic_expression(self, symbol):
        n_symbol = 1.0 - symbol.value
        return self.symbolize(n_symbol, symbol.variables)

    def make_observation(self, var, obs):
        density_name = var.name
        dimensions = var.dimensions
        assert dimensions == 1  # TODO allow for multivariate observations

        for c in var.components:
            self.construct_algebraic_expression(c)

        obs = self.construct_algebraic_expression(obs)
        density = self.densities[density_name]

        if var.distribution_functor in ("delta",):
            arg = var.distribution_args[0]
            if isinstance(arg, (SymbolicConstant,)):
                value = torch.exp(density.log_prob(torch.tensor(obs.value)))
                dmu = 0
                observation_weight = ObservationWeight(value, dmu)
            else:
                observation_weight = self.make_observation(arg, obs)
        else:
            value = torch.exp(density.log_prob(torch.tensor(obs.value)))
            dmu = 1
            observation_weight = ObservationWeight(value, dmu)

        self.random_values[density_name][dimensions - 1] = obs.value
        # this is the line that relates to the comment above

        return observation_weight
