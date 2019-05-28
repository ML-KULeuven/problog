import torch
import pyro
import math

from .algebra import Algebra, BaseS

str2distribution = {
    "delta" : pyro.distributions.Delta,
    "normal" : pyro.distributions.Normal,
    "normalMV" : pyro.distributions.MultivariateNormal,
    "uniform" : pyro.distributions.Uniform,
    "beta" : pyro.distributions.Beta,
    "poisson" : pyro.distributions.Poisson
}

class S(BaseS):
    def __init__(self, tensor, variables=set()):
        BaseS.__init__(self, tensor, variables)

    def __add__(self, other):
        return S(self.value+other.value,variables = self.variables | other.variables)
    def __sub__(self, other):
        return S(self.value-other.value,variables = self.variables | other.variables)
    def __mul__(self, other):
        return S(self.value*other.value, variables = self.variables | other.variables)
    def __truediv__(self, other):
        return S(self.value/other.value,variables = self.variables | other.variables)
    def __pow__(self, other):
        return S(self.value**other.value,variables = self.variables | other.variables)

    def exp(self):
        if isinstance(self, (int, float)):
            return math.exp(self)
        else:
            return S(torch.exp(self.value), variables=self.variables)
    def sigmoid(self):
        if isinstance(self, (int, float)):
            return 1.0/(1+math.exp(-self))
        else:
            return S(torch.sigmoid(self.value), variables=self.variables)

    @staticmethod
    def gtz(a):
        if isinstance(a, (int,float)):
            return max(a,0)/a
        else:
            z = torch.zeros((1,))
            a = torch.abs(torch.max(a,z)/a)
            return a


    def lt(self, other):
        value = other.value-self.value
        value = self.gtz(value)
        return S(value,variables = self.variables | other.variables)
    def le(self, other):
        value = other.value-self.value
        value = self.gtz(value)
        s = S(value,variables = self.variables | other.variables)
        return s
    def gt(self, other):
        value = self.value-other.value
        value = self.gtz(value)
        s = S(value,variables = self.variables | other.variables)
        return s
    def ge(self, other):
        value = self.value-other.value
        value = self.gtz(value)
        s = S(value,variables = self.variables | other.variables)
        return s
    def eq(self, other):
        raise NotImplementedError()
    def ne(self, other):
        raise NotImplementedError()

    def obs(self,other):
        raise NotImplementedError()

        value = self.value-other.value
        s = S(value,variables = self.variables | other.variables)
        return s

class Pyro(Algebra):
    def __init__(self, values, free_variables, n_samples, ttype, device):
        Algebra.__init__(self, values, free_variables)
        self.Tensor = self.setup_tensor(ttype, device)
        torch.set_default_tensor_type(self.Tensor)

        self.free_variables = free_variables
        self.n_samples = n_samples
        self.device = torch.device(device)

    def setup_tensor(self, ttype, device):
        if ttype=="float64" and device=="cpu":
            Tensor = torch.DoubleTensor
        elif ttype=="float32" and device=="cpu":
            Tensor = torch.FloatTensor
        elif ttype=="float64":
            Tensor = torch.cuda.DoubleTensor
        elif ttype=="float32":
            Tensor = torch.cuda.FloatTensor
        return Tensor

    def symbolize(self, expression, variables=set()):
        if isinstance(expression, (int, float)):
            return S(float(expression))
        else:
            return S(expression, variables=set(variables))

    def integrate(self, weight):
        if isinstance(weight.value, (int,float)):
            return S(weight.value*self.n_samples)
        elif self.normalization:
            return S(torch.sum(weight.value))
        else:
            return S(torch.sum(weight.value))


    @staticmethod
    def _format_density(density, dim, n_samples):
        if tuple(density.batch_shape)==(dim, n_samples):
            return density
        elif tuple(density.batch_shape)==(n_samples,):
            return density.expand_by(torch.Size((dim,)))
        else:
            return density.expand_by(torch.Size((dim, n_samples)))

    def construct_density(self, name, dim, functor, args):
        args = [a.value for a in args]
        if functor in (
            pyro.distributions.Delta,
            pyro.distributions.Normal,
            pyro.distributions.Uniform,
            pyro.distributions.Beta,
            pyro.distributions.Poisson
        ):
            # return functor(*args)
            density = functor(*args)
            return self._format_density(density, dim, self.n_samples)
        # elif functor in (torch.normalInd_pdf,):
        #     return functor(*args)

    def make_values(self, name, dimension_values, functor, args):
        if name in self.random_values:
            pass
        else:
            functor = str2distribution[functor]
            density = self.construct_density(name, len(dimension_values), functor, args)
            samples = pyro.sample(name, density)
            self.densities[name] = density
            self.random_values[name] = samples

    def construct_negated_algebraic_expression(self, symbol):
        n_symbol = 1.0-symbol.value
        return self.symbolize(n_symbol, symbol.variables)
