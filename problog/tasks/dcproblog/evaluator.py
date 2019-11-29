from problog.evaluator import Semiring, FormulaEvaluator




class SemiringHAL(Semiring):
    def __init__(self, neutral, abe, density_values):
        Semiring.__init__(self)
        self.neutral = neutral
        from .algebra.algebra import get_algebra
        self.algebra = get_algebra(abe, density_values)
        self.pos_values = {}
        self.neg_values = {}


    def zero(self):
        return self.algebra.zero()
    def one(self):
        return self.algebra.one()
    def plus(self, a, b, index=None):
        return self.algebra.plus(a,b)
    def times(self, a, b, index=None):
        result = self.algebra.times(a,b)
        return result
    def negate(self, a):
        return self.algebra.negate(a)
    def pos_value(self, a, key, index):
        if index in self.pos_values:
            return self.pos_values[index]
        else:
            pv = self.value(a)
            self.pos_values[index] = pv
            return pv
    def neg_value(self, a, key, index):
        if index in self.neg_values:
            return self.neg_values[index]
        else:
            nv = self.pos_values[index]
            nv = self.negate(nv)
            return nv
    def value(self, a):
        return self.algebra.value(a)
    def result(self, a, normalization=False):
        return self.algebra.result(a, normalization=normalization)
    def is_dsp(self):
        return True
    def is_nsp(self):
        return False
