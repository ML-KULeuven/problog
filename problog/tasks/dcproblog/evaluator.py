from problog.evaluator import Semiring, FormulaEvaluator

from .algebra.algebra import get_algebra



class SemiringHAL(Semiring):
    def __init__(self, neutral, abe, density_values, density_queries, free_variables):
        Semiring.__init__(self)
        self.neutral = neutral
        self.density_queries = density_queries
        self.algebra = get_algebra(abe, density_values, free_variables)
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
    def pos_value(self, a, key, index=None):
        if key in self.pos_values:
            return self.pos_values[key]
        else:
            pv = self.value(a)
            self.pos_values[key] = pv
            return pv
    def neg_value(self, a, key, index=None):
        if key in self.neg_values:
            return self.neg_values[key]
        else:
            nv = self.pos_values[key]
            nv = self.negate(nv)
            return nv
    def value(self, a):
        return self.algebra.value(a)
    def result(self, evaluator, index, formula=None, normalization=False):
        a = evaluator.get_weight(index)
        return self.algebra.result(a, formula=formula)
    def is_dsp(self):
        return True
    def is_nsp(self):
        return False


class FormulaEvaluatorHAL(FormulaEvaluator):
    def __init__(self, formula, semiring, weights=None):
        FormulaEvaluator.__init__(self, formula, semiring, weights=None)

    def evaluate(self, index, normalization=False):
        result =  self.semiring.result(self, index, formula=self.formula)
        return result

    def compute_weight(self, index):
        """Compute the weight of the node with the given index.

        :param index: integer or formula.TRUE or formula.FALSE
        :return: weight of the node
        """
        if index == self.formula.TRUE:
            return self.semiring.one()
        elif index == self.formula.FALSE:
            return self.semiring.zero()
        else:
            node = self.formula.get_node(abs(index))
            ntype = type(node).__name__
            if ntype == 'atom':
                return self.semiring.one()
            else:
                childprobs = [self.get_weight(c) for c in node.children]
                if ntype == 'conj':
                    assert len(childprobs) == 2
                    p = self.semiring.times(*childprobs, index=index)
                    return p
                elif ntype == 'disj':
                    assert len(childprobs) >= 2
                    p = childprobs[0]
                    for cp in childprobs[1:]:
                        p = self.semiring.plus(p, cp, index=index)
                    return p
                else:
                    raise TypeError("Unexpected node type: '%s'." % ntype)

    def get_weight(self, index):
        """Get the weight of the node with the given index.

        :param index: integer or formula.TRUE or formula.FALSE
        :return: weight of the node
        """

        if index == self.formula.TRUE:
            return self.semiring.one()
        elif index == self.formula.FALSE:
            return self.semiring.zero()
        elif index < 0:
            weight = self._fact_weights.get(abs(index))
            if weight is None:
                # This will only work if the semiring support negation!
                nw = self.get_weight(-index)
                return self.semiring.negate(nw)
            else:
                return weight[1]
        else:
            weight = self._fact_weights.get(index)
            if weight is None:
                weight = self._computed_weights.get(index)
                if weight is None:
                    weight = self.compute_weight(index)
                    self._computed_weights[index] = weight
                return weight
            else:
                return weight[0]
