from __future__ import print_function

from problog.program import PrologFile
from problog.formula import LogicDAG
from problog.evaluator import Semiring, SemiringProbability, FormulaEvaluatorNSP, FormulaEvaluator
from problog.sdd_formula import SDD


class SemiringMPE(Semiring):

    def __init__(self):
        Semiring.__init__(self)

    def zero(self):
        return 0.0

    def one(self):
        return 1.0

    def plus(self, a, b):
        return max(a, b)

    def times(self, a, b):
        return a * b

    def pos_value(self, a, key=None):
        return float(a)

    def neg_value(self, a, key=None):
        return 1.0 - float(a)

    def is_nsp(self):
        return True


class SemiringMPEState(Semiring):

    def __init__(self):
        Semiring.__init__(self)

    def zero(self):
        return 0.0, set()

    def one(self):
        return 1.0, set()

    def plus(self, a, b):
        if a[0] > b[0]:
            return a
        elif a[0] < b[0]:
            return b
        else:
            return a[0], a[1] | b[1]

    def times(self, a, b):
        return a[0] * b[0], a[1] | b[1]

    def pos_value(self, a, key=None):
        return float(a), {key}

    def neg_value(self, a, key=None):
        return 1.0 - float(a), {-key}

    def is_nsp(self):
        return True

    def result(self, a, formula=None):
        p, state = a
        state1 = []
        for i in state:
            if i > 0:
                state1.append(formula.get_node(i).name)
            else:
                state1.append(-formula.get_node(-i).name)
        return p, state1


def solve(model, semiring):

    if semiring.is_dsp():
        ev = SDD.create_from(model, label_all=True).to_formula()
    else:
        ev = LogicDAG.create_from(model, label_all=True)

    if semiring.is_nsp():
        fe = FormulaEvaluatorNSP(ev, semiring)
    else:
        fe = FormulaEvaluator(ev, semiring)

    weights = ev.extract_weights(semiring=semiring)
    fe.set_weights(weights)

    result = {}
    for n, q in ev.queries():
        p = fe.evaluate(q)
        result[n] = p
    return result


def main(inputfile, semiring, **kwdargs):

    pl = PrologFile(inputfile)

    if semiring == 'prob':
        sm = SemiringProbability()
    elif semiring == 'mpe':
        sm = SemiringMPE()
    elif semiring == 'mpe_state':
        sm = SemiringMPEState()

    result = solve(pl, semiring=sm)

    for k, v in result.items():
        print ('%s: %s' % (k, v))


def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('-s', '--semiring', choices=['prob', 'mpe', 'mpe_state'], default='prob')
    return parser


if __name__ == '__main__':
    main(**vars(argparser().parse_args()))