from collections import OrderedDict

from problog.dd_formula import DDEvaluator
from problog.evaluator import SemiringLogProbability, SemiringProbability
from .formula import LogicNNFHAL


class DDEvaluatorHAL(DDEvaluator):
    def __init__(self, formula, semiring,  weights=None, **kwargs):
        DDEvaluator.__init__(self, formula, semiring, weights, **kwargs)

    def propagate(self):
        self._initialize()
        if isinstance(self.semiring, SemiringLogProbability) or isinstance(self.semiring, SemiringProbability):
            self.normalization = self._get_manager().wmc_true(self.weights, self.semiring)
        else:
            self.normalization = None

    def _initialize(self, with_evidence=True):
        self.weights.clear()
        weights = self.formula.extract_weights(self.semiring, self.given_weights)
        for atom, weight in weights.items():
            av = self.formula.atom2var.get(atom)
            if av is not None:
                self.weights[av] = weight
        if with_evidence:
            for ev in self.evidence():
                if ev in self.formula.atom2var:
                    # Only for atoms
                    self.set_evidence(self.formula.atom2var[ev], ev > 0)


    def get_sdds(self):#node
        result = {}
        constraint_inode = self.formula.get_constraint_inode()
        evidence_nodes = [self.formula.get_inode(ev) for ev in self.evidence()]
        self.evidence_inode = self._get_manager().conjoin(constraint_inode, *(evidence_nodes))
        result["e"] = self.evidence_inode
        result["qe"] = OrderedDict()
        for query, node in self.formula.queries():
            if node is self.formula.FALSE:
                result["qe"][query] = self.formula.FALSE
            else:
                query_def_inode = self.formula.get_inode(node)
                evidence_inode = self.evidence_inode
                query_sdd = self._get_manager().conjoin(query_def_inode, evidence_inode)
                result["qe"][query] = query_sdd
        return result

    def evaluate_sdd(self, sdd, semiring, normalization=False, evaluation_last=False):
        formula = LogicNNFHAL()
        i = self.formula._to_formula(formula, sdd)
        semiring.algebra.normalization = normalization
        result =  formula.evaluate(index=i, semiring=semiring)
        if evaluation_last:
            self._get_manager().deref(sdd)

        return result
