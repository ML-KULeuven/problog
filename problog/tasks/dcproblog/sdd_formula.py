from collections import OrderedDict

from problog.core import transform, transform_create_as
from problog.dd_formula import build_dd
from problog.sdd_formula import SDD, SDDEvaluator
from problog.formula import LogicFormula
from problog.evaluator import SemiringLogProbability

from .logic import Mixture
from .formula import atom, LogicFormulaHAL



class SDDHAL(SDD, LogicFormulaHAL):
    def __init__(self, **kwdargs):
        SDD.__init__(self, **kwdargs)
        LogicFormulaHAL.__init__(self, **kwdargs)
        self.density_values = {}

    def _create_evaluator(self, semiring, weights, **kwargs):
        return SDDEvaluatorHAL(self, semiring, weights, **kwargs)

    def get_evaluator(self, semiring=None, evidence=None, weights=None, keep_evidence=False, **kwargs):
        """Get an evaluator for computing queries on this formula.
        It creates an new evaluator and initializes it with the given or predefined evidence.

        :param semiring: semiring to use
        :param evidence: evidence values (override values defined in formula)
        :type evidence: dict(Term, bool)
        :param weights: weights to use
        :return: evaluator for this formula
        """
        # if semiring is None:
        #     # TODO change this to hal semiring
        #     # if no semiring is given here test_try_call.py fails? for some reason?
        #     semiring = SemiringLogProbability()
        # assert semiring


        evaluator = self._create_evaluator(semiring, weights, **kwargs)

        for ev_name, ev_index, ev_value in self.evidence_all():
            if ev_index == 0 and ev_value > 0:
                pass  # true evidence is deterministically true
            elif ev_index is None and ev_value < 0:
                pass  # false evidence is deterministically false
            elif ev_index == 0 and ev_value < 0:
                raise InconsistentEvidenceError(source='evidence('+str(ev_name)+',false)')  # true evidence is false
            elif ev_index is None and ev_value > 0:
                raise InconsistentEvidenceError(source='evidence('+str(ev_name)+',true)')  # false evidence is true
            elif evidence is None and ev_value != 0:
                evaluator.add_evidence(ev_value * ev_index)
            elif evidence is not None:
                try:
                    value = evidence[ev_name]
                    if value is None:
                        pass
                    elif value:
                        evaluator.add_evidence(ev_index)
                    else:
                        evaluator.add_evidence(-ev_index)
                except KeyError:
                    if keep_evidence:
                        evaluator.add_evidence(ev_value * ev_index)

        for ob_name, ob_index, ob_value in self.observation_all():
            evaluator.add_observation(ob_index)

        evaluator.propagate()
        return evaluator


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
        if current_node is None:
            retval = formula.FALSE
        elif self.get_manager().is_true(current_node):
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


    def extract_weights(self, semiring, weights=None):
        """Extracts the positive and negative weights for all atoms in the data structure.

        :param semiring: semiring that determines the interpretation of the weights
        :param weights: dictionary of { node name : weight } that overrides the builtin weights
        :returns: dictionary { key: (positive weight, negative weight) }
        :rtype: dict[int, tuple[any]]

        Atoms with weight set to neutral will get weight ``(semiring.one(), semiring.one())``.

        If the weights argument is given, it completely replaces the formula's weights.

        All constraints are applied to the weights.
        """

        if weights is None:
            weights = self.get_weights()
        else:
            oweights = dict(self.get_weights().items())
            oweights.update({self.get_node_by_name(n): v for n, v in weights.items()})
            weights = oweights

        result = {}
        observation_weight_nodes = [w for w  in weights if weights[w].functor=="observation"]
        for on in observation_weight_nodes:
            name = self._get_name(on)
            result[on] = semiring.pos_value(weights[on], name, on), semiring.neg_value(weights[on], name, on)


        for n, w in weights.items():
            if n in observation_weight_nodes:
                continue
            name = self._get_name(n)
            if w == self.WEIGHT_NEUTRAL and type(self.WEIGHT_NEUTRAL) == type(w):
                result[n] = semiring.one(), semiring.one()
            elif w == False:
                result[n] = semiring.false(name)
            elif w is None:
                result[n] = semiring.true(name)
            else:
                result[n] = semiring.pos_value(w, name, n), semiring.neg_value(w, name, n)

        for c in self.constraints():
            c.update_weights(result, semiring)

        return result

    def _get_name(self, n):
        if hasattr(self, 'get_name'):
            name = self.get_name(n)
        else:
            name = n
        return name





class SDDEvaluatorHAL(SDDEvaluator):
    def __init__(self, formula, semiring,  weights=None, **kwargs):
        SDDEvaluator.__init__(self, formula, semiring, weights, **kwargs)
        self.__observation = []

    def observation(self):
        """Iterate over observation."""
        return iter(self.__observation)

    def add_observation(self, node):
        """Add observation"""
        self.__observation.append(node)

    def has_observation(self):
        """Checks whether there is active observation."""
        return self.__observation != []


    def propagate(self):
        self._initialize()
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


    def get_sdds(self):
        result = {}
        constraint_inode = self.formula.get_constraint_inode()
        evidence_nodes = [self.formula.get_inode(ev) for ev in self.evidence()]
        observation_nodes = [self.formula.get_inode(ob) for ob in self.observation()]

        self.evidence_inode = self._get_manager().conjoin(constraint_inode, *(observation_nodes), *(evidence_nodes))
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
        result["dqe"] = OrderedDict()
        for dquery, _ in self.formula.dqueries():
            dq = dquery[0]
            components_sdd = []
            for c in dquery[1].args:
                dquery_def_inode = self.formula.get_inode(c[1])
                evidence_inode = self.evidence_inode
                dquery_sdd = self._get_manager().conjoin(dquery_def_inode, evidence_inode)
                components_sdd.append((c[0], dquery_sdd))

            mixture = Mixture(*components_sdd)
            result["dqe"][dq] = mixture
        return result

    def evaluate_sdd(self, sdd, normalization=False, free_variable=None, evaluation_last=False):
        if sdd is None:
            result = self.semiring.zero()
        elif sdd.is_true():
            if not self.semiring.is_nsp():
                result = self.semiring.one()
            else:
                result = self._evidence_weight
        elif sdd.is_false():
            result = self.semiring.zero()
        else:
            smooth_to_root = self.semiring.is_nsp()
            result = self._get_manager().wmc(sdd, weights=self.weights, semiring=self.semiring,
                                                 pr_semiring=False, perform_smoothing=True,
                                                 smooth_to_root=smooth_to_root)
            result = self.semiring.result(result, free_variable=free_variable, normalization=normalization)
            if evaluation_last:
                self._get_manager().deref(sdd)

        return result


@transform(LogicFormula, SDDHAL)
def build_sdd(source, destination, **kwdargs):
    result = build_dd(source, destination, **kwdargs)
    return result

transform_create_as(SDDHAL, LogicFormula)
