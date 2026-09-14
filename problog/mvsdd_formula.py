"""
problog.mvsdd_formula - Multi-valued Sentential Decision Diagrams
-----------------------------------------------------------------

Interface to multi-valued SDDs (MV-SDD), which represent annotated disjunctions natively.

An SDD encodes an annotated disjunction with one Boolean variable per head, plus one for the
choice of none of them, and has to compile the constraint that exactly one of those is true.
An MV-SDD gives the annotated disjunction a single variable with one value per head (and one
for none of them), so there is no constraint to compile and the vtree keeps the alternatives
of a choice together.

..
    Part of the ProbLog distribution.

    Copyright 2026 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import logging
from array import array

from .constraint import ConstraintAD
from .core import transform
from .dd_formula import DD, DDManager, DDEvaluator, build_dd
from .errors import InstallError, InconsistentEvidenceError
from .evaluator import SemiringLogProbability, SemiringProbability
from .formula import LogicDAG, LogicFormula, atom
from .logic import Term

# noinspection PyBroadException
try:
    import pymvsdd

    if not hasattr(pymvsdd.Manager, "wmc"):
        # Releases before weighted model counting cannot evaluate a program.
        pymvsdd = None
except Exception:
    pymvsdd = None

ISSUE_URL = "https://github.com/rmanhaeve/mv-sdd/issues"


class MVSDD(DD):
    """A propositional logic formula represented as a multi-valued SDD.

    The variables of the MV-SDD are only known once the formula is complete: every
    annotated disjunction becomes one variable and every other atom a Boolean one.
    The manager is therefore created on first use, and atoms can not be added after that.
    """

    transform_preference = 5

    def __init__(self, **kwdargs):
        if pymvsdd is None:
            raise InstallError(
                "The MV-SDD library is not available. Please install the mv-sdd package"
            )
        DD.__init__(self, auto_compact=False, **kwdargs)
        self._variables = None

    @classmethod
    def is_available(cls):
        """Checks whether the MV-SDD library is available."""
        return pymvsdd is not None

    def _create_atom(
        self, identifier, probability, group, name=None, source=None, is_extra=False
    ):
        # Variables are assigned when the manager is created.
        return atom(identifier, probability, group, name, source, is_extra)

    def _create_manager(self):
        self._variables = self._assign_variables()
        return MVSDDManager([v.domain_size for v in self._variables])

    def _create_evaluator(self, semiring, weights, **kwargs):
        return MVSDDEvaluator(self, semiring, weights, **kwargs)

    def _assign_variables(self):
        """Give every atom a variable and a value, and return the variables.

        A non-trivial annotated disjunction becomes a variable whose values are its heads and,
        last, its extra atom (the choice of none of the heads).  Atoms outside such a group
        become Boolean variables with the atom as value 1.  Variables are numbered in the order
        their first atom appears in the formula.
        """
        groups = {}
        for c in self.constraints():
            if isinstance(c, ConstraintAD) and c.is_nontrivial():
                values = sorted(c.nodes)
                if c.extra_node is not None:
                    values.append(c.extra_node)
                if len(values) > pymvsdd.MAX_DOMAIN:
                    self._warn_large_choice(c, values)
                    continue
                group = _Variable(values, c)
                for node in values:
                    groups[node] = group

        variables = []
        for index, node, nodetype in self:
            if nodetype != "atom" or index in self.atom2var:
                continue
            group = groups.get(index)
            if group is None:
                group = _Variable([index])
            group.var = len(variables)
            variables.append(group)
            for value, node_index in group.value_atoms():
                self.atom2var[node_index] = (group.var, value)
                self.var2atom[(group.var, value)] = node_index

        if not variables:
            # A vtree needs a variable.  This one is never used, so any weights summing to one do.
            variables.append(_Variable([]))
        return variables

    def _warn_large_choice(self, constraint, values):
        heads = sorted(constraint.nodes)
        name = self.get_node(heads[0]).name
        if isinstance(name, Term) and name.functor == "choice" and name.arity == 4:
            name = name.args[2]
        logging.getLogger("problog").warning(
            "The annotated disjunction with head %s has %d alternatives, but MV-SDD "
            "represents at most %d (including the choice of none of them). It is encoded "
            "with one Boolean variable per alternative instead, as in an SDD, which is "
            "slower. If you need larger annotated disjunctions, please open an issue at %s",
            name,
            len(values),
            pymvsdd.MAX_DOMAIN,
            ISSUE_URL,
        )

    def get_variables(self):
        """The variables of the MV-SDD, indexed by variable number."""
        self.get_manager()
        return self._variables

    def build_constraint_dd(self):
        """Build the constraint of this formula, leaving out the annotated disjunctions that
        are variables of the MV-SDD."""
        native = set(id(v.constraint) for v in self.get_variables() if v.constraint)
        mgr = self.get_manager()
        mgr.constraint_dd = mgr.true()
        for c in self.constraints():
            if id(c) in native:
                continue
            for rule in c.as_clauses():
                clause = mgr.disjoin(*[self.get_inode(r) for r in rule])
                mgr.constraint_dd = mgr.conjoin(mgr.constraint_dd, clause)

    def value_weights(self, weights, semiring):
        """Weights for every value of every variable.

        :param weights: weights as returned by :func:`LogicFormula.extract_weights`
        :param semiring: semiring the weights belong to
        :return: one list of weights per variable, one weight per value
        """
        neutral = (semiring.one(), semiring.one())
        result = []
        for variable in self.get_variables():
            if variable.constraint is not None:
                result.append([weights.get(n, neutral)[0] for n in variable.atoms])
            elif variable.atoms:
                pos, neg = weights.get(variable.atoms[0], neutral)
                result.append([neg, pos])
            else:
                result.append([semiring.one(), semiring.zero()])
        return result

    def to_formula(self):
        """Extracts a LogicFormula from the MV-SDD.

        A value of an annotated disjunction becomes its atom together with the negation of the
        other atoms of the group.  The atoms keep their group, so the formula has the
        annotated disjunctions as constraints again.
        """
        formula = LogicFormula(keep_order=True)
        mgr = self.get_manager()
        cache = {}
        for name, key, label in self.labeled():
            node = mgr.conjoin(self.get_inode(key), self.get_constraint_inode())
            formula.add_name(name, self._to_formula(formula, node, cache), label)
        return formula

    def _to_formula(self, formula, current_node, cache=None):
        if cache is not None and current_node.node_id in cache:
            return cache[current_node.node_id]
        if current_node.is_top:
            result = formula.TRUE
        elif current_node.is_bot:
            result = formula.FALSE
        elif current_node.is_terminal:
            variable = self.get_variables()[current_node.var]
            result = formula.add_or(
                [
                    self._value_to_formula(formula, variable, value)
                    for value in current_node.values
                ]
            )
        else:
            result = formula.add_or(
                [
                    formula.add_and(
                        (
                            self._to_formula(formula, p, cache),
                            self._to_formula(formula, s, cache),
                        )
                    )
                    for p, s in current_node.elements
                ]
            )
        if cache is not None:
            cache[current_node.node_id] = result
        return result

    def _value_to_formula(self, formula, variable, value):
        def add(index):
            node = self.get_node(index)
            return formula.add_atom(
                index,
                probability=node.probability,
                name=node.name,
                group=node.group,
                cr_extra=False,
                is_extra=node.is_extra,
            )

        if variable.constraint is None:
            literal = add(variable.atoms[0])
            return literal if value == 1 else -literal
        return formula.add_and(
            [add(n) if i == value else -add(n) for i, n in enumerate(variable.atoms)]
        )

    def to_dot(self, *args, **kwargs):
        if kwargs.get("use_internal"):
            return self.get_manager().to_dot(
                [(name, self.get_inode(key)) for name, key in self.queries()],
                self._value_name,
            )
        else:
            return self.to_formula().to_dot(*args, **kwargs)

    def _value_name(self, var, value):
        variable = self.get_variables()[var]
        if variable.constraint is None:
            name = self.get_node(variable.atoms[0]).name
            return str(name) if value == 1 else "\\+%s" % name
        return str(self.get_node(variable.atoms[value]).name)


class _Variable(object):
    """A variable of the MV-SDD: the atoms of an annotated disjunction, one per value, or the
    single atom of a Boolean variable."""

    def __init__(self, atoms, constraint=None):
        self.atoms = atoms
        self.constraint = constraint
        self.var = None

    @property
    def domain_size(self):
        return len(self.atoms) if self.constraint is not None else 2

    def value_atoms(self):
        """Pairs of value and atom."""
        if self.constraint is not None:
            return list(enumerate(self.atoms))
        return [(1, a) for a in self.atoms]


class MVSDDManager(DDManager):
    """Manager for MV-SDDs.

    Nodes are ``pymvsdd.Node`` objects, which release their reference when they are garbage
    collected, so reference counting is a no-op.  A literal is a pair of variable and value.
    """

    def __init__(self, domain_sizes):
        DDManager.__init__(self)
        self.__manager = pymvsdd.Manager(pymvsdd.Vtree.balanced(domain_sizes))
        self._vtree = self.__manager.vtree_nodes()
        self._vtree_root = self.__manager.vtree_root()

    def get_manager(self):
        """Get the underlying MV-SDD manager."""
        return self.__manager

    def add_variable(self, label=0):
        raise NotImplementedError("The variables of an MV-SDD are fixed.")

    def literal(self, label):
        var, value = label
        return self.__manager.literal(var, value)

    def is_true(self, node):
        return node.is_top

    def true(self):
        return self.__manager.top()

    def is_false(self, node):
        return node.is_bot

    def false(self):
        return self.__manager.bot()

    def conjoin2(self, a, b):
        return self.__manager.conjoin(a, b)

    def disjoin2(self, a, b):
        return self.__manager.disjoin(a, b)

    def negate(self, node):
        return self.__manager.negate(node)

    def same(self, node1, node2):
        if node1 is None or node2 is None:
            return node1 is node2
        return node1.node_id == node2.node_id

    def ref(self, *nodes):
        pass

    def deref(self, *nodes):
        pass

    def wmc(self, node, weights, log_mode=False):
        """Weighted model count over all variables.

        :param node: node to count
        :param weights: weights of all values, variable by variable, as ``array('d')``
        :param log_mode: whether weights and result are natural logarithms
        """
        return self.__manager.wmc(node, weights, log_mode)

    def wmc_semiring(self, node, weights, semiring):
        """Weighted model count over all variables in an arbitrary semiring.

        :param node: node to count
        :param weights: one list of weights per variable, with one weight per value
        :param semiring: semiring to count in
        """
        vtree = self._vtree
        totals = [None] * len(vtree)

        def total(v):
            parent, left, right, var = vtree[v]
            if var is None:
                t = semiring.times(total(left), total(right))
            else:
                t = _sum(semiring, weights[var])
            totals[v] = t
            return t

        total(self._vtree_root)
        memo = {}

        def lift(child, ancestor):
            # Weighted count of child over the variables below ancestor.
            if child.is_bot:
                return semiring.zero()
            if child.is_top:
                return totals[ancestor]
            result = local(child)
            pos = child.vtree_pos
            while pos != ancestor:
                parent = vtree[pos][0]
                _, left, right, _ = vtree[parent]
                result = semiring.times(result, totals[right if left == pos else left])
                pos = parent
            return result

        def local(n):
            result = memo.get(n.node_id)
            if result is None:
                if n.is_terminal:
                    w = weights[n.var]
                    result = _sum(semiring, [w[v] for v in n.values])
                else:
                    _, left, right, _ = vtree[n.vtree_pos]
                    result = _sum(
                        semiring,
                        [
                            semiring.times(lift(p, left), lift(s, right))
                            for p, s in n.elements
                        ],
                    )
                memo[n.node_id] = result
            return result

        return lift(node, self._vtree_root)

    def to_dot(self, roots, value_name):
        """Graphviz rendering of the given nodes.

        :param roots: list of (name, node) pairs
        :param value_name: function from variable and value to a label
        """
        lines = ["digraph mvsdd {", "overlap=false"]
        seen = set()

        def visit(n):
            key = "n%d" % n.node_id
            if n.node_id in seen:
                return key
            seen.add(n.node_id)
            if n.is_top or n.is_bot:
                lines.append('%s [label="%s",shape=box];' % (key, n.is_top))
            elif n.is_terminal:
                label = " | ".join(value_name(n.var, v) for v in n.values)
                lines.append('%s [label="%s",shape=box];' % (key, label))
            else:
                lines.append('%s [label="OR",shape=circle];' % key)
                for i, (p, s) in enumerate(n.elements):
                    element = "%s_%d" % (key, i)
                    lines.append('%s [label="AND",shape=point];' % element)
                    lines.append("%s -> %s;" % (key, element))
                    lines.append("%s -> %s [style=dotted];" % (element, visit(p)))
                    lines.append("%s -> %s;" % (element, visit(s)))
            return key

        for name, root in roots:
            lines.append('"%s" [shape=plaintext];' % name)
            lines.append('"%s" -> %s;' % (name, visit(root)))
        lines.append("}")
        return "\n".join(lines)

    def __del__(self):
        pass


def _sum(semiring, values):
    result = None
    for value in values:
        result = value if result is None else semiring.plus(result, value)
    return semiring.zero() if result is None else result


class MVSDDEvaluator(DDEvaluator):
    """Evaluator for MV-SDDs.

    Evidence is conjoined into the diagram rather than applied to the weights.  Probabilities
    are counted by the MV-SDD library; other semirings by traversing the diagram in Python.
    """

    def __init__(self, formula, semiring, weights=None, **kwargs):
        DDEvaluator.__init__(self, formula, semiring, weights, **kwargs)
        self._value_weights = None
        self._flat_weights = None
        self._true_weight = None

    def _initialize(self, with_evidence=True):
        weights = self.formula.extract_weights(self.semiring, self.given_weights)
        self._value_weights = self.formula.value_weights(weights, self.semiring)
        # A weight given to node 0 (True) scales every count.
        self._true_weight = weights[0][0] if 0 in weights else None
        self._flat_weights = None
        if self._is_probability():
            try:
                self._flat_weights = array(
                    "d", [w for ws in self._value_weights for w in ws]
                )
            except TypeError:
                pass  # Not floats: count in the semiring instead.

    def _wmc(self, node):
        mgr = self._get_manager()
        if self._flat_weights is not None:
            log_mode = isinstance(self.semiring, SemiringLogProbability)
            result = mgr.wmc(node, self._flat_weights, log_mode)
        else:
            result = mgr.wmc_semiring(node, self._value_weights, self.semiring)
        if self._true_weight is not None:
            result = self.semiring.times(result, self._true_weight)
        return result

    def _is_probability(self):
        return isinstance(self.semiring, SemiringProbability)

    def propagate(self):
        self._initialize()
        if self._is_probability():
            self.normalization = self._wmc(self._get_manager().true())
        else:
            self.normalization = None
        self.evaluate_evidence(recompute=True)

    def _evaluate_evidence(self, recompute=False):
        if self._evidence_weight is None or recompute:
            self.evidence_inode = self._get_manager().conjoin(
                self.formula.get_constraint_inode(),
                *[self.formula.get_inode(ev) for ev in self.evidence()]
            )
            result = self._wmc(self.evidence_inode)
            if self.semiring.is_zero(result):
                raise InconsistentEvidenceError(context=" during compilation")
            if self.normalization is None:
                self._evidence_weight = result
            else:
                self._evidence_weight = self.semiring.normalize(
                    result, self.normalization
                )
        return self._evidence_weight

    def evaluate(self, node):
        if node == self.formula.TRUE:
            if self.semiring.is_nsp():
                result = self.semiring.normalize(
                    self._evidence_weight, self._evidence_weight
                )
            else:
                result = self.semiring.one()
        elif node is self.formula.FALSE:
            result = self.semiring.zero()
        else:
            query = self._get_manager().conjoin(
                self.formula.get_inode(node), self.evidence_inode
            )
            result = self._wmc(query)
            if self.normalization is not None:
                result = self.semiring.normalize(result, self.normalization)
            result = self.semiring.normalize(result, self._evidence_weight)
        return self.semiring.result(result, self.formula)

    def evaluate_fact(self, node):
        return self.evaluate(node)


@transform(LogicDAG, MVSDD)
def build_mvsdd(source, destination, **kwdargs):
    """Build an MV-SDD from another formula.

    :param source: source formula
    :type source: LogicDAG
    :param destination: destination formula
    :type destination: MVSDD
    :param kwdargs: extra arguments
    :return: destination
    """
    return build_dd(source, destination, **kwdargs)
