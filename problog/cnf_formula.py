"""

problog.cnf_formula - CNF
-------------------------

Provides access to CNF and weighted CNF.

..
    Part of the ProbLog distribution.

    Copyright 2015 KU Leuven, DTAI Research Group

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
from __future__ import print_function

from .formula import BaseFormula, LogicDAG

from .core import transform
from .util import Timer

from .evaluator import SemiringLogProbability


class CNF(BaseFormula):
    """A logic formula in Conjunctive Normal Form."""

    # noinspection PyUnusedLocal
    def __init__(self, **kwdargs):
        BaseFormula.__init__(self)
        self._clauses = []        # All clauses in the CNF (incl. comment)
        self._clausecount = 0     # Number of actual clauses (not incl. comment)

    # noinspection PyUnusedLocal
    def add_atom(self, atom, force=False):
        """Add an atom to the CNF.

        :param atom: name of the atom
        :param force: add a clause for each atom to force it's existence in the final CNF
        """
        self._atomcount += 1
        if force:
            self._clauses.append([atom, -atom])
            self._clausecount += 1

    def add_comment(self, comment):
        """Add a comment clause.

        :param comment: text of the comment
        """
        self._clauses.append(['c', comment])

    def add_clause(self, head, body):
        """Add a clause to the CNF.

        :param head: head of the clause (i.e. atom it defines)
        :param body: body of the clause
        """
        self._clauses.append([head] + list(body))
        self._clausecount += 1

    def add_constraint(self, constraint, force=False):
        """Add a constraint.

        :param constraint: constraint to add
        :param force: force constraint to be true even though none of its values are set
        :type constraint: problog.constraint.Constraint
        """
        BaseFormula.add_constraint(self, constraint)
        for c in constraint.as_clauses():
            self.add_clause(force, c)

    def _clause2str(self, clause, weighted=False):
        if weighted:
            raise NotImplementedError()
        else:
            if clause[1] is None:
                return ' '.join(map(str, clause[2:])) + ' 0'
            else:
                return ' '.join(map(str, clause[1:])) + ' 0'

    def to_dimacs(self, partial=False, weighted=False, semiring=None, smart_constraints=False, names=False):
        """Transform to a string in DIMACS format.

        :param partial: split variables if possibly true / certainly true
        :param weighted: created a weighted (False, :class:`int`, :class:`float`)
        :param semiring: semiring for weight transformation (if weighted)
        :param names: Print names in comments
        :return: string in DIMACS format
        """
        if weighted:
            t = 'wcnf'
        else:
            t = 'cnf'

        header, content = self._contents(partial=partial, weighted=weighted,
                                         semiring=semiring, smart_constraints=smart_constraints)

        result = 'p %s %s\n' % (t, ' '.join(map(str, header)))
        if names:
            tpl = 'c {{:<{}}} {{}}\n'.format(len(str(self._atomcount)) + 1)
            for n, i, l in self.get_names_with_label():
                result += tpl.format(i, n)
        result += '\n'.join(map(lambda cl: ' '.join(map(str, cl)) + ' 0', content))
        return result

    def to_lp(self, partial=False, semiring=None, smart_constraints=False):
        """Transfrom to CPLEX lp format (MIP program).
        This is always weighted.

        :param partial: split variables in possibly true / certainly true
        :param semiring: semiring for weight transformation (if weighted)
        :param smart_constraints: only enforce constraints when variables are set
        :return: string in LP format
        """
        header, content = self._contents(partial=partial, weighted=False,
                                         semiring=semiring, smart_constraints=smart_constraints)

        if semiring is None:
            semiring = SemiringLogProbability()

        var2str = lambda var: 'x%s' % var if var > 0 else '-x%s' % -var

        if partial:
            ct = lambda it: 2 * it
            pt = lambda it: ct(it) - 1

            weights = self.extract_weights(semiring)
            objective = []
            for v in range(0, self.atomcount + 1):
                w_pos, w_neg = weights.get(v, (semiring.one(), semiring.one()))
                if not semiring.is_one(w_pos):
                    w_ct = w_pos
                    objective.append('%s x%s' % (w_ct, ct(v)))
                if not semiring.is_one(w_neg):
                    w_pt = -w_neg
                    objective.append('%s x%s' % (w_pt, pt(v)))
            objective = ' + '.join(objective)

        else:
            weights = {}
            for i, w in self.extract_weights(semiring).items():
                w = w[0] - w[1]
                if w != 0:
                    weights[i] = str(w)

            objective = ' + '.join(['%s x%s' % (weights[i], i)
                                    for i in range(0, self.atomcount + 1) if i in weights])
        result = 'maximize\n'
        result += '    obj:' + objective + '\n'
        result += 'subject to\n'
        for clause in content:
            n_neg = len([c for c in clause if c < 0])
            result += '    ' + ' + '.join(map(var2str, clause)) + ' >= ' + str(1 - n_neg) + '\n'
        result += 'bounds\n'
        for i in range(1, self.atomcount + 1):
            result += '    0 <= x%s <= 1\n' % i
        result += 'binary\n'
        result += '    ' + ' '.join(map(var2str, range(1, self.atomcount + 1))) + '\n'
        result += 'end\n'
        return result

    def _contents(self, partial=False, weighted=False, semiring=None, smart_constraints=False):
        # Helper function to determine the certainly true / possibly true names (for partial)

        ct = lambda i: 2 * i
        pt = lambda i: ct(i) - 1
        cpt = lambda i: -pt(-i) if i < 0 else ct(i)

        w_mult = 1
        w_max = []
        weights = None
        if weighted == int:
            w_mult = 10000
            w_min = -10000
            wt = lambda w: int(max(w_min, w) * w_mult)
        elif weighted == float:
            w_mult = 1
            w_min = -10000
            wt = lambda w: w
        elif weighted:
            w_min = -10000
            w_mult = 10000
            wt = lambda w: int(max(w_min, w) * w_mult)

        if weighted:
            if semiring is None:
                semiring = SemiringLogProbability()
            weights = self.extract_weights(semiring)

            w_sum = 0.0
            for w_pos, w_neg in weights.values():
                w_sum += max(w_pos, w_min) + max(w_neg, w_min)
            w_max = [int(-w_sum * w_mult) + 1]

        atomcount = self.atomcount
        if partial:
            atomcount *= 2
        clausecount = self.clausecount
        if partial:
            clausecount += atomcount

        clauses = []

        if partial:
            # For each atom: add constraint
            for a in range(1, self.atomcount + 1):
                clauses.append(w_max + [pt(a), -ct(a)])

                if weighted:
                    w_pos, w_neg = weights.get(a, (semiring.one(), semiring.one()))
                    if not semiring.is_one(w_pos):
                        clauses.append([-wt(w_pos), -ct(a)])
                    if not semiring.is_one(w_neg):
                        clauses.append([-wt(w_neg), pt(a)])

            # For each clause:
            for c in self.clauses:
                head, body = c[0], c[1:]
                if type(head) != bool:
                    # Clause does not represent a constraint.
                    head_neg = (head < 0)
                    head = abs(head)
                    head1, head2 = ct(head), pt(head)
                    if head_neg:
                        head1, head2 = -head1, -head2
                    clauses.append(w_max + [head1, head2] + list(map(cpt, body)))
                elif smart_constraints and not head:
                    # It's a constraint => add an indicator variable.
                    # a \/ -b ===> -pt(a) \/ I  => for all
                    atomcount += 1
                    ind = atomcount
                    v = []
                    for b in body:
                        clauses.append(w_max + [-ct(abs(b)), ind])
                        clauses.append(w_max + [pt(abs(b)), ind])
                        v += [ct(abs(b)), -pt(abs(b))]
                    clauses.append(w_max + v + [-ind])
                    clauses.append(w_max + list(map(cpt, body)) + [-ind])
                else:
                    clauses.append(w_max + list(map(cpt, body)))
        else:
            if weighted:
                for a in range(1, self.atomcount + 1):
                    w_pos, w_neg = weights.get(a, (semiring.one(), semiring.one()))
                    if not semiring.is_one(w_pos):
                        clauses.append([-wt(w_pos), -a])
                    if not semiring.is_one(w_neg):
                        clauses.append([-wt(w_neg), a])
            for c in self.clauses:
                head, body = c[0], c[1:]
                if head is None or type(head) == bool and not head:
                    clauses.append(w_max + list(body))
                else:
                    clauses.append(w_max + [head] + list(body))

        return [atomcount, len(clauses)] + w_max, clauses

    def from_partial(self, atoms):
        """Translates a (complete) conjunction in the partial formula back to the complete formula.

        For example: given an original formula with one atom '1',
         this atom is translated to two atoms '1' (pt) and '2' (ct).

        The possible conjunctions are:

            * [1, 2]    => [1]  certainly true (and possibly true) => true
            * [-1, -2]  => [-1] not possibly true (and certainly true) => false
            * [1, -2]   => []   possibly true but not certainly true => unknown
            * [-1, 2]   => INVALID   certainly true but not possible => invalid (not checked)

        :param atoms: complete list of atoms in partial CNF
        :return: partial list of atoms in full CNF
        """
        result = []
        for s in atoms:
            if s % 2 == 1 and s < 0:
                r = (abs(s)+1)//2
                if r in self.get_weights():
                    result.append(-r)
            elif s % 2 == 0 and s > 0:
                r = (abs(s)+1)//2
                if r in self.get_weights():
                    result.append(r)
        return result

    def is_trivial(self):
        """Checks whether the CNF is trivial (i.e. contains no clauses)"""
        return self.clausecount == 0

    @property
    def clauses(self):
        """Return the list of clauses"""
        return self._clauses

    @property
    def clausecount(self):
        """Return the number of clauses"""
        return self._clausecount


# noinspection PyUnusedLocal
@transform(LogicDAG, CNF)
def clarks_completion(source, destination, force_atoms=False, **kwdargs):
    """Transform an acyclic propositional program to a CNF using Clark's completion.

    :param source: acyclic program to transform
    :param destination: target CNF
    :param kwdargs: additional options (ignored)
    :return: destination
    """
    with Timer('Clark\'s completion'):
        # Each rule in the source formula will correspond to an atom.
        num_atoms = len(source)

        # Copy weight information.
        destination.set_weights(source.get_weights())

        # Add atoms.
        for i in range(0, num_atoms):
            destination.add_atom(i+1, force=force_atoms)

        # Complete other nodes
        # Note: assumes negation is encoded as negative number.
        for index, node, nodetype in source:
            if nodetype == 'conj':
                destination.add_clause(index, list(map(lambda x: -x, node.children)))
                for c in node.children:
                    destination.add_clause(-index, [c])
            elif nodetype == 'disj':
                destination.add_clause(-index, node.children)
                for c in node.children:
                    destination.add_clause(index, [-c])
            elif nodetype == 'atom':
                pass
            else:
                raise ValueError("Unexpected node type: '%s'" % nodetype)

        # Copy constraints.
        for c in source.constraints():
            destination.add_constraint(c)

        # Copy node names.
        for n, i, l in source.get_names_with_label():
            destination.add_name(n, i, l)

        return destination
