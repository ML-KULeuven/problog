"""
problog.sdd_formula - Sentential Decision Diagrams
--------------------------------------------------

Interface to Sentential Decision Diagrams (SDD)

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

from .formula import LogicDAG, LogicFormula
from .core import transform
from .errors import InstallError
from .dd_formula import DD, build_dd, DDManager
from .util import mktempfile
import os


# noinspection PyBroadException
try:
    import sdd
except Exception as err:
    sdd = None


class SDD(DD):
    """A propositional logic formula consisting of and, or, not and atoms represented as an SDD.

    This class has two restrictions with respect to the default LogicFormula:

        * The number of atoms in the SDD should be known at construction time.
        * It does not support updatable nodes.

    This means that this class can not be used directly during grounding.
    It can be used as a target for the ``makeAcyclic`` method.
    """

    transform_preference = 10

    def __init__(self, sdd_auto_gc=False, **kwdargs):
        if sdd is None:
            raise InstallError('The SDD library is not available. Please run the installer.')
        self.auto_gc = sdd_auto_gc
        DD.__init__(self, auto_compact=False, **kwdargs)

    def _create_manager(self):
        return SDDManager(auto_gc=self.auto_gc)

    @classmethod
    def is_available(cls):
        """Checks whether the SDD library is available."""
        return sdd is not None

    def to_formula(self):
        """Extracts a LogicFormula from the SDD."""
        formula = LogicFormula(keep_order=True)

        for n, q, l in self.labeled():
            node = self.get_inode(q)
            constraints = self.get_constraint_inode()
            nodec = self.get_manager().conjoin(node, constraints)
            i = self._to_formula(formula, nodec, {})
            formula.add_name(n, i, formula.LABEL_QUERY)
        return formula

    def _to_formula(self, formula, current_node, cache=None):
        if cache is not None and int(current_node) in cache:
            return cache[int(current_node)]
        if self.get_manager().is_true(current_node):
            retval = formula.TRUE
        elif self.get_manager().is_false(current_node):
            retval = formula.FALSE
        elif sdd.sdd_node_is_literal(current_node):  # it's a literal
            lit = sdd.sdd_node_literal(current_node)
            at = self.var2atom[abs(lit)]
            node = self.get_node(at)
            if lit < 0:
                retval = -formula.add_atom(-lit, probability=node.probability, name=node.name, group=node.group)
            else:
                retval = formula.add_atom(lit, probability=node.probability, name=node.name, group=node.group)
        else:  # is decision
            size = sdd.sdd_node_size(current_node)
            elements = sdd.sdd_node_elements(current_node)
            primes = [sdd.sdd_array_element(elements, i) for i in range(0, size * 2, 2)]
            subs = [sdd.sdd_array_element(elements, i) for i in range(1, size * 2, 2)]

            # Formula: (p1^s1) v (p2^s2) v ...
            children = []
            for p, s in zip(primes, subs):
                p_n = self._to_formula(formula, p, cache)
                s_n = self._to_formula(formula, s, cache)
                c_n = formula.add_and((p_n, s_n))
                children.append(c_n)
            retval = formula.add_or(children)
        if cache is not None:
            cache[int(current_node)] = retval
        return retval



class SDDManager(DDManager):
    """
    Manager for SDDs.
    It wraps around the SDD library and offers some additional methods.
    """

    def __init__(self, varcount=0, auto_gc=True):
        """Create a new SDD manager.

        :param varcount: number of initial variables
        :type varcount: int
        :param auto_gc: use automatic garbage collection and minimization
        :type auto_gc: bool
        """
        DDManager.__init__(self)
        if varcount is None or varcount == 0:
            varcount = 1
        self.__manager = sdd.sdd_manager_create(varcount, auto_gc)
        self.varcount = varcount

    def get_manager(self):
        """Get the underlying sdd manager."""
        return self.__manager

    def add_variable(self, label=0):
        if label == 0 or label > self.varcount:
            sdd.sdd_manager_add_var_after_last(self.__manager)
            self.varcount += 1
            return self.varcount
        else:
            return label

    def literal(self, label):
        self.add_variable(abs(label))
        return sdd.sdd_manager_literal(label, self.__manager)

    def is_true(self, node):
        assert node is not None
        return sdd.sdd_node_is_true(node)

    def true(self):
        return sdd.sdd_manager_true(self.__manager)

    def is_false(self, node):
        assert node is not None
        return sdd.sdd_node_is_false(node)

    def false(self):
        return sdd.sdd_manager_false(self.__manager)

    def conjoin2(self, a, b):
        assert a is not None
        assert b is not None
        return sdd.sdd_conjoin(a, b, self.__manager)

    def disjoin2(self, a, b):
        assert a is not None
        assert b is not None
        return sdd.sdd_disjoin(a, b, self.__manager)

    def negate(self, node):
        assert node is not None
        new_sdd = sdd.sdd_negate(node, self.__manager)
        self.ref(new_sdd)
        return new_sdd

    def same(self, node1, node2):
        # Assumes SDD library always reuses equivalent nodes.
        if node1 is None or node2 is None:
            return node1 == node2
        else:
            return int(node1) == int(node2)

    def ref(self, *nodes):
        for node in nodes:
            assert node is not None
            sdd.sdd_ref(node, self.__manager)

    def deref(self, *nodes):
        for node in nodes:
            assert node is not None
            sdd.sdd_deref(node, self.__manager)

    def write_to_dot(self, node, filename):
        sdd.sdd_save_as_dot(filename, node)

    def wmc(self, node, weights, semiring, literal=None):
        logspace = 0
        if semiring.one() == 0.0:
            logspace = 1
        wmc_manager = sdd.wmc_manager_new(node, logspace, self.get_manager())
        varcount = sdd.sdd_manager_var_count(self.get_manager())
        for n in weights:
            pos, neg = weights[n]
            if n <= varcount:
                sdd.wmc_set_literal_weight(n, pos, wmc_manager)  # Set positive literal weight
                sdd.wmc_set_literal_weight(-n, neg, wmc_manager)  # Set negative literal weight
        result = sdd.wmc_propagate(wmc_manager)
        if literal is not None:
            result = sdd.wmc_literal_pr(literal, wmc_manager)
        sdd.wmc_manager_free(wmc_manager)
        return result

    def wmc_literal(self, node, weights, semiring, literal):
        return self.wmc(node, weights, semiring, literal)

    def wmc_true(self, weights, semiring):
        return self.wmc(self.true(), weights, semiring)

    def __del__(self):
        # if sdd is not None and sdd.sdd_manager_free is not None:
        #     sdd.sdd_manager_free(self.__manager)
        self.__manager = None

    def __getstate__(self):
        tempfile = mktempfile()
        vtree = sdd.sdd_manager_vtree(self.get_manager())
        sdd.sdd_vtree_save(tempfile, vtree)
        with open(tempfile) as f:
            vtree_data = f.read()

        nodes = []
        for n in self.nodes:
            if n is not None:
                sdd.sdd_save(tempfile, n)

                with open(tempfile) as f:
                    nodes.append(f.read())
            else:
                nodes.append(None)

        sdd.sdd_save(tempfile, self.constraint_dd)
        with open(tempfile) as f:
            constraint_dd = f.read()

        os.remove(tempfile)
        return {'varcount': self.varcount, 'nodes': nodes, 'vtree': vtree_data, 'constraint_dd': constraint_dd}

    def __setstate__(self, state):
        self.nodes = []
        self.varcount = state['varcount']
        tempfile = mktempfile()
        with open(tempfile, 'w') as f:
            f.write(state['vtree'])
        vtree = sdd.sdd_vtree_read(tempfile)
        self.__manager = sdd.sdd_manager_new(vtree)

        for n in state['nodes']:
            if n is None:
                self.nodes.append(None)
            else:
                with open(tempfile, 'w') as f:
                    f.write(n)
                self.nodes.append(sdd.sdd_read(tempfile, self.__manager))

        with open(tempfile, 'w') as f:
            f.write(state['constraint_dd'])
        self.constraint_dd = sdd.sdd_read(tempfile, self.__manager)
        os.remove(tempfile)
        return


@transform(LogicDAG, SDD)
def build_sdd(source, destination, **kwdargs):
    """Build an SDD from another formula.

    :param source: source formula
    :param destination: destination formula
    :param kwdargs: extra arguments
    :return: destination
    """
    return build_dd(source, destination, **kwdargs)
