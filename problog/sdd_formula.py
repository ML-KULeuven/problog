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
from collections import namedtuple

from .formula import LogicDAG, LogicFormula, LogicNNF
from .core import transform
from .errors import InstallError, InconsistentEvidenceError
from .dd_formula import DD, build_dd, DDManager, DDEvaluator
from .evaluator import FormulaEvaluatorNSP, SemiringLogProbability, SemiringProbability

from .util import mktempfile
import os

# noinspection PyBroadException
try:
    from pysdd import sdd
    from pysdd.iterator import SddIterator
    from pysdd.util import sdd_to_dot
    from pysdd.sdd import Vtree
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

    def __init__(
        self, sdd_auto_gc=False, var_constraint=None, init_varcount=-1, **kwdargs
    ):
        """
        Create an SDD

        :param sdd_auto_gc: Auto garbage collection and minimize (during disjoin and conjoin)
        :param var_constraint: A variable ordering constraint. Currently only x_constrained namedtuple are allowed.
        :type var_constraint: x_constrained
        :param init_varcount: The amount of variables to initialize the manager with.
        :param kwdargs:
        :raise InstallError: When the SDD library is not available.
        """
        if sdd is None:
            raise InstallError(
                "The SDD library is not available. Please install the PySDD package."
            )
        self.auto_gc = sdd_auto_gc
        self._var_constraint = var_constraint
        self._init_varcount = init_varcount
        DD.__init__(self, auto_compact=False, **kwdargs)

    @property
    def var_constraint(self):
        return self._var_constraint

    @var_constraint.setter
    def var_constraint(self, var_constraint):
        """
        Set the variable constraint
        :param var_constraint: A variable ordering constraint. Currently only x_constrained namedtuple are allowed.
        :type var_constraint: x_constrained
        """
        self._var_constraint = var_constraint

    @property
    def init_varcount(self):
        return self._init_varcount

    @init_varcount.setter
    def init_varcount(self, value=0):
        """Set the varcount with which to initialise the manager. Only call before calling the manager """
        assert self.inode_manager is None
        self._init_varcount = value

    def _create_manager(self):
        return SDDManager(
            auto_gc=self.auto_gc,
            var_constraint=self.var_constraint,
            varcount=self.init_varcount,
        )

    def _create_evaluator(self, semiring, weights, **kwargs):
        return SDDEvaluator(self, semiring, weights, **kwargs)

    @classmethod
    def is_available(cls):
        """Checks whether the SDD library is available."""
        return sdd is not None

    def to_internal_dot(self, node=None):
        """
        SDD for the given node, formatted for use with Graphviz dot.

        :param node: The node to get the dot from.
        :type node: SddNode
        :return: The dot format of the given node. When node is None, the shared_sdd will be used (contains all active
        sdd structures).
        :rtype: str
        """
        return self.get_manager().to_internal_dot(node=node)

    def sdd_to_dot(self, node, litnamemap=None, show_id=False, merge_leafs=False):
        """
        SDD for the given node, formatted for use with Graphviz dot. This method provides more control over the used
        symbols than to_internal_dot (see litnamemap). Primes are given by a dotted line, subs by a full line.

        :param node: The node to get the dot from.
        :type node: SddNode
        :param litnamemap: A dictionary providing the symbols to use. The following options are available:
            1. literals, e.g. {1:'A', -1:'-A', ...},
            2. True/False, e.g. {true':'1', 'false':'0'}
            3. And/Or e.g. {'mult':'x', 'add':'+'}
            When litnamemap = True, self.get_litnamemap() will be used.
        :type litnamemap: dict[(int | str), str] | bool | None
        :param show_id: Whether to display the ids of each sdd node.
        :param merge_leafs: Whether to merge the same leaf nodes. True results in less nodes but makes it harder to
        render without having crossing lines.
        :return: The dot format of the given node. When node is None, this mgr is used instead.
        :rtype: str
        """
        if litnamemap is True:
            litnamemap = self.get_litnamemap()
        return self.get_manager().sdd_to_dot(
            node=node, litnamemap=litnamemap, show_id=show_id, merge_leafs=merge_leafs
        )

    def get_litnamemap(self):
        """ Get a dictionary mapping literal IDs (inode index) to names. e.g; {1:'x', -1:'-x'}"""
        litnamemap = dict()
        var_count = self.get_manager().varcount
        for (name, index) in self.get_names():
            inode_index = self.atom2var.get(index, -1)
            if 0 <= inode_index < var_count:
                litnamemap[inode_index] = name
                litnamemap[-inode_index] = "-{}".format(name)
        return litnamemap

    def to_formula(self):
        """Extracts a LogicFormula from the SDD."""
        formula = LogicFormula(keep_order=True)

        for n, q, l in self.labeled():
            node = self.get_inode(q)
            constraints = self.get_constraint_inode()
            nodec = self.get_manager().conjoin(node, constraints)
            i = self._to_formula(formula, nodec, {})
            formula.add_name(n, i, l)
        return formula

    def _to_formula(self, formula, current_node, cache=None):
        if cache is not None and current_node.id in cache:
            return cache[current_node.id]
        if self.get_manager().is_true(current_node):
            retval = formula.TRUE
        elif self.get_manager().is_false(current_node):
            retval = formula.FALSE
        elif current_node.is_literal():  # it's a literal
            lit = current_node.literal
            at = self.var2atom[abs(lit)]
            node = self.get_node(at)
            if lit < 0:
                retval = -formula.add_atom(
                    -lit,
                    probability=node.probability,
                    name=node.name,
                    group=node.group,
                    cr_extra=False,
                )
            else:
                retval = formula.add_atom(
                    lit,
                    probability=node.probability,
                    name=node.name,
                    group=node.group,
                    cr_extra=False,
                )
        else:  # is decision
            # Formula: (p1^s1) v (p2^s2) v ...
            children = []
            for p, s in current_node.elements():
                p_n = self._to_formula(formula, p, cache)
                s_n = self._to_formula(formula, s, cache)
                c_n = formula.add_and((p_n, s_n))
                children.append(c_n)
            retval = formula.add_or(children)
        if cache is not None:
            cache[current_node.id] = retval
        return retval


class SDDManager(DDManager):
    """
    Manager for SDDs.
    It wraps around the SDD library and offers some additional methods.
    """

    def __init__(self, varcount=0, auto_gc=False, var_constraint=None):
        """Create a new SDD manager.

        :param varcount: number of initial variables
        :type varcount: int
        :param auto_gc: use automatic garbage collection and minimization
        :type auto_gc: bool
        :param var_constraint: A variable ordering constraint. Currently only x_constrained namedtuple are allowed.
        :type var_constraint: x_constrained
        """
        DDManager.__init__(self)
        if varcount is None or varcount <= 0:
            varcount = 1
        vtree = None
        if var_constraint is not None and varcount > 1:
            x_constraint = self._to_x_constrained_list(varcount, var_constraint)
            vtree = Vtree.new_with_X_constrained(
                var_count=varcount, is_X_var=x_constraint, vtree_type="balanced"
            )

        self.__manager = sdd.SddManager(
            var_count=varcount, auto_gc_and_minimize=auto_gc, vtree=vtree
        )
        self._assigned_varcount = 0

    def _to_x_constrained_list(self, varcount, var_constraint):
        """
        Convert the X-constrained var_constraint into a list of size varcount+1 specifying variables X. For variables i
        where 1 ≤ i ≤ varcount, if is_X_var[i] is 1 then i is in X, and if it is 0 then i is not in X.
        :param varcount: The amount of variables the vtree must have. This must be at least as high as the highest
            number in var_constraint.
        :param var_constraint: The X-constrained variable constraint
        :type var_constraint: x_constrained
        :return: The is_x_var input for the X-constrained vTree
        """
        is_x_var = [0] * (varcount + 1)
        if var_constraint is not None and len(var_constraint) > 0:
            for x in var_constraint.X:
                is_x_var[x] = 1
        return is_x_var

    def get_manager(self):
        """Get the underlying sdd manager."""
        return self.__manager

    @property
    def varcount(self):
        return self.get_manager().var_count()

    @property
    def assigned_varcount(self):
        return self._assigned_varcount

    def add_variable(self, label=0):
        if label == 0 or label > self.assigned_varcount:
            self._assigned_varcount += 1
            if self.assigned_varcount > self.varcount:
                self.get_manager().add_var_after_last()

            return self.assigned_varcount
        else:
            return label

    def literal(self, label):
        self.add_variable(abs(label))
        return self.get_manager().literal(label)

    def is_true(self, node):
        assert node is not None
        return node.is_true()

    def true(self):
        return self.get_manager().true()

    def is_false(self, node):
        assert node is not None
        return node.is_false()

    def false(self):
        return self.get_manager().false()

    def conjoin2(self, a, b):
        assert a is not None
        assert b is not None
        return self.get_manager().conjoin(a, b)

    def set_auto_gc_and_minimize(self, set_to=True):
        if set_to:
            self.__manager.auto_gc_and_minimize_on()
        else:
            self.__manager.auto_gc_and_minimize_off()

    def is_auto_gc_and_minimize_on(self):
        return self.__manager.is_auto_gc_and_minimize_on()

    def disjoin2(self, a, b):
        assert a is not None
        assert b is not None
        return self.get_manager().disjoin(a, b)

    def negate(self, node):
        assert node is not None
        new_sdd = self.get_manager().negate(node)
        self.ref(new_sdd)
        return new_sdd

    def same(self, node1, node2):
        # Assumes SDD library always reuses equivalent nodes.
        if node1 is None or node2 is None:
            return node1 == node2
        else:
            return node1.id == node2.id

    def ref(self, *nodes):
        for node in nodes:
            assert node is not None
            node.ref()

    def deref(self, *nodes):
        for node in nodes:
            assert node is not None
            node.deref()

    def write_to_dot(self, node, filename, litnamemap=None):
        if litnamemap is None:
            self.get_manager().save_as_dot(filename.encode(), node)
        else:
            with open(filename, "w") as file:
                file.write(self.sdd_to_dot(node=node, litnamemap=litnamemap))

    def to_internal_dot(self, node=None):
        """
        SDD for the given node, formatted for use with Graphviz dot.

        :param node: The node to get the dot from.
        :type node: SddNode
        :return: The dot format of the given node. When node is None, the shared_sdd will be used (contains all active
        sdd structures).
        :rtype: str
        """
        return self.get_manager().dot(node=node)

    def sdd_to_dot(self, node, litnamemap=None, show_id=False, merge_leafs=False):
        """
        SDD for the given node, formatted for use with Graphviz dot. This method provides more control over the used
        symbols than to_internal_dot (see litnamemap). Primes are given by a dotted line, subs by a full line.

        :param node: The node to get the dot from.
        :param litnamemap: A dictionary providing the symbols to use. The following options are available:
            1. literals, e.g. {1:'A', -1:'-A', ...},
            2. True/False, e.g. {true':'1', 'false':'0'}
            3. And/Or e.g. {'mult':'x', 'add':'+'}
        :type litnamemap: dict[(int | str), str] | None
        :param show_id: Whether to display the ids of each sdd node.
        :param merge_leafs: Whether to merge the same leaf nodes. True results in less nodes but makes it harder to
        render without having crossing lines.
        :return: The dot format of the given node. When node is None, the mgr is used (this behavior can be overriden).
        :rtype: str
        """
        used_node = node if node is not None else self.get_manager()
        return sdd_to_dot(
            node=used_node,
            litnamemap=litnamemap,
            show_id=show_id,
            merge_leafs=merge_leafs,
        )

    def wmc(
        self,
        node,
        weights,
        semiring,
        literal=None,
        pr_semiring=True,
        perform_smoothing=True,
        smooth_to_root=False,
        wmc_func=None,
    ):
        """Perform Weighted Model Count on the given node or the given literal.

        Common usage: wmc(node, weights, semiring) and wmc(node, weights, semiring, smooth_to_root=True)

        :param node: node to evaluate Type: SddNode
        :param weights: weights for the variables in the node. Type: {literal_id : (pos_weight, neg_weight)}
        :param semiring: use the operations defined by this semiring. Type: Semiring
        :param literal: When a literal is given, the result of WMC(literal) is returned instead.
        :param pr_semiring: Whether the given semiring is a (logspace) probability semiring.
        :param perform_smoothing: Whether to perform smoothing. When pr_semiring is True, smoothing is performed
            regardless.
        :param smooth_to_root: Whether to perform smoothing compared to the root. When pr_semiring is True, smoothing
            compared to the root is not performed regardless of this flag.
        :param wmc_func: The WMC function to use. If None, a built_in one will be used that depends on the given
            semiring. Type: function[SddNode, List[Tuple[prime_weight, sub_weight, Set[prime_used_lit],
            Set[sub_used_lit]]], Set[expected_prime_lit], Set[expected_sub_lit]] -> weight
        :type weights: dict[int, tuple[Any, Any]]
        :type semiring: Semiring
        :type pr_semiring: bool
        :type perform_smoothing: bool
        :type smooth_to_root: bool
        :type wmc_func: function
        :return: weighted model count of node if literal=None, else the weights are propagated up to node but the
            weighted model count of literal is returned.
        """
        varcount = self.get_manager().var_count()

        if pr_semiring and wmc_func is None:  # library built_in (WmcManager)
            logspace = 0
            if semiring.one() == 0.0:
                logspace = 1

            # setup
            wmc_manager = sdd.WmcManager(node, log_mode=logspace)
            for (
                n
            ) in weights:  # TODO wmc_manager.set_literal_weights_from_array is faster
                pos, neg = weights[n]
                if n <= varcount:
                    wmc_manager.set_literal_weight(n, pos)
                    wmc_manager.set_literal_weight(-n, neg)
            # Cover edge case e.g. node=SddNode(True)
            if varcount == 1 and weights.get(1) is None:
                wmc_manager.set_literal_weight(1, semiring.one())
                wmc_manager.set_literal_weight(-1, semiring.zero())

            # Calculate result
            result = wmc_manager.propagate()
            if literal is not None:
                result = wmc_manager.literal_pr(literal)
            if weights.get(0) is not None:  # Times the weight of True
                result = result * weights[0][0]
        else:  # manual iteration (SddIterator)
            if wmc_func is None:
                wmc_func = self._get_wmc_func(
                    weights=weights,
                    semiring=semiring,
                    perform_smoothing=perform_smoothing,
                )

            # Cover edge case e.g. node=SddNode(True)
            modified_weights = False
            if varcount == 1 and weights.get(1) is None:
                modified_weights = True
                weights[1] = (
                    semiring.one(),
                    semiring.zero(),
                )  # because 1 + 0 = 1 and 1 * x = x

            # Calculate result
            query_node = (
                node if literal is None else self.get_manager().literal(literal)
            )
            sdd_iterator = SddIterator(
                self.get_manager(), smooth_to_root=smooth_to_root
            )
            result = sdd_iterator.depth_first(query_node, wmc_func)

            if weights.get(0) is not None:  # Times the weight of True
                result = semiring.times(result, weights[0][0])

            # Restore edge case modification
            if modified_weights:
                weights.pop(1)

        self.get_manager().set_prevent_transformation(prevent=False)
        return result

    @staticmethod
    def _get_wmc_func(weights, semiring, perform_smoothing=True):
        """
        Get the function used to perform weighted model counting with the SddIterator. Smoothing supported.

        :param weights: The weights used during computations.
        :type weights: dict[int, tuple[Any, Any]]
        :param semiring: The semiring used for the operations.
        :param perform_smoothing: Whether smoothing must be performed. If false but semiring.is_nsp() then
            smoothing is still performed.
        :return: A WMC function that uses the semiring operations and weights, Performs smoothing if needed.
        """

        smooth_flag = perform_smoothing or semiring.is_nsp()

        def func_weightedmodelcounting(
            node, rvalues, expected_prime_vars, expected_sub_vars
        ):
            """ Method to pass on to SddIterator's ``depth_first`` to perform weighted model counting."""
            if rvalues is None:
                # Leaf
                if node.is_true():
                    result_weight = semiring.one()

                    # If smoothing, go over literals missed in scope
                    if smooth_flag:
                        missing_literals = (
                            expected_prime_vars
                            if expected_prime_vars is not None
                            else set()
                        )
                        missing_literals |= (
                            expected_sub_vars
                            if expected_sub_vars is not None
                            else set()
                        )

                        for missing_literal in missing_literals:
                            missing_pos_weight, missing_neg_weight = weights[
                                missing_literal
                            ]
                            missing_combined_weight = semiring.plus(
                                missing_pos_weight, missing_neg_weight
                            )
                            result_weight = semiring.times(
                                result_weight, missing_combined_weight
                            )

                    return result_weight

                elif node.is_false():
                    return semiring.zero()

                elif node.is_literal():
                    p_weight, n_weight = weights.get(abs(node.literal))
                    result_weight = p_weight if node.literal >= 0 else n_weight

                    # If smoothing, go over literals missed in scope
                    if smooth_flag:
                        lit_scope = {abs(node.literal)}

                        if expected_prime_vars is not None:
                            missing_literals = expected_prime_vars.difference(lit_scope)
                        else:
                            missing_literals = set()
                        if expected_sub_vars is not None:
                            missing_literals |= expected_sub_vars.difference(lit_scope)

                        for missing_literal in missing_literals:
                            missing_pos_weight, missing_neg_weight = weights[
                                missing_literal
                            ]
                            missing_combined_weight = semiring.plus(
                                missing_pos_weight, missing_neg_weight
                            )
                            result_weight = semiring.times(
                                result_weight, missing_combined_weight
                            )

                    return result_weight

                else:
                    raise Exception("Unknown leaf type for node {}".format(node))
            else:
                # Decision node
                if node is not None and not node.is_decision():
                    raise Exception("Expected a decision node for node {}".format(node))

                result_weight = None
                for prime_weight, sub_weight, prime_vars, sub_vars in rvalues:
                    branch_weight = semiring.times(prime_weight, sub_weight)

                    # If smoothing, go over literals missed in scope
                    if smooth_flag:
                        missing_literals = expected_prime_vars.difference(
                            prime_vars
                        ) | expected_sub_vars.difference(sub_vars)
                        for missing_literal in missing_literals:
                            missing_pos_weight, missing_neg_weight = weights[
                                missing_literal
                            ]
                            missing_combined_weight = semiring.plus(
                                missing_pos_weight, missing_neg_weight
                            )
                            branch_weight = semiring.times(
                                branch_weight, missing_combined_weight
                            )

                    # Add to current intermediate result
                    if result_weight is not None:
                        result_weight = semiring.plus(result_weight, branch_weight)
                    else:
                        result_weight = branch_weight

                return result_weight

        return func_weightedmodelcounting

    def wmc_literal(self, node, weights, semiring, literal):
        return self.wmc(node, weights, semiring, literal)

    def wmc_true(self, weights, semiring):
        return self.wmc(self.true(), weights, semiring)

    def count(self):
        return self.get_manager().count()

    def get_deepcopy_noref(
        self,
    ):  # TODO might be cleaner to maintain refcounts and deref everything afterwards in SDDExplicit
        """
        Get a deep copy of this without reference counts to inodes.
        Notes: No inode will have a reference count and auto_gc_and_minimize will be disabled.
        :return: A deep copy of this without any reference counts.
        """
        new_mgr = SDDManager(varcount=self.varcount, auto_gc=False)
        new_mgr._assigned_varcount = self.assigned_varcount

        # Code that is slower but works (see commented out code)
        mapping = self._copy_internal_SDD_to_noref(new_mgr=new_mgr, cache=None)

        # Construct node list based on current + mapping
        new_nodes = [None] * len(self.nodes)
        for i in range(0, len(self.nodes)):
            node = self.nodes[i]
            if node is not None:
                new_nodes[i] = mapping.get(self.nodes[i].id, None)

        new_mgr.nodes = new_nodes

        if self.constraint_dd is not None:
            new_mgr.constraint_dd = mapping.get(self.constraint_dd.id, None)

        # Old code which can be more efficient if we fix the "OSError Cannot allocate memory"
        # new_nodes = self.nodes.copy()
        # # fill in None spots with inode True
        # none_indices = []
        # true_node = self.true()
        # for i in range(0, len(new_nodes)):
        #     if new_nodes[i] is None:
        #         new_nodes[i] = true_node
        #         none_indices.append(i)
        # # push constraint node
        # if self.constraint_dd is not None:
        #     new_nodes.append(self.constraint_dd)
        #
        # # copy
        # new_mgr.__manager = self.get_manager().copy(new_nodes)
        #
        # # pop constraint node
        # if self.constraint_dd is not None:
        #     new_mgr.constraint_dd = new_nodes.pop()
        # # put back None
        # for i in none_indices:
        #     new_nodes[i] = None
        #
        # new_mgr.nodes = new_nodes

        # DEBUG - print before and after ref_counts
        # print("before %s" % [(node.ref_count() if node is not None else None) for node in self.nodes])
        # print("after %s" % [(node.ref_count() if node is not None else None) for node in new_mgr.nodes])

        return new_mgr

    def _copy_internal_SDD_to_noref(self, new_mgr, cache=None):
        """
        Copy the SDD structure of this manager to the given new_mgr. ALl refcounts will be 0 so the auto_gc on the
        new_mgr should be disabled.

        :param new_mgr: The manager to copy this SDD structure to. Note: the var_count must be high enough.
        :type new_mgr: SDDManager
        :param cache: The dictionary to use for the end result (mapping). If None, an empty one will be created.
        :type cache: dict[int, SddNode]
        :return: A mapping (cache) of {id, SddNode}. More specifically, {x:y} where x is the ID of an SddNode in this
            manager which corresponds to the newly created SddNode y of the new_mgr.
        :rtype: dict[int, SddNode]
        """
        if cache is None:
            cache = dict()
        root_nodes = self.nodes

        for root in root_nodes:
            if root is not None and root.ref_count() > 0:
                self._copy_internal_SDD_to_noref_aux(new_mgr, cache, root)
        return cache

    def _copy_internal_SDD_to_noref_aux(self, new_mgr, nodes_cache, node):
        """
        Copy node of the current SDDManager into new_mgr using nodes_cache to map (cache) SDDNode equivalences between
        the SDDNodes (id) of this manager to the SDDNodes in the new_mgr. This is an auxiliary method for
        self._copy_internal_SDD_to_noref(...).

        :param new_mgr: The new manager
        :type new_mgr: SDDManager
        :param nodes_cache: The cache to use to retrieve corresponding nodes between both managers. Has type
        {id: SddNode}. Must not be None.
        :type nodes_cache: dict[int, SddNode]
        :param node: The SddNode in this manager to copy over to new_mgr.
        :type node: SddNode
        :return: The SDDNode in new_mgr corresponding to node in this SDDManager.
        :rtype: SddNode
        """

        # cached_node: (SddNode x, SddNode y, int z)
        # with # x = node to process, y = x processed upto index z)
        cached_node = nodes_cache.get(node.id, None)
        if cached_node is not None:
            return cached_node

        stack = [(node, new_mgr.false(), -1)]
        while len(stack):
            process_node, inter_result, inter_index = stack.pop()

            if process_node.is_true():
                nodes_cache[process_node.id] = new_mgr.true()
                continue
            elif process_node.is_false():
                nodes_cache[process_node.id] = new_mgr.false()
                continue
            elif process_node.is_literal():
                new_node = new_mgr.literal(process_node.literal)
                nodes_cache[process_node.id] = new_node
                continue
            else:
                index = -1
                or_node = inter_result
                completed_for = True
                for p, s in process_node.elements():
                    # skip to inter_result
                    index = index + 1
                    if index <= inter_index:
                        continue

                    new_p = nodes_cache.get(p.id, None)
                    if new_p is None:
                        stack.append((process_node, or_node, index - 1))
                        stack.append((p, new_mgr.false(), -1))
                        completed_for = False
                        break

                    new_s = nodes_cache.get(s.id, None)
                    if new_s is None:
                        stack.append((process_node, or_node, index - 1))
                        stack.append((s, new_mgr.false(), -1))
                        completed_for = False
                        break

                    conjoined = new_mgr.conjoin(new_p, new_s)
                    or_node_n = new_mgr.disjoin(or_node, conjoined)
                    or_node_n.deref()  # ensure refcount = 0
                    conjoined.deref()  # ensure refcount = 0
                    # conjoined.deref()
                    # or_node.deref()
                    or_node = or_node_n
                if completed_for:
                    nodes_cache[process_node.id] = or_node

        return nodes_cache[node.id]

    def _copy_internal_SDD_to_aux_noref_rec(self, new_mgr, nodes_cache, node):
        """
        Copy node of the current SDDManager into new_mgr using nodes_cache to map (cache) SDDNode equivalences between
        the SDDNodes (id) of this manager to the SDDNodes in the new_mgr. This is an auxiliary method for
        self._copy_internal_SDD_to_noref(...) which uses recursion.

        :param new_mgr: The new manager
        :type new_mgr: SDDManager
        :param nodes_cache: The cache to use to retrieve corresponding nodes between both managers. Has type
        {id: SddNode}. Must not be None.
        :type nodes_cache: dict[int, SddNode]
        :param node: The SddNode in this manager to copy over to new_mgr.
        :type node: SddNode
        :return: The SDDNode in new_mgr corresponding to node in this SDDManager.
        :rtype: SddNode
        """
        cached = nodes_cache.get(node.id, None)
        if cached is not None:
            return cached

        if node.is_true():
            return new_mgr.true()
        elif node.is_false():
            return new_mgr.false()
        elif node.is_literal():
            new_node = new_mgr.literal(node.literal)
            nodes_cache[node.id] = new_node
            return new_node
        else:
            or_node = new_mgr.false()
            for p, s in node.elements():
                new_p = self._copy_internal_SDD_to_aux_noref_rec(
                    new_mgr, nodes_cache, p
                )
                new_s = self._copy_internal_SDD_to_aux_noref_rec(
                    new_mgr, nodes_cache, s
                )

                conjoined = new_mgr.conjoin(new_p, new_s)
                or_node_n = new_mgr.disjoin(or_node, conjoined)
                or_node_n.deref()  # ensure refcount = 0
                conjoined.deref()  # ensure refcount = 0
                # conjoined.deref()
                # or_node.deref()
                or_node = or_node_n
            nodes_cache[node.id] = or_node
            return or_node

    def __del__(self):
        # if sdd is not None and sdd.sdd_manager_free is not None:
        #     sdd.sdd_manager_free(self.__manager)
        self.__manager = None

    def __getstate__(self):
        tempfile = mktempfile()
        vtree = self.get_manager().vtree()  # not a copy
        vtree.save(tempfile.encode())
        with open(tempfile) as f:
            vtree_data = f.read()

        nodes = []
        for n in self.nodes:
            if n is not None:
                self.get_manager().save(tempfile.encode(), n)

                with open(tempfile) as f:
                    nodes.append(f.read())
            else:
                nodes.append(None)

        self.get_manager().save(tempfile.encode(), self.constraint_dd)
        with open(tempfile) as f:
            constraint_dd = f.read()

        os.remove(tempfile)
        return {
            "varcount": self.assigned_varcount,
            "nodes": nodes,
            "vtree": vtree_data,
            "constraint_dd": constraint_dd,
        }

    def __setstate__(self, state):
        self.nodes = []
        self._assigned_varcount = state["varcount"]
        tempfile = mktempfile()
        with open(tempfile, "w") as f:
            f.write(state["vtree"])
        vtree = sdd.Vtree.from_file(tempfile)
        self.__manager = sdd.SddManager.from_vtree(vtree)

        for n in state["nodes"]:
            if n is None:
                self.nodes.append(None)
            else:
                with open(tempfile, "w") as f:
                    f.write(n)
                self.nodes.append(self.__manager.read_sdd_file(tempfile.encode()))

        with open(tempfile, "w") as f:
            f.write(state["constraint_dd"])
        self.constraint_dd = self.__manager.read_sdd_file(tempfile.encode())
        os.remove(tempfile)
        return


x_constrained = namedtuple(
    "x_constrained", "X"
)  # X = list of literalIDs that have to appear before the rest


class SDDEvaluator(DDEvaluator):
    def __init__(self, formula, semiring, weights=None, **kwargs):
        DDEvaluator.__init__(self, formula, semiring, weights, **kwargs)

    def evaluate_custom(self, node):
        # Trivial case: node is deterministically True or False
        if node == self.formula.TRUE:
            if not self.semiring.is_nsp():
                result = self.semiring.one()
            else:
                # WMC(Theory & True & Evidence) / same. The constraints are already included in the evidence node.
                result = self.semiring.normalize(
                    self._evidence_weight, self._evidence_weight
                )
        elif node is self.formula.FALSE:
            result = self.semiring.zero()
        else:
            query_def_inode = self.formula.get_inode(node)
            evidence_inode = self.evidence_inode
            query_sdd = self._get_manager().conjoin(query_def_inode, evidence_inode)

            smooth_to_root = self.semiring.is_nsp()
            # perform_smoothing=True because of indicator variables # TODO There are no indicator variables in SDDs.
            result = self._get_manager().wmc(
                query_sdd,
                weights=self.weights,
                semiring=self.semiring,
                pr_semiring=False,
                perform_smoothing=True,
                smooth_to_root=smooth_to_root,
            )

            self._get_manager().deref(query_sdd)

            # TODO only normalize when there are evidence or constraints.
            #            result = self.semiring.normalize(result, self.normalization)
            result = self.semiring.normalize(result, self._evidence_weight)
        return self.semiring.result(result, self.formula)

    def _evaluate_evidence(self, recompute=False):
        if self._evidence_weight is None or recompute:
            constraint_inode = self.formula.get_constraint_inode()
            evidence_nodes = [self.formula.get_inode(ev) for ev in self.evidence()]
            self.evidence_inode = self._get_manager().conjoin(
                constraint_inode, *evidence_nodes
            )

            pr_semiring = isinstance(
                self.semiring, (SemiringProbability, SemiringLogProbability)
            )
            result = self._get_manager().wmc(
                self.evidence_inode,
                self.weights,
                self.semiring,
                pr_semiring=pr_semiring,
                perform_smoothing=True,
                smooth_to_root=False,
            )
            if self.semiring.is_zero(result):
                raise InconsistentEvidenceError(context=" during compilation")
            if self.normalization is None:
                self._evidence_weight = result
            else:
                self._evidence_weight = self.semiring.normalize(
                    result, self.normalization
                )

        return self._evidence_weight


@transform(LogicDAG, SDD)
def build_sdd(source, destination, **kwdargs):
    """Build an SDD from another formula.

    :param source: source formula
    :type source: LogicDAG
    :param destination: destination formula
    :type destination: SDD
    :param kwdargs: extra arguments
    :return: destination
    """
    init_varcount = kwdargs.get("init_varcount", -1)
    destination.init_varcount = (
        init_varcount if init_varcount != -1 else source.atomcount
    )
    return build_dd(source, destination, **kwdargs)
