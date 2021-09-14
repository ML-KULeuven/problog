"""
problog.nnf_formula - d-DNNF
----------------------------

Provides access to d-DNNF formulae.

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
import os
import subprocess
import tempfile
from collections import defaultdict

from . import system_info
from .cnf_formula import CNF
from .constraint import ConstraintAD
from .core import transform
from .errors import CompilationError
from .errors import InconsistentEvidenceError
from .evaluator import Evaluator, EvaluatableDSP
from .formula import LogicDAG
from .util import Timer, subprocess_check_call


class DSharpError(CompilationError):
    """DSharp has crashed."""

    def __init__(self):
        msg = "DSharp has encountered an error"
        if system_info["os"] == "darwin":
            msg += ". This is a known issue. See KNOWN_ISSUES for details on how to resolve this problem"
        CompilationError.__init__(self, msg)


class DDNNF(LogicDAG, EvaluatableDSP):
    """A d-DNNF formula."""

    transform_preference = 20

    # noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
    def __init__(self, **kwdargs):
        LogicDAG.__init__(self, auto_compact=False)

    def _create_evaluator(self, semiring, weights, **kwargs):
        return SimpleDDNNFEvaluator(self, semiring, weights)


class SimpleDDNNFEvaluator(Evaluator):
    """Evaluator for d-DNNFs."""

    def __init__(self, formula, semiring, weights=None, **kwargs):
        Evaluator.__init__(self, formula, semiring, weights, **kwargs)
        self.cache_intermediate = {}  # weights of intermediate nodes

    def _initialize(self, with_evidence=True):
        self.weights.clear()

        model_weights = self.formula.extract_weights(self.semiring, self.given_weights)
        self.weights = model_weights.copy()

        if with_evidence:
            for ev in self.evidence():
                self.set_evidence(abs(ev), ev > 0)

        if self.semiring.is_zero(self._get_z()):
            raise InconsistentEvidenceError(context=" during evidence evaluation")

    def propagate(self):
        self._initialize()

    def _get_z(self):
        result = self.get_root_weight()
        return result

    def evaluate_evidence(self, recompute=False):
        return self.semiring.result(
            self._evaluate_evidence(recompute=recompute), self.formula
        )

    # noinspection PyUnusedLocal
    def _evaluate_evidence(self, recompute=False):
        self._initialize(False)
        for ev in self.evidence():
            self._set_value(abs(ev), ev > 0)

        result = self.get_root_weight()
        return result

    def evaluate_fact(self, node):
        return self.evaluate(node)

    def evaluate(self, node):
        if node == 0:
            # if query = True
            if not self.semiring.is_nsp():
                result = self.semiring.one()
            else:
                result = self.get_root_weight()
                result = self.semiring.normalize(result, self._get_z())
        elif node is None:
            result = self.semiring.zero()
        else:
            p = self._get_weight(abs(node))
            n = self._get_weight(-abs(node))
            self._set_value(abs(node), (node > 0))
            result = self.get_root_weight()
            self._reset_value(abs(node), p, n)
            if self.has_evidence() or self.semiring.is_nsp() or self.has_constraints(ignore_type={ConstraintAD}):
                result = self.semiring.normalize(result, self._get_z())
        return self.semiring.result(result, self.formula)

    def has_constraints(self, ignore_type=None):
        """
        Check whether the formula has any constraints that are not of the ignore_type.
        :param ignore_type: A set of constraint classes to ignore.
        :type ignore_type: None | Set
        """
        ignore_type = ignore_type or set()
        return any(type(constraint) not in ignore_type for constraint in self.formula.constraints())

    def _reset_value(self, index, pos, neg):
        self.set_weight(index, pos, neg)

    def get_root_weight(self):
        """Get the WMC of the root of this formula.

        :return: The WMC of the root of this formula (WMC of node len(self.formula)), multiplied with weight of True
        (self.weights.get(0)).
        """
        result = self._get_weight(len(self.formula))
        return (
            self.semiring.times(result, self.weights.get(0)[0])
            if self.weights.get(0) is not None
            else result
        )

    def _get_weight(self, index):
        if index == 0:
            return self.semiring.one()
        elif index is None:
            return self.semiring.zero()
        else:
            abs_index = abs(index)
            w = self.weights.get(abs_index) or self.cache_intermediate.get(abs_index)
            if w is not None:
                return w[index < 0]
            else:
                w = self._calculate_weight(index)
                self.cache_intermediate[abs_index] = w, w
                return w

    def set_weight(self, index, pos, neg):
        # index = index of atom in weights, so atom2var[key] = index
        self.weights[index] = (pos, neg)
        self.cache_intermediate.clear()

    def set_evidence(self, index, value):
        curr_pos_weight, curr_neg_weight = self.weights.get(index)
        pos, neg = self.semiring.to_evidence(
            curr_pos_weight, curr_neg_weight, sign=value
        )

        if (value and self.semiring.is_zero(curr_pos_weight)) or (
            not value and self.semiring.is_zero(curr_neg_weight)
        ):
            raise InconsistentEvidenceError(self._deref_node(index))

        self.set_weight(index, pos, neg)

    def _deref_node(self, index):
        return self.formula.get_node(index).name

    def _set_value(self, index, value):
        """Set value for given node.

        :param index: index of node
        :param value: value
        """
        if value:
            pos = self._get_weight(index)
            self.set_weight(index, pos, self.semiring.zero())
        else:
            neg = self._get_weight(-index)
            self.set_weight(index, self.semiring.zero(), neg)

    def _calculate_weight(self, key):
        assert key != 0
        assert key is not None
        # assert(key > 0)

        node = self.formula.get_node(abs(key))
        ntype = type(node).__name__

        if ntype == "atom":
            return self.semiring.one()
        else:
            assert key > 0
            childprobs = [self._get_weight(c) for c in node.children]
            if ntype == "conj":
                p = self.semiring.one()
                for c in childprobs:
                    p = self.semiring.times(p, c)
                return p
            elif ntype == "disj":
                p = self.semiring.zero()
                for c in childprobs:
                    p = self.semiring.plus(p, c)
                return p
            else:
                raise TypeError("Unexpected node type: '%s'." % ntype)


class Compiler(object):
    """Interface to CNF to d-DNNF compiler tool."""

    __compilers = {}

    @classmethod
    def get_default(cls):
        """Get default compiler for this system."""
        if system_info.get("c2d", False):
            return _compile_with_c2d
        else:
            return _compile_with_dsharp

    @classmethod
    def get(cls, name):
        """Get compiler by name (or default if name not found).

        :param name: name of the compiler
        :returns: function used to call compiler
        """
        result = cls.__compilers.get(name)
        if result is None:
            result = cls.get_default()
        return result

    @classmethod
    def add(cls, name, func):
        """Add a compiler.

        :param name: name of the compiler
        :param func: function used to call the compiler
        """
        cls.__compilers[name] = func


if system_info.get("c2d", False):
    # noinspection PyUnusedLocal
    @transform(CNF, DDNNF)
    def _compile_with_c2d(cnf, nnf=None, smooth=True, **kwdargs):
        fd, cnf_file = tempfile.mkstemp(".cnf")
        os.close(fd)
        nnf_file = cnf_file + ".nnf"
        if smooth:
            smoothl = ["-smooth_all"]
        else:
            smoothl = []

        cmd = ["cnf2dDNNF", "-dt_method", "0"] + smoothl + ["-reduce", "-in", cnf_file]

        try:
            os.remove(cnf_file)
        except OSError:
            pass
        try:
            os.remove(nnf_file)
        except OSError:
            pass

        return _compile(cnf, cmd, cnf_file, nnf_file)

    Compiler.add("c2d", _compile_with_c2d)


# noinspection PyUnusedLocal
@transform(CNF, DDNNF)
def _compile_with_dsharp(cnf, nnf=None, smooth=True, **kwdargs):
    result = None
    with Timer("DSharp compilation"):
        fd1, cnf_file = tempfile.mkstemp(".cnf")
        fd2, nnf_file = tempfile.mkstemp(".nnf")
        os.close(fd1)
        os.close(fd2)
        if smooth:
            smoothl = ["-smoothNNF"]
        else:
            smoothl = []
        cmd = ["dsharp", "-Fnnf", nnf_file] + smoothl + ["-disableAllLits", cnf_file]  #

        try:
            result = _compile(cnf, cmd, cnf_file, nnf_file)
        except subprocess.CalledProcessError:
            raise DSharpError()

        try:
            os.remove(cnf_file)
        except OSError:
            pass
        try:
            os.remove(nnf_file)
        except OSError:
            pass

    return result


Compiler.add("dsharp", _compile_with_dsharp)


def _compile(cnf, cmd, cnf_file, nnf_file):
    names = cnf.get_names_with_label()

    if cnf.is_trivial():
        nnf = DDNNF()
        weights = cnf.get_weights()
        for i in range(1, cnf.atomcount + 1):
            nnf.add_atom(i, weights.get(i))
        or_nodes = []
        for i in range(1, cnf.atomcount + 1):
            or_nodes.append(nnf.add_or((i, -i)))
        if or_nodes:
            nnf.add_and(or_nodes)

        for name, node, label in names:
            nnf.add_name(name, node, label)
        for c in cnf.constraints():
            nnf.add_constraint(c.copy())

        return nnf
    else:
        with open(cnf_file, "w") as f:
            f.write(cnf.to_dimacs())

        attempts_left = 1
        success = False
        while attempts_left and not success:
            try:
                with open(os.devnull, "w") as OUT_NULL:
                    subprocess_check_call(cmd, stdout=OUT_NULL)
                success = True
            except subprocess.CalledProcessError as err:
                attempts_left -= 1
                if attempts_left == 0:
                    raise err
        return _load_nnf(nnf_file, cnf)


def _load_nnf(filename, cnf):
    nnf = DDNNF()

    weights = cnf.get_weights()

    names_inv = defaultdict(list)
    for name, node, label in cnf.get_names_with_label():
        names_inv[node].append((name, label))

    with open(filename) as f:
        line2node = {}
        rename = {}
        lnum = 0
        for line in f:
            line = line.strip().split()
            if line[0] == "nnf":
                pass
            elif line[0] == "L":
                name = int(line[1])
                prob = weights.get(abs(name), True)
                node = nnf.add_atom(abs(name), prob)
                rename[abs(name)] = node
                if name < 0:
                    node = -node
                line2node[lnum] = node
                if name in names_inv:
                    for actual_name, label in names_inv[name]:
                        nnf.add_name(actual_name, node, label)
                    del names_inv[name]
                lnum += 1
            elif line[0] == "A":
                children = map(lambda x: line2node[int(x)], line[2:])
                line2node[lnum] = nnf.add_and(children)
                lnum += 1
            elif line[0] == "O":
                children = map(lambda x: line2node[int(x)], line[3:])
                line2node[lnum] = nnf.add_or(children)
                lnum += 1
            else:
                print("Unknown line type")
        for name in names_inv:
            for actual_name, label in names_inv[name]:
                if name == 0:
                    nnf.add_name(actual_name, 0, label)
                else:
                    nnf.add_name(actual_name, None, label)
    for c in cnf.constraints():
        nnf.add_constraint(c.copy(rename))

    return nnf
