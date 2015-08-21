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
from __future__ import print_function

import tempfile, os, sys, subprocess
from collections import defaultdict

from . import system_info
from .evaluator import Evaluator, Evaluatable, InconsistentEvidenceError
from .formula import LogicDAG
from .logic import LogicProgram
from .cnf_formula import CNF
from .core import transform, CompilationError
from .util import Timer, subprocess_check_call

class DSharpError(CompilationError) :

    def __init__(self) :
        CompilationError.__init__(self, 'DSharp has encountered an error. See INSTALL for instructions on how to install an alternative knowledge compiler.')

class NNF(LogicDAG, Evaluatable) :

    def __init__(self, **kwdargs):
        LogicDAG.__init__(self, auto_compact=False)

    def create_evaluator(self, semiring, weights):
        return SimpleNNFEvaluator(self, semiring, weights)


class SimpleNNFEvaluator(Evaluator) :

    def __init__(self, formula, semiring, weights=None) :
        Evaluator.__init__(self, formula, semiring)
        self.__nnf = formula
        self.__probs = {}
        self.__given_weights = weights

    def get_names(self, label=None) :
        return self.__nnf.get_names(label)

    def initialize(self, with_evidence=True) :
        self.__probs.clear()

        model_weights = self.__nnf.extract_weights(self.semiring, self.__given_weights)
        for n, p in model_weights.items() :
            self.__probs[n] = p[0]
            self.__probs[-n] = p[1]

        if with_evidence :
            for ev in self.evidence() :
                self.set_evidence( abs(ev), ev > 0 )

        if self.semiring.result(self.get_z()) == 0.0 :
            raise InconsistentEvidenceError()


    def propagate(self) :
        self.initialize()

    def get_z(self) :
        result = self.get_weight( len(self.__nnf) )
        return result

    def evaluate_evidence(self):
        self.initialize(False)
        for ev in self.evidence():
            self.set_value(abs(ev), ev > 0)

        result = self.get_weight(len(self.__nnf))
        return result

    def evaluate_fact(self, node):
        return self.evaluate(node)

    def evaluate(self, node):
        if node == 0:
            result = self.semiring.one()
        elif node is None:
            result = self.semiring.zero()
        else:
            p = self.get_weight(abs(node))
            n = self.get_weight(-abs(node))
            self.set_value(abs(node), (node > 0))
            result = self.get_weight(len(self.__nnf))
            self.reset_value(abs(node), p, n)
            if self.has_evidence():
                result = self.semiring.normalize(result, self.get_z())
        return self.semiring.result(result)

    def reset_value(self, index, pos, neg):
        self.set_weight(index, pos, neg)

    def get_weight(self, index) :
        if index == 0 :
            return self.semiring.one()
        elif index is None :
            return self.semiring.zero()
        else :
            w = self.__probs.get(index)
            if w is None :
                w = self._calculateWeight(index)
                return w
            else :
                return w

    def set_weight(self, index, pos, neg) :
        self.__probs[index] = pos
        self.__probs[-index] = neg

    def set_evidence(self, index, value ) :
        pos = self.semiring.one()
        neg = self.semiring.zero()
        if value :
            self.set_weight(index, pos, neg)
        else :
            self.set_weight(index, neg, pos)

    def set_value(self, index, value ) :
        if value :
            pos = self.get_weight(index)
            self.set_weight( index, pos, self.semiring.zero() )
        else :
            neg = self.get_weight(-index)
            self.set_weight( index, self.semiring.zero(), neg )

    def _calculateWeight(self, key):
        assert(key != 0)
        assert(key is not None)
        # assert(key > 0)

        node = self.__nnf.get_node(abs(key))
        ntype = type(node).__name__

        if ntype == 'atom':
            return self.semiring.one()
        else:
            assert key > 0
            childprobs = [self.get_weight(c) for c in node.children]
            if ntype == 'conj':
                p = self.semiring.one()
                for c in childprobs:
                    p = self.semiring.times(p, c)
                return p
            elif ntype == 'disj':
                p = self.semiring.zero()
                for c in childprobs:
                    p = self.semiring.plus(p, c)
                return p
            else :
                raise TypeError("Unexpected node type: '%s'." % ntype)


class Compiler(object) :

    __compilers = {}

    @classmethod
    def getDefault(cls) :
        if system_info.get('c2d', False) :
            return _compile_with_c2d
        else :
            return _compile_with_dsharp

    @classmethod
    def get(cls, name) :
        result = cls.__compilers.get(name)
        if result is None : result = cls.getDefault()
        return result

    @classmethod
    def add(cls, name, func) :
        cls.__compilers[name] = func


if system_info.get('c2d', False) :
    @transform(CNF, NNF)
    def _compile_with_c2d( cnf, nnf=None, **kwdargs) :
        fd, cnf_file = tempfile.mkstemp('.cnf')
        os.close(fd)
        nnf_file = cnf_file + '.nnf'
        cmd = ['cnf2dDNNF', '-dt_method', '0', '-smooth_all', '-reduce', '-in', cnf_file ]

        try :
            os.remove(cnf_file)
        except OSError :
            pass
        try :
            os.remove(nnf_file)
        except OSError :
            pass

        return _compile( cnf, cmd, cnf_file, nnf_file )
    Compiler.add( 'c2d', _compile_with_c2d )


@transform(CNF, NNF)
def _compile_with_dsharp( cnf, nnf=None, **kwdargs) :
    with Timer('DSharp compilation'):
        cnf_file = tempfile.mkstemp('.cnf')[1]
        nnf_file = tempfile.mkstemp('.nnf')[1]
        cmd = ['dsharp', '-Fnnf', nnf_file, '-smoothNNF','-disableAllLits', cnf_file ] #

        try :
            result = _compile( cnf, cmd, cnf_file, nnf_file )
        except subprocess.CalledProcessError :
            raise DSharpError()

        try :
            os.remove(cnf_file)
        except OSError :
            pass
        try :
            os.remove(nnf_file)
        except OSError :
            pass

    return result
Compiler.add( 'dsharp', _compile_with_dsharp )


def _compile(cnf, cmd, cnf_file, nnf_file) :
    names = cnf.get_names_with_label()

    if cnf.is_trivial() :
        nnf = NNF()
        weights = cnf.get_weights()
        for i in range(1,cnf.atomcount+1) :
            nnf.add_atom( i, weights.get(i))
        or_nodes = []
        for i in range(1,cnf.atomcount+1) :
            or_nodes.append( nnf.add_or( (i, -i) ) )
        if or_nodes :
            nnf.add_and( or_nodes )

        for name, node, label in names:
            nnf.add_name(name, node, label)
        for c in cnf.constraints() :
            nnf.add_constraint(c.copy())

        return nnf
    else :
        with open(cnf_file, 'w') as f :
            f.write(cnf.to_dimacs())

        attempts_left = 1
        success = False
        while attempts_left and not success :
            try :
                with open(os.devnull, 'w') as OUT_NULL :
                    subprocess_check_call(cmd, stdout=OUT_NULL)
                success = True
            except subprocess.CalledProcessError as err :
                attempts_left -= 1
                if attempts_left == 0 :
                    raise err
        return _load_nnf( nnf_file, cnf)


def _load_nnf(filename, cnf) :

    nnf = NNF()

    weights = cnf.get_weights()

    names_inv = defaultdict(list)
    for name,node,label in cnf.get_names_with_label():
        names_inv[node].append((name,label))

    with open(filename) as f :
        line2node = {}
        rename = {}
        lnum = 0
        for line in f :
            line = line.strip().split()
            if line[0] == 'nnf' :
                pass
            elif line[0] == 'L' :
                name = int(line[1])
                prob = weights.get(abs(name), True)
                node = nnf.add_atom( abs(name), prob )
                rename[abs(name)] = node
                if name < 0 : node = -node
                line2node[lnum] = node
                if name in names_inv :
                    for actual_name, label in names_inv[name] :
                        nnf.add_name(actual_name, node, label)
                    del names_inv[name]
                lnum += 1
            elif line[0] == 'A' :
                children = map(lambda x : line2node[int(x)] , line[2:])
                line2node[lnum] = nnf.add_and( children )
                lnum += 1
            elif line[0] == 'O' :
                children = map(lambda x : line2node[int(x)], line[3:])
                line2node[lnum] = nnf.add_or( children )
                lnum += 1
            else :
                print ('Unknown line type')
        for name in names_inv :
            for actual_name, label in names_inv[name] :
                nnf.add_name(actual_name, None, label)
    for c in cnf.constraints() :
        nnf.add_constraint(c.copy(rename))

    return nnf
