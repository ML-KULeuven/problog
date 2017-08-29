"""
Conditional Probability Distributions.

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

import itertools
from datetime import datetime
import re
from functools import reduce
import sys
import math
from collections import Counter, defaultdict, OrderedDict
import logging


logger = logging.getLogger('be.kuleuven.cs.dtai.problog.cpd')


class PGM(object):
    __count = 0

    def __init__(self, directed=True, variables=None, factors=None, name=None):
        """Probabilistic Graphical Model.

        Simple Probabilistic Graphical Model (PGM) representation to
        - translate between PGMs and ProbLog
        - translate between different PGM formats
        - perform a number of PGM transformations (split, compress, ...)
        """
        PGM.__count += 1
        if name:
            self.name = name
        else:
            self.name = 'PGM {}'.format(PGM.__count)
            PGM.__count += 1
        self.comments = []
        self.directed = directed
        self.factors = OrderedDict()
        self.vars = OrderedDict()
        if variables:
            for var in variables:
                self.add_var(var)
        if factors:
            for factor in factors:
                self.add_factor(factor)

    def copy(self, **kwargs):
        variables = kwargs.get('variables', None)
        if variables is None:
            variables = list(self.vars.values())
        factors = kwargs.get('factors', None)
        if factors is None:
            factors = list(self.factors.values())
        return PGM(
            directed=kwargs.get('directed', self.directed),
            factors=factors,
            variables=variables
        )

    def split_topological(self, split_vars):
        """Split the PGM based on the given set of variables.

        The graph will be split in two graphs where all nodes in graph 1 are an ancestor
        of a variable in the given set of split variables.

        TODO: We split on all parents. Would be sufficient to split on parents with lower
              index. But this requires a marginalize operator

        :param split_vars: Set of variable names.
        """
        if not self.directed:
            return self

        children = dict()
        for factor in self.factors.values():
            if factor.rv is not None and factor.rv not in children:
                children[factor.rv] = set()
            for parent in factor.parents:
                if parent in children:
                    children[parent].add(factor.rv)
                else:
                    children[parent] = {factor.rv}

        all_links = set(children.keys())
        nonroot = reduce(set.union, children.values())
        root = all_links - nonroot

        first_roots = set()
        queue = list(split_vars)
        while len(queue) > 0:
            qvar = queue.pop()
            if len(self.factors[qvar].parents) > 0:
                queue.extend(self.factors[qvar].parents)
            else:
                first_roots.add(qvar)
        delay_roots= root - first_roots

        factor_strata = self.factors_topological(keep_strata=True, delay_roots=[delay_roots])
        # for fs_i, fs in enumerate(factor_strata):
        #     print('{}: {}'.format(fs_i, ', '.join([f.rv for f in fs])))
        strata = dict()
        for strata_i, factors in enumerate(factor_strata):
            for factor in factors:
                strata[factor.rv] = strata_i
        max_strata = max((strata[svar] for svar in split_vars))

        def rootvar(var):
            return var+"_root"
        pgm = self.copy(factors=[])
        for factor in self.factors.values():
            if strata[factor.rv] <= max_strata and any((strata[cvar] > max_strata for cvar in children[factor.rv])):
                nb_values = len(self.vars[factor.rv].values)
                pgm.add_var(self.vars[factor.rv].copy(name=rootvar(factor.rv)))
                pgm.add_factor(Factor(pgm, rootvar(factor.rv), [], [1.0 / nb_values] * nb_values))
            if strata[factor.rv] > max_strata and any((strata[pvar] <= max_strata for pvar in factor.parents)):
                parents = []
                for pvar in factor.parents:
                    if strata[pvar] <= max_strata:
                        parents.append(rootvar(pvar))
                    else:
                        parents.append(pvar)
                pgm.add_factor(factor.copy(pgm=pgm, parents=parents))
            else:
                pgm.add_factor(factor.copy(pgm=pgm))
        return pgm

    def split(self, variables):
        """Split the PGM based on the given set of variables.

        The given nodes will be duplicated in a node with no outgoing edge
        (thus a sink node) and a node with no incoming edges (thus a root
        node).

        :param variables: Set of variable names.
        """
        def splitvar(var):
            return var+"_sink"
        new_vars = [self.vars[var].copy(name=splitvar(var)) for var in variables]
        pgm = self.copy(factors=[], variables=list(self.vars.values()) + new_vars)
        for factor in self.factors.values():
            if factor.rv in variables:
                nb_values = len(self.vars[factor.rv].values)
                pgm.add_factor(Factor(pgm, factor.rv, [], [1.0/nb_values]*nb_values))
                pgm.add_factor(factor.copy(pgm=pgm, rv=splitvar(factor.rv)))
            else:
                pgm.add_factor(factor.copy(pgm=pgm))
        return pgm

    def to_connected_parts(self):
        """Return a set of PGMs that are disconnected."""
        children = dict()
        for factor in self.factors.values():
            if factor.rv is not None and factor.rv not in children:
                children[factor.rv] = set()
            for parent in factor.parents:
                if parent in children:
                    children[parent].add(factor.rv)
                else:
                    children[parent] = {factor.rv}
        parts = dict()
        nb_parts = 0
        vars = list(self.vars.values())
        queue = [(v.name, None) for v in vars[1:]]
        queue.append((vars[0].name, 0))
        while len(queue) > 0:
            var, cur_part = queue.pop()
            if var in parts:
                continue
            if cur_part is None:
                nb_parts += 1
                cur_part = nb_parts
            parts[var] = cur_part
            logger.debug('PGM[{}] = {}'.format(var, cur_part))
            for cvar in children[var]:
                logger.debug('{} -> {}'.format(var, cvar))
                queue.append((cvar, cur_part))
            for pvar in self.factors[var].parents:
                logger.debug('{} -> {}'.format(pvar, var))
                queue.append((pvar, cur_part))
        nb_parts += 1
        pgms = [self.copy(factors=[], variables=[]) for i in range(nb_parts)]
        logger.debug('Creating {} PGMs'.format(len(pgms)))
        for var, part in parts.items():
            factor = self.factors[var]
            if factor.rv:
                pgms[part].add_var(self.vars[factor.rv].copy())
            for pvar in factor.parents:
                pgms[part].add_var(self.vars[pvar].copy())
            pgms[part].add_factor(factor.copy())
        return pgms

    def add_var(self, var):
        self.vars[var.name] = var

    def add_factor(self, factor):
        """Add a CPD.
        :param factor: New factor to add to PGM
        """
        if self.directed:
            name = factor.rv
        else:
            name = factor.name
        assert (name is not None)
        factor.pgm = self
        if name in self.factors:
            self.factors[name] += factor
        else:
            self.factors[name] = factor

    def factors_topological(self, keep_strata=False, delay_roots=None):
        """Return the factors in a topological order."""
        # Links from parent-node to child-node
        if delay_roots is None:
            delay_roots = []
        if not self.directed:
            return self.factors.values()
        links = dict()
        for factor in self.factors.values():
            if factor.rv not in links:
                links[factor.rv] = set()
            for parent in factor.parents:
                if parent in links:
                    links[parent].add(factor.rv)
                else:
                    links[parent] = {factor.rv}
        all_links = set(links.keys())
        nonroot = reduce(set.union, links.values())
        root = all_links - nonroot
        if delay_roots:
            queue = root - reduce(set.union, delay_roots)
        else:
            queue = root
        visited = set()
        factors = []
        while len(queue) > 0 or len(delay_roots) > 0:
            if len(queue) == 0:
                queue = delay_roots.pop()
            strata = []
            visited_new = set()
            for rv in queue:
                if rv in visited:
                    continue
                all_parents_visited = True
                try:
                    factor = self.factors[rv]
                except KeyError as exc:
                    print('Error: random variable has no CPD associated:')
                    print(exc)
                    sys.exit(1)
                for parent_rv in factor.parents:
                    if parent_rv not in visited:
                        all_parents_visited = False
                        break
                if all_parents_visited:
                    strata.append(factor)
                    visited_new.add(rv)
            visited.update(visited_new)
            if keep_strata:
                if len(strata) > 0:
                    factors.append(strata)
            else:
                factors.extend(strata)
            queue2 = set()
            for rv in queue:
                if rv in links:
                    for next_rv in links[rv]:
                        queue2.add(next_rv)
            queue = queue2
        return factors

    def compress_tables(self, allow_disjunct=False):
        """Analyze CPTs and join rows that have the same probability
        distribution."""
        factors = []
        for idx, factor in enumerate(self.factors.values()):
            factors.append(factor.compress(allow_disjunct=allow_disjunct))
        return self.copy(factors=factors)

    def to_hugin_net(self, include_layout=False):
        """Export PGM to the Hugin net format.
        http://www.hugin.com/technology/documentation/api-manuals
        """
        assert self.directed
        cpds = [cpd.to_factor() for cpd in self.factors_topological()]
        lines = ["%% Hugin Net: {}\n".format(self.name),
                 "%% Created on {}\n".format(datetime.now())]
        if include_layout:
            lines += ["net {",
                      "  node_size = (50,50);",
                      "}\n"]
        lines += ["%% Nodes\n"]
        lines += [cpd.to_huginnet_node(include_layout=include_layout) for cpd in cpds]
        lines += ["%% Potentials\n"]
        lines += [cpd.to_huginnet_potential(include_layout=include_layout) for cpd in cpds]
        return '\n'.join(lines)

    def to_xdsl(self):
        """Export PGM to the XDSL format defined by SMILE.
        https://dslpitt.org/genie/wiki/Appendices:_XDSL_File_Format_-_XML_Schema_Definitions
        http://support.bayesfusion.com/docs/
        """
        assert self.directed
        cpds = [cpd.to_factor() for cpd in self.factors_topological()]
        lines = ['<?xml version="1.0" encoding="ISO-8859-1" ?>',
                 '<smile version="1.0" id="Aa" numsamples="1000">',
                 '  <nodes>']
        lines += [cpd.to_xdsl_cpt() for cpd in cpds]
        lines += ['  </nodes>',
                  '  <extensions>',
                  '    <genie version="1.0" app="ProbLog" name="{}" faultnameformat="nodestate">'.format(self.name)]
        lines += [cpd.to_xdsl_node() for cpd in cpds]
        lines += ['    </genie>',
                  '  </extensions>',
                  '</smile>']
        return '\n'.join(lines)

    def to_xmlbif(self):
        """Export PGM to the XMLBIF format defined in
        http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
        """
        assert self.directed
        cpds = [cpd.to_factor() for cpd in self.factors_topological()]
        lines = ['<?xml version="1.0" encoding="US-ASCII"?>',
                 '<BIF VERSION="0.3">',
                 '<NETWORK>']
        lines += [cpd.to_xmlbif_node() for cpd in cpds]
        lines += [cpd.to_xmlbif_cpt() for cpd in cpds]
        lines += ['</NETWORK>',
                  '</BIF>']
        return '\n'.join(lines)

    def to_uai08(self):
        """Export PGM to the format used in the UAI 2008 competition.
        http://graphmod.ics.uci.edu/uai08/FileFormat
        """
        assert self.directed
        # do not sort topological such that reading in and writing out UAI format is identical (thus ordered by
        # variable name instead of topological)
        cpds = [cpd.to_factor() for cpd in self.factors.values()]
        number_variables = str(len(cpds))
        domain_sizes = [str(len(self.vars[cpd.rv].values)) for cpd in cpds]
        number_functions = str(len(cpds))
        lines = ['BAYES',
                 number_variables,
                 ' '.join(domain_sizes),
                 number_functions]
        lines += [cpd.to_uai08_preamble(cpds) for cpd in cpds]
        lines += ['']
        lines += [cpd.to_uai08_function() for cpd in cpds]
        return '\n'.join(lines)

    def to_problog(self, drop_zero=False, use_neglit=False, value_as_term=True, ad_is_function=False):
        """Export PGM to ProbLog.
        :param ad_is_function: Experimental
        :param value_as_term: Include the variable's value as the last term instead of as part of the predicate name
        :param use_neglit: Use negative literals if it simplifies the program
        :param drop_zero: Do not include head literals with probability zero
        """
        factors = [factor.to_factor() for factor in self.factors_topological()]
        lines = ["%% ProbLog program: {}".format(self.name),
                 "%% Created on {}".format(datetime.now())]
        if len(self.comments) > 0:
            lines += ["%% {}".format(comment) for comment in self.comments]
        lines += [""]
        if self.directed:
            lines += [factor.to_problog(self, drop_zero=drop_zero,
                                        use_neglit=use_neglit,
                                        value_as_term=value_as_term,
                                        ad_is_function=ad_is_function) for factor in factors]
        else:
            lines += [factor.to_problog_undirected(self, drop_zero=drop_zero,
                                                   use_neglit=use_neglit,
                                                   value_as_term=value_as_term,
                                                   ad_is_function=ad_is_function) for factor in factors]
            # if ad_is_function:
            # lines += ["evidence(false_constraints,false)."]
        return '\n'.join(lines)

    def to_graphviz(self):
        """Export PGM to Graphviz dot format.
        http://www.graphviz.org
        """
        assert self.directed
        lines = [
            '// Graphiz DOT: {}'.format(self.name),
            "// Created on {}\n".format(datetime.now()),
            'digraph pgm {'
        ]
        for cpd in self.factors.values():
            lines.append('  {} [label="{}"];'.format(self.vars[cpd.rv].clean(), cpd.rv))
            for p in cpd.parents:
                lines.append('  {} -> {}'.format(self.vars[cpd.rv].clean(p), self.vars[cpd.rv].clean()))
        lines += ['}']
        return '\n'.join(lines)

    def __str__(self):
        return '\n'.join([str(factor) for factor in self.factors.values()])


re_tounderscore = re.compile(r'[\(\),\]\[ ]')
re_toremove = re.compile(r"""[^a-zA-Z0-9_]""")

boolean_values = [
    ['0', '1'],
    [0, 1],
    ['f', 't'],
    ['false', 'true'],
    ['no', 'yes'],
    ['n', 'y'],
    ['neg', 'pos'],
    ['nay', 'aye'],
    [False, True]
]


class Variable(object):
    def __init__(self, name, values, detect_boolean=True, force_boolean=False, boolean_true=None,
                 latent=False):
        """Conditional Probability Distribution."""
        self.name = name
        self.values = values
        self.latent = latent
        self.boolean_true = boolean_true  # Value that represents true
        if (force_boolean or detect_boolean) and len(self.values) == 2:
            for values in boolean_values:
                if (values[0] == self.values[0] and values[1] == self.values[1]) or \
                   (type(self.values[0]) == str and type(self.values[1]) == str and
                   values[0] == self.values[0].lower() and values[1] == self.values[1].lower()):
                    self.boolean_true = 1
                    break
                elif (values[1] == self.values[0] and values[0] == self.values[1]) or \
                    (type(self.values[0]) == str and type(self.values[1]) == str and
                        values[1] == self.values[0].lower() and values[0] == self.values[1].lower()):
                    self.boolean_true = 0
                    break
            if force_boolean and self.boolean_true is None:
                self.boolean_true = 1

    def copy(self, **kwargs):
        return Variable(
            name=kwargs.get('name', self.name),
            values=kwargs.get('values', self.values),
            latent=kwargs.get('latent', self.latent),
            detect_boolean=False,
            force_boolean=False,
            boolean_true=kwargs.get('boolean_true', self.boolean_true),
        )

    def clean(self, value=None):
        if value is None:
            rv = self.name
        else:
            rv = value
        rv = re_tounderscore.sub('_', rv)
        rv = re_toremove.sub('', rv)
        if not rv[0].islower():
            rv = rv[0].lower() + rv[1:]
            if not rv[0].islower():
                rv = "v" + rv
        return rv

    def to_problog_value(self, value, value_as_term=True):
        if self.boolean_true is None:
            if type(value) is frozenset and len(value) == 1:
                value, = value
            if type(value) is frozenset:
                # It is a disjunction of possible values
                if len(value) == len(self.values) - 1:
                    # This is the negation of one value
                    new_value, = frozenset(self.values) - value
                    if value_as_term:
                        return '\+' + self.clean() + '("' + str(new_value) + '")'
                    else:
                        return '\+' + self.clean() + '_' + self.clean(str(new_value))
                else:
                    if value_as_term:
                        return '(' + '; '.join(
                            [self.clean() + '("' + str(new_value) + '")' for new_value in value]) + ')'
                    else:
                        return '(' + \
                               '; '.join([self.clean() + '_' + self.clean(str(new_value)) for new_value in value]) + ')'
            else:
                if value_as_term:
                    return self.clean() + '("' + str(value) + '")'
                else:
                    return self.clean() + '_' + self.clean(str(value))
        else:
            # It is a Boolean atom
            if self.values[self.boolean_true] == value:
                return self.clean()
            elif self.values[1 - self.boolean_true] == value:
                return '\+' + self.clean()
            else:
                raise Exception('Unknown value: {} = {}'.format(self.name, value))

    def __str__(self):
        return '{}'.format(self.name)


class Factor(object):
    def __init__(self, pgm, rv, parents, table, name=None, *args, **kwargs):
        """Conditional Probability Table with discrete probabilities.

        :param rv: random variable. None for an undirected factor.
        :param parents: Parent random variables.
        :param table:
        :param name: Name, for undirected factor.

        a = Factor(pgm, 'a', [], [0.4,0.6])
        b = Factor(pgm, 'b', [a],
                   {('f',): [0.2,0.8],
                    ('t',): [0.7,0.3]})
        """
        self.rv = rv
        self.pgm = pgm
        self.name = name
        self.parents = parents
        if isinstance(table, list) or isinstance(table, tuple):
            self.table = {(): table}
        elif isinstance(table, dict):
            self.table = table
        else:
            raise ValueError('Unknown type (expected list, tuple or dict): {}'.format(type(table)))

    def to_factor(self):
        return self

    def copy(self, **kwargs):
        return Factor(
            rv=kwargs.get('rv', self.rv),
            pgm=kwargs.get('pgm', self.pgm),
            name=kwargs.get('name', self.name),
            parents=kwargs.get('parents', self.parents),
            table=kwargs.get('table', self.table)
        )

    def compress(self, allow_disjunct=False):
        """Table to tree using the ID3 decision tree algorithm.

        From:
            {('f', 'f'): (0.2, 0.8),
             ('f', 't'): (0.2, 0.8),
             ('t', 'f'): (0.1, 0.9),
             ('t', 't'): (0.6, 0.4)}
        To:
            {('f', None): (0.2, 0.8),
             ('t', 'f'):  (0.1, 0.9),
             ('t', 't'):  (0.6, 0.4)}
        """
        # Traverse through the tree
        # First tuple is path with no value assignment and all rows in the table
        table = []
        for k, v in self.table.items():
            # Tuples allow hashing for IG
            table.append((k, tuple(v)))
        nodes = [(tuple([None] * len(self.parents)), table)]
        new_table = {}
        cnt = 0
        while len(nodes) > 0:
            cnt += 1
            curpath, node = nodes.pop()
            if len(node) == 0:
                continue
            # All the same or all different? Then stop.
            k, v = zip(*node)
            c = Counter(v)
            if len(c) == len(v):
                for new_path, new_probs in node:
                    new_table[new_path] = new_probs
                continue
            if len(c) == 1:
                new_table[curpath] = node[0][1]
                continue
            # Find max information gain
            # ig_idx = self.maxinformationgainparent(node)
            ps = [cnt / len(v) for cnt in c.values()]  # Pr(x_i)
            h_cur = -sum([p * math.log(p, 2) for p in ps])  # Entropy
            igr_idx = None
            igr_max = -9999999
            for parent_idx, parent in enumerate(self.parents):
                # print('parent_idx:{}, parent:{}'.format(parent_idx, parent))
                if curpath[parent_idx] is not None:
                    continue
                bins = {}
                for value in self.pgm.vars[parent].values:
                    bins[value] = []
                for k, v in node:
                    bins[k[parent_idx]].append(v)
                ig = h_cur  # Information Gain
                iv = 0  # Intrinsic Value
                for bin_value, bin_labels in bins.items():
                    label_cnt = Counter(bin_labels)
                    ps = [cnt / len(bin_labels) for cnt in label_cnt.values()]  # Pr(x_i)
                    h = -sum([p * math.log(p, 2) for p in ps])  # Entropy
                    r = len(bin_labels) / len(node)
                    ig -= r * h
                    if r == 0:
                        iv -= 9999999
                    else:
                        iv -= r * math.log(r, 2)
                igr = ig / iv  # Information Gain Ratio
                # print('ig={:.4f}, iv={:.4f}, igr={:.4f}, hc={:.4f}, idx={}, parent={}'
                # .format(ig, iv, igr, h_cur, parent_idx, self.parents[parent_idx]))
                if igr > igr_max:
                    igr_max = igr
                    igr_idx = parent_idx
            # Create next nodes
            if igr_idx is None:
                # No useful split found
                for new_path, new_probs in node:
                    new_table[new_path] = new_probs
                continue
            if allow_disjunct \
                    and len(self.pgm.vars[self.parents[igr_idx]].values) > max(2, len(c)):
                # If disjuncts are allowed, find values that lead to identical subtrees
                # and merge them
                branches = defaultdict(set)
                for value in self.pgm.vars[self.parents[igr_idx]].values:
                    probs = set()
                    values = set()
                    for parent_values, prob in node:
                        if parent_values[igr_idx] == value:
                            new_parent_values = [v for v in parent_values]
                            new_parent_values[igr_idx] = None
                            probs.add(tuple([tuple(new_parent_values), prob]))
                            values.add(value)
                    probs = frozenset(probs)
                    branches[probs].update(values)
                for probs, values in branches.items():
                    newpath = [v for v in curpath]
                    newpath[igr_idx] = frozenset(values)
                    newnode = [tuple(newpath), []]
                    for parent_values, prob in probs:
                        parent_values = list(parent_values)
                        parent_values[igr_idx] = frozenset(values)
                        newnode[1].append((tuple(parent_values), prob))
                    nodes.append(newnode)
            else:
                for value in self.pgm.vars[self.parents[igr_idx]].values:
                    newpath = [v for v in curpath]
                    newpath[igr_idx] = value
                    newnode = [tuple(newpath), []]
                    for parent_values, prob in node:
                        if parent_values[igr_idx] == value:
                            newnode[1].append((parent_values, prob))
                    nodes.append(newnode)
        return self.copy(table=new_table)

    def to_huginnet_node(self, include_layout=False):
        rv = self.pgm.vars[self.rv]
        lines = ["node {} {{".format(rv.clean()),
                 "  label = \"{}\";".format(self.rv)]
        if include_layout:
            lines += ["  position = (100,100);"]
        lines += ["  states = ({});".format(' '.join(['"{}"'.format(v) for v in sorted(rv.values)])),
                 "}\n"]
        return '\n'.join(lines)

    def to_huginnet_potential(self, include_layout=False):
        rv = self.pgm.vars[self.rv]
        name = rv.clean()
        value_idxs = sorted((v,i) for i,v in enumerate(rv.values))
        if len(self.parents) > 0:
            name += ' | ' + ' '.join([rv.clean(p) for p in self.parents])
        lines = ['potential ({}) {{'.format(name),
                 '  % ' + ' '.join([str(v) for v, _ in value_idxs]),
                 '  data = (']
        table = sorted(self.table.items())
        for k, probs in table:
            lines.append('    ' + ' '.join([str(probs[value_idx[1]]) for value_idx in value_idxs]) + ' % ' + ' '.join([str(kk) for kk in k]))
        lines += ['  );',
                  '}\n']
        return '\n'.join(lines)

    def to_xdsl_cpt(self):
        rv = self.pgm.vars[self.rv]
        lines = ['    <cpt id="{}">'.format(rv.clean())]
        value_idxs = sorted((v,i) for i,v in enumerate(rv.values))
        for v, _ in value_idxs:
            lines.append('      <state id="{}" />'.format(v))
        if len(self.parents) > 0:
            lines.append('      <parents>{}</parents>'.format(' '.join([rv.clean(p) for p in self.parents])))
        table = sorted(self.table.items())
        probs = ' '.join([str(probs[value_idx[1]]) for k, probs in table for value_idx in value_idxs])
        lines.append('      <probabilities>{}</probabilities>'.format(probs))
        lines.append('    </cpt>')
        return '\n'.join(lines)

    def to_xdsl_node(self):
        rv = self.pgm.vars[self.rv]
        lines = [
            '      <node id="{}">'.format(rv.clean()),
            '        <name>{}</name>'.format(self.rv),
            '        <interior color="e5f6f7" />',
            '        <outline color="000080" />',
            '        <font color="000000" name="Arial" size="8" />',
            '        <position>100 100 150 150</position>',
            '      </node>']
        return '\n'.join(lines)

    def to_xmlbif_cpt(self):
        rv = self.pgm.vars[self.rv]
        lines = ['<DEFINITION>',
                 '  <FOR>{}</FOR>'.format(rv.name)]
        for parent in self.parents:
            lines.append('  <GIVEN>{}</GIVEN>'.format(parent))
        value_idxs = sorted((v,i) for i,v in enumerate(rv.values))
        table = sorted(self.table.items())
        probs = ' '.join([str(probs[value_idx[1]]) for k, probs in table for value_idx in value_idxs])
        lines.append('  <TABLE>{}</TABLE>'.format(probs))
        lines.append('</DEFINITION>')
        return '\n'.join(lines)

    def to_xmlbif_node(self):
        rv = self.pgm.vars[self.rv]
        lines = [
            '<VARIABLE TYPE="nature">',
            '  <NAME>{}</NAME>'.format(rv)
        ]
        for v in sorted(rv.values):
            lines += ['  <OUTCOME>{}</OUTCOME>'.format(v)]
        lines += ['</VARIABLE>']
        return '\n'.join(lines)

    def to_uai08_preamble(self, cpds):
        function_size = 1 + len(self.parents)
        rv_to_idx = {}
        for idx, cpd in enumerate(cpds):
            rv_to_idx[cpd.rv] = idx
        variables = [str(rv_to_idx[rv]) for rv in self.parents] + [str(rv_to_idx[self.rv])]
        return '{} {}'.format(function_size, ' '.join(variables))

    def to_uai08_function(self):
        rv = self.pgm.vars[self.rv]
        number_entries = str(len(self.table) * len(rv.values))
        lines = [number_entries]
        try:
            table = sorted(self.table.items())
        except TypeError:
            print('ERROR: Cannot convert compressed CPTs to UAI08 format')
            sys.exit(1)
        for k, v in table:
            lines.append(' ' + ' '.join([str(vv) for vv in v]))
        lines.append('')
        return '\n'.join(lines)

    def to_problog(self, pgm, drop_zero=False, use_neglit=False, value_as_term=True, ad_is_function=False):
        lines = []
        var = pgm.vars[self.rv]  # type: Variable
        name = var.clean()
        # if len(self.parents) > 0:
        #   name += ' | '+' '.join([self.rv_clean(p) for p in self.parents])
        # table = sorted(self.table.items())
        table = self.table.items()
        # value_assignments = itertools.product(*[pgm.cpds[parent].value for parent in self.parents)

        line_cnt = 0
        for k, v in table:
            if var.boolean_true is not None and drop_zero and v[var.boolean_true] == 0.0 and not use_neglit:
                continue
            head_problits = []
            if var.boolean_true is None:
                for idx, vv in enumerate(v):
                    if not (drop_zero and vv == 0.0):
                        head_problits.append((vv, var.to_problog_value(var.values[idx], value_as_term)))
            elif isinstance(var.boolean_true, int):
                if drop_zero and v[var.boolean_true] == 0.0 and use_neglit:
                    head_problits.append((None, var.to_problog_value(var.values[1 - var.boolean_true], value_as_term)))
                elif v[var.boolean_true] == 1.0:
                    head_problits.append((None, var.to_problog_value(var.values[var.boolean_true], value_as_term)))
                else:
                    head_problits.append((v[var.boolean_true],
                                          var.to_problog_value(var.values[var.boolean_true], value_as_term)))
            body_lits = []
            for parent, parent_value in zip(self.parents, k):
                if parent_value is not None:
                    parent_var = pgm.vars[parent]
                    body_lits.append(parent_var.to_problog_value(parent_value, value_as_term))

            if ad_is_function:
                for head_cnt, (head_prob, head_lit) in enumerate(head_problits):
                    if len(body_lits) == 0:
                        lines.append('({};1.0)::{}.'.format(head_prob, head_lit))
                    else:
                        # new_probfact = 'pf_{}_{}_{}'.format(self.rv_clean(), line_cnt, head_cnt)
                        # new_body_lits = body_lits + [new_probfact]
                        # lines.append('({};1.0)::{}.'.format(head_prob, new_probfact))
                        # lines.append('{} :- {}.'.format(head_lit, ', '.join(new_body_lits)))
                        lines.append('({};1.0)::{} :- {}.'.format(head_prob, head_lit, ', '.join(body_lits)))
            else:
                if len(body_lits) > 0:
                    body_str = ' :- ' + ', '.join(body_lits)
                else:
                    body_str = ''
                head_strs = []
                for prob, lit in head_problits:
                    if prob is None:
                        head_strs.append(str(lit))
                    else:
                        head_strs.append('{}::{}'.format(prob, lit))
                head_str = '; '.join(head_strs)
                lines.append('{}{}.'.format(head_str, body_str))
            line_cnt += 1

        if ad_is_function and var.boolean_true is None:
            head_lits = [var.to_problog_value(value, value_as_term) for value in var.values]
            # lines.append('false_constraints :- '+', '.join(['\+'+l for l in head_lits])+'.')
            lines.append('constraint([' + ', '.join(['\+' + l for l in head_lits]) + '], false).')
            for lit1, lit2 in itertools.combinations(head_lits, 2):
                # lines.append('false_constraints :- {}, {}.'.format(lit1, lit2))
                lines.append('constraint([{}, {}], false).'.format(lit1, lit2))

        return '\n'.join(lines)

    def to_problog_undirected(self, pgm, drop_zero=False, use_neglit=False, value_as_term=True, ad_is_function=False):
        lines = []
        name = self.name
        table = self.table.items()
        line_cnt = 0
        for k, v in table:
            assert (len(v) == 1)
            body_lits = []
            for parent, parent_value in zip(self.parents, k):
                if parent_value is not None:
                    parent_var = pgm.vars[parent]
                    body_lits.append(parent_var.to_problog_value(parent_value, value_as_term))

            body_str = ' :- ' + ', '.join(body_lits)
            head_str = '{}::{}({})'.format(v[0], name, line_cnt)
            lines.append('{}{}.'.format(head_str, body_str))
            line_cnt += 1

        return '\n'.join(lines)

    def __str__(self):
        lines = []
        try:
            table = sorted(self.table.items())
        except TypeError:
            table = self.table.items()
        for k, v in table:
            lines.append('{}: {}'.format(k, v))
        table = '\n'.join(lines)
        parents = ''
        if len(self.parents) > 0:
            parents = ', '.join(map(str, self.parents))

        if self.rv is not None:
            var = self.pgm.vars[self.rv]
            rv = var.name
            if len(self.parents) > 0:
                rv += ' | '
            return 'Factor ({}{}) = {}\n{}\n'.format(rv, parents, ', '.join([str(v) for v in var.values]), table)
        return 'Factor ({})\n{}\n'.format(parents, table)


class OrCPT(Factor):
    def __init__(self, pgm, rv, parentvalues=None):
        super(OrCPT, self).__init__(pgm, rv, [], [])
        if parentvalues is None:
            self.parentvalues = []
        else:
            self.parentvalues = parentvalues
        self.parents += [pv[0] for pv in self.parentvalues]

    def add(self, parentvalues):
        """Add list of tuples [('a', 1)].
        :param parentvalues: List of tuples (parent, index)
        """
        self.parentvalues += parentvalues
        self.parents += [pv[0] for pv in parentvalues]

    def to_factor(self):
        rv = self.pgm.vars[self.rv]
        parents = sorted(list(set([pv[0] for pv in self.parentvalues])))
        table = dict()
        parent_values = [self.pgm.vars[parent].values for parent in parents]
        for keys in itertools.product(*parent_values):
            is_true = False
            for parent, value in zip(parents, keys):
                if (parent, value) in self.parentvalues:
                    is_true = True
            if is_true:
                table[keys] = [0.0, 1.0]
            else:
                table[keys] = [1.0, 0.0]
        return Factor(self.pgm, self.rv, parents, table)

    def __add__(self, other):
        return OrCPT(self.pgm, self.rv, self.parentvalues + other.parentvalues)

    def __str__(self):
        rv = self.pgm.vars[self.rv]
        table = '\n'.join(['{}'.format(pv) for pv in self.parentvalues])
        parents = ''
        if len(self.parents) > 0:
            parents = ' -- {}'.format(','.join(self.parents))
        return 'OrCPT {} [{}]{}\n{}\n'.format(self.rv, ','.join(map(str, rv.values)), parents, table)
