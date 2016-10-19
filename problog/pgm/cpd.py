"""
Conditional Probability Distributions (CPD) to expert ProbLog to a
Probabilistic Graphical Model (PGM).

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
from collections import Counter, defaultdict


class PGM(object):
    def __init__(self, directed=True, vars=None, factors=None):
        """Probabilistic Graphical Model."""
        self.directed = directed
        self.factors = {}
        self.vars = {}
        if vars:
            for var in vars:
                self.add_var(var)
        if factors:
            for factor in factors:
                self.add_factor(factor)

    def copy(self, **kwargs):
        return PGM(
            directed=kwargs.get('directed', self.directed),
            factors=kwargs.get('factors', self.factors.values()),
            vars=kwargs.get('vars', self.vars.values())
        )

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
        assert(name is not None)
        factor.pgm = self
        if name in self.factors:
            self.factors[name] += factor
        else:
            self.factors[name] = factor

    def factors_topological(self):
        """Return the factors in a topological order."""
        # Links from parent-node to child-node
        if not self.directed:
            return self.factors.values()
        links = dict()
        for factor in self.factors.values():
            for parent in factor.parents:
                if parent in links:
                    links[parent].add(factor.rv)
                else:
                    links[parent] = {factor.rv}
        all_links = set(links.keys())
        nonroot = reduce(set.union, links.values())
        root = all_links - nonroot
        queue = root
        visited = set()
        factors = []
        while len(queue) > 0:
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
                    factors.append(factor)
                    visited.add(rv)
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

    def to_hugin_net(self):
        """Export PGM to the Hugin net format.
        http://www.hugin.com/technology/documentation/api-manuals
        """
        assert self.directed
        cpds = [cpd.to_factor(self) for cpd in self.factors_topological()]
        lines = ["%% Hugin Net format",
                 "%% Created on {}\n".format(datetime.now()),
                 "%% Network\n",
                 "net {",
                 "  node_size = (50,50);",
                 "}\n",
                 "%% Nodes\n"]
        lines += [cpd.to_HuginNetNode() for cpd in cpds]
        lines += ["%% Potentials\n"]
        lines += [cpd.to_HuginNetPotential() for cpd in cpds]
        return '\n'.join(lines)

    def to_xdsl(self):
        """Export PGM to the XDSL format defined by SMILE.
        https://dslpitt.org/genie/wiki/Appendices:_XDSL_File_Format_-_XML_Schema_Definitions
        """
        assert self.directed
        cpds = [cpd.to_factor(self) for cpd in self.factors_topological()]
        lines = ['<?xml version="1.0" encoding="ISO-8859-1" ?>',
                 '<smile version="1.0" id="Aa" numsamples="1000">',
                 '  <nodes>']
        lines += [cpd.to_XdslCpt() for cpd in cpds]
        lines += ['  </nodes>',
                  '  <extensions>',
                  '    <genie version="1.0" app="ProbLog" name="Network1" faultnameformat="nodestate">']
        lines += [cpd.to_XdslNode() for cpd in cpds]
        lines += ['    </genie>',
                  '  </extensions>',
                  '</smile>']
        return '\n'.join(lines)

    def to_uai08(self):
        """Export PGM to the format used in the UAI 2008 competition.
        http://graphmod.ics.uci.edu/uai08/FileFormat
        """
        assert self.directed
        cpds = [cpd.to_factor(self) for cpd in self.factors_topological()]
        number_variables = str(len(cpds))
        domain_sizes = [str(len(cpd.values)) for cpd in cpds]
        number_functions = str(len(cpds))
        lines = ['BAYES',
                 number_variables,
                 ' '.join(domain_sizes),
                 number_functions]
        lines += [cpd.to_Uai08Preamble(cpds) for cpd in cpds]
        lines += ['']
        lines += [cpd.to_Uai08Function() for cpd in cpds]
        return '\n'.join(lines)

    def to_problog(self, drop_zero=False, use_neglit=False, value_as_term=True, ad_is_function=False):
        """Export PGM to ProbLog.
        :param ad_is_function: Experimental
        :param value_as_term: Include the variable's value as the last term instead of as part of the predicate name
        :param use_neglit: Use negative literals if it simplifies the program
        :param drop_zero: Do not include head literals with probability zero
        """
        factors = [factor.to_factor() for factor in self.factors_topological()]
        lines = ["%% ProbLog program",
                 "%% Created on {}\n".format(datetime.now())]
        if self.directed:
            lines += [factor.to_ProbLog(self, drop_zero=drop_zero, use_neglit=use_neglit,
                                        value_as_term=value_as_term, ad_is_function=ad_is_function) for factor in factors]
        else:
            lines += [factor.to_ProbLog_undirected(self, drop_zero=drop_zero, use_neglit=use_neglit,
                                                   value_as_term=value_as_term, ad_is_function=ad_is_function) for factor in factors]
        # if ad_is_function:
            # lines += ["evidence(false_constraints,false)."]
        return '\n'.join(lines)

    def to_graphviz(self):
        """Export PGM to Graphviz dot format.
        http://www.graphviz.org
        """
        assert self.directed
        lines = ['digraph bayesnet {']
        for cpd in self.factors.values():
            lines.append('  {} [label="{}"];'.format(cpd.rv_clean(), cpd.rv))
            for p in cpd.parents:
                lines.append('  {} -> {}'.format(cpd.rv_clean(p), cpd.rv_clean()))
        lines += ['}']
        return '\n'.join(lines)

    def __str__(self):
        return '\n'.join([str(factor) for factor in self.factors.values()])


re_toundercore = re.compile(r"[\(\),\]\[ ]")
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
    def __init__(self, name, values, detect_boolean=True, force_boolean=False, boolean_true=None):
        """Conditional Probability Distribution."""
        self.name = name
        self.values = values
        self.latent = False
        self.booleantrue = boolean_true  # Value that represents true
        if (force_boolean or detect_boolean) and len(self.values) == 2:
            for values in boolean_values:
                if (values[0] == self.values[0] and values[1] == self.values[1]) or \
                   (type(self.values[0]) == str and type(self.values[1]) == str and
                    values[0] == self.values[0].lower() and values[1] == self.values[1].lower()):
                    self.booleantrue = 1
                    break
                elif (values[1] == self.values[0] and values[0] == self.values[1]) or \
                     (type(self.values[0]) == str and type(self.values[1]) == str and
                      values[1] == self.values[0].lower() and values[0] == self.values[1].lower()):
                    self.booleantrue = 0
                    break
            if force_boolean and self.booleantrue is None:
                self.booleantrue = 1

    def clean(self, value=None):
        if value is None:
            rv = self.name
        else:
            rv = value
        rv = re_toundercore.sub('_', rv)
        rv = re_toremove.sub('', rv)
        if not rv[0].islower():
            rv = rv[0].lower() + rv[1:]
            if not rv[0].islower():
                rv = "v"+rv
        return rv

    def to_ProbLogValue(self, value, value_as_term=True):
        if self.booleantrue is None:
            if type(value) is frozenset and len(value) == 1:
                value, = value
            if type(value) is frozenset:
                # It is a disjunction of possible values
                if len(value) == len(self.values)-1:
                    # This is the negation of one value
                    new_value, = frozenset(self.values) - value
                    if value_as_term:
                        return '\+'+self.clean() + '("' + str(new_value) + '")'
                    else:
                        return '\+'+self.clean() + '_' + self.clean(str(new_value))
                else:
                    if value_as_term:
                        return '('+'; '.join([self.clean() + '("' + str(new_value) + '")' for new_value in value])+')'
                    else:
                        return '(' + '; '.join([self.clean() + '_' + self.clean(str(new_value)) for new_value in value]) + ')'
            else:
                if value_as_term:
                    return self.clean()+'("'+str(value)+'")'
                else:
                    return self.clean()+'_'+self.clean(str(value))
        else:
            # It is a Boolean atom
            if self.values[self.booleantrue] == value:
                return self.clean()
            elif self.values[1-self.booleantrue] == value:
                return '\+'+self.clean()
            else:
                raise Exception('Unknown value: {} = {}'.format(self.name, value))

    def __str__(self):
        return '{}'.format(self.name)

    def copy(self, **kwargs):
        return Variable(
            name=kwargs.get('name', self.name),
            values=kwargs.get('values', self.values),
            boolean_true=kwargs.get('boolean_true', self.booleantrue)
        )


class Factor(object):
    def __init__(self, pgm, rv, parents, table, name=None, *args, **kwargs):
        """Conditional Probability Table with discrete probabilities.

        :param rv: random variable
        :param values: random variable domain
        :param parents: parent random variables
        :param table:

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
        nodes = [(tuple([None]*len(self.parents)), table)]
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
            if len(c.keys()) == len(v):
                for new_path, new_probs in node:
                    new_table[new_path] = new_probs
                continue
            if len(c.keys()) == 1:
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
                    ps = [cnt/len(bin_labels) for cnt in label_cnt.values()]  # Pr(x_i)
                    h = -sum([p*math.log(p, 2) for p in ps])  # Entropy
                    r = len(bin_labels)/len(node)
                    ig -= r*h
                    if r == 0:
                        iv -= 9999999
                    else:
                        iv -= r*math.log(r,2)
                igr = ig/iv  # Information Gain Ratio
                # print('ig={:.4f}, iv={:.4f}, igr={:.4f}, hc={:.4f}, idx={}, parent={}'.format(ig, iv, igr, h_cur, parent_idx, self.parents[parent_idx]))
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

    def to_HuginNetNode(self):
        rv = self.pgm.vars[self.rv]
        lines = ["node {} {{".format(rv.clean()),
                 "  label = \"{}\";".format(self.rv),
                 "  position = (100,100);",
                 "  states = ({});".format(' '.join(['"{}"'.format(v) for v in rv.values])),
                 "}\n"]
        return '\n'.join(lines)

    def to_HuginNetPotential(self):
        rv = self.pgm.vars[self.rv]
        name = rv.clean()
        if len(self.parents) > 0:
            name += ' | '+' '.join([rv.clean(p) for p in self.parents])
        lines = ['potential ({}) {{'.format(name),
                 '  % '+' '.join([str(v) for v in rv.values]),
                 '  data = (']
        table = sorted(self.table.items())
        for k, v in table:
            lines.append('    '+' '.join([str(vv) for vv in v])+' % '+' '.join([str(kk) for kk in k]))
        lines += ['  );',
                  '}\n']
        return '\n'.join(lines)

    def to_XdslCpt(self):
        rv = self.pgm.vars[self.rv]
        lines = ['    <cpt id="{}">'.format(rv.clean())]
        for v in rv.values:
            lines.append('      <state id="{}" />'.format(v))
        if len(self.parents) > 0:
            lines.append('      <parents>{}</parents>'.format(' '.join([rv.clean(p) for p in self.parents])))
        table = sorted(self.table.items())
        probs = ' '.join([str(value) for k, values in table for value in values])
        lines.append('      <probabilities>{}</probabilities>'.format(probs))
        lines.append('    </cpt>')
        return '\n'.join(lines)

    def to_XdslNode(self):
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

    def to_Uai08Preamble(self, cpds):
        function_size = 1 + len(self.parents)
        rvToIdx = {}
        for idx, cpd in enumerate(cpds):
            rvToIdx[cpd.rv] = idx
        variables = [str(rvToIdx[rv]) for rv in self.parents] + [str(rvToIdx[self.rv])]
        return '{} {}'.format(function_size, ' '.join(variables))

    def to_Uai08Function(self):
        rv = self.pgm.vars[self.rv]
        number_entries = str(len(self.table)*len(rv.values))
        lines = [number_entries]
        table = sorted(self.table.items())
        for k, v in table:
            lines.append(' '+' '.join([str(vv) for vv in v]))
        lines.append('')
        return '\n'.join(lines)

    def to_ProbLog(self, pgm, drop_zero=False, use_neglit=False, value_as_term=True, ad_is_function=False):
        lines = []
        var = pgm.vars[self.rv]
        name = var.clean()
        # if len(self.parents) > 0:
        #   name += ' | '+' '.join([self.rv_clean(p) for p in self.parents])
        # table = sorted(self.table.items())
        table = self.table.items()
        # value_assignments = itertools.product(*[pgm.cpds[parent].value for parent in self.parents)

        line_cnt = 0
        for k, v in table:
            if var.booleantrue is not None and drop_zero and v[var.booleantrue] == 0.0 and not use_neglit:
                continue
            head_problits = []
            if var.booleantrue is None:
                for idx, vv in enumerate(v):
                    if not (drop_zero and vv == 0.0):
                        head_problits.append((vv, var.to_ProbLogValue(var.values[idx], value_as_term)))
            else:
                if drop_zero and v[var.booleantrue] == 0.0 and use_neglit:
                    head_problits.append((None, var.to_ProbLogValue(var.values[1-var.booleantrue], value_as_term)))
                elif v[var.booleantrue] == 1.0:
                    head_problits.append((None,var.to_ProbLogValue(var.values[var.booleantrue], value_as_term)))
                else:
                    head_problits.append((v[var.booleantrue],
                                          var.to_ProbLogValue(var.values[var.booleantrue], value_as_term)))
            body_lits = []
            for parent, parent_value in zip(self.parents, k):
                if parent_value is not None:
                    parent_var = pgm.vars[parent]
                    body_lits.append(parent_var.to_ProbLogValue(parent_value, value_as_term))

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

        if ad_is_function and var.booleantrue is None:
            head_lits = [var.to_ProbLogValue(value, value_as_term) for value in var.values]
            # lines.append('false_constraints :- '+', '.join(['\+'+l for l in head_lits])+'.')
            lines.append('constraint(['+', '.join(['\+'+l for l in head_lits])+'], false).')
            for lit1, lit2 in itertools.combinations(head_lits, 2):
                # lines.append('false_constraints :- {}, {}.'.format(lit1, lit2))
                lines.append('constraint([{}, {}], false).'.format(lit1, lit2))

        return '\n'.join(lines)

    def to_ProbLog_undirected(self, pgm, drop_zero=False, use_neglit=False, value_as_term=True, ad_is_function=False):
        lines = []
        name = self.name
        table = self.table.items()
        line_cnt = 0
        for k, v in table:
            assert(len(v) == 1)
            body_lits = []
            for parent, parent_value in zip(self.parents, k):
                if parent_value is not None:
                    parent_var = pgm.vars[parent]
                    body_lits.append(parent_var.to_ProbLogValue(parent_value, value_as_term))

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
        super(OrCPT, self).__init__(pgm, rv, set(), [])
        if parentvalues is None:
            self.parentvalues = []
        else:
            self.parentvalues = parentvalues
        self.parents.update([pv[0] for pv in self.parentvalues])

    def add(self, parentvalues):
        """Add list of tuples [('a', 1)].
        :param parentvalues: List of tuples (parent, index)
        """
        self.parentvalues += parentvalues
        self.parents.update([pv[0] for pv in parentvalues])

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
        return Factor(self.pgm, self.rv, rv.values, parents, table)

    def __add__(self, other):
        return OrCPT(self.pgm, self.rv, self.parentvalues + other.parentvalues)

    def __str__(self):
        rv = self.pgm.vars[self.rv]
        table = '\n'.join(['{}'.format(pv) for pv in self.parentvalues])
        parents = ''
        if len(self.parents) > 0:
            parents = ' -- {}'.format(','.join(self.parents))
        return 'OrCPT {} [{}]{}\n{}\n'.format(self.rv, ','.join(map(str,rv.values)), parents, table)
