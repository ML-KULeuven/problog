#!/usr/bin/env python3
# encoding: utf-8
"""
dt2problog.py

Convert Scikit-learn Decision Trees to ProbLog and ProbFoil.

Created by Wannes Meert on 30-11-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""

import sys
import argparse
import logging
from sklearn import tree
import re
import numpy as np
from itertools import product
import pickle

logger = logging.getLogger(__name__)


re_problog = re.compile("[ .()]")

def clean_problog(name):
    if type(name) in [float, np.float64]:
        return "{:.2f}".format(name)
    return re_problog.sub("_", name)


class Rules:
    def __init__(self, decision_tree:tree, feature_names:list = None, class_names:list = None):
        """Create a set of rules from a given decision tree.

        :param decision_tree: Scikit-learn DecisionTreeClassifier
        :param feature_names: List of feature names
        :param class_names: List of class names
        """
        self.clf = decision_tree
        # print(self.decision_tree)
        if feature_names is None:
            self.f_names = ["f_{}".format(i) for i in range(clf.n_features_)]
        else:
            self.f_names = feature_names
        if class_names is None:
            self.c_names = ["c_{}".format(i) for i in range(clf.n_classes_)]
        else:
            self.c_names = class_names
        self.bins = self._compute_bins()
        # print(self.bins)

    def to_problog(self, use_comparison=False, min_prob=0.0):
        """Translate the decision tree to a set of rules representing the tree.

        :param min_prob: Minimum probability. Reduce all probabilities lower
            than this value to 0.0 (to in effect ignore them).
        """
        return self._node_to_problog(0, [],
                                     use_comparison=use_comparison, min_prob=min_prob)

    def to_problog_facts(self):
        r, _ = self._node_to_problog_facts(0, [], 0)
        return r

    def to_probfoil(self, target_idx=None, min_prob=0.0):
        """Translate the decision tree to a set of settings and facts to use
        ProbFoil to learn rules from the paths in the decision tree.

        :param target_idx: Filter on this factor (this changes the model).
        :param min_prob: Minimum probability. Reduce all probabilities lower
            than this value to 0.0 (to in effect ignore them).
        """
        settings = ""

        settings += "%% MODES\n"
        for feature_i in self.bins.keys():
            settings += "mode(f_{}(c,c,+)).\n".format(clean_problog(self.f_names[feature_i]))

        settings += "%% TYPES\n"
        for c_name in self.c_names:
            settings+= "base({}(example)).\n".format(c_name)
        for feature_i in self.bins.keys():
            settings += "base(f_{}(lowerbound, upperbound, example)).\n".format(clean_problog(self.f_names[feature_i]))

        settings += "%% TARGET\n"
        for i, c_name in enumerate(self.c_names):
            if target_idx is None or target_idx == i:
                settings += "learn({}/{}).\n".format(c_name, 1)

        data, _ = self._node_to_problog_facts(0, [], 0, target_idx, min_prob=min_prob)

        return settings, data

    def _compute_bins(self):
        bins = {}
        for f in range(self.clf.n_features_):
            bins[f] = []
        for f, t in zip(self.clf.tree_.feature, self.clf.tree_.threshold):
            if f == -2:
                continue
            bins[f].append(t)
        for ts in bins.values():
            ts.sort()
        return bins

    def _get_true_bins(self, feature_idx, t_min, t_max, ex_i=None):
        bins = self.bins[feature_idx]
        # print("_get_bins {}, {}, {}, {}".format(feature_idx, t_min, t_max, bins))
        bins_str = []
        for b_min, b_max in zip([-np.inf]+bins, bins+[np.inf]):
            if b_min >= t_min and b_max <= t_max:
                if b_min == -np.inf:
                    b_min = "ninf"
                if ex_i is None:
                    ex_i_s = ""
                else:
                    ex_i_s = ",{}".format(ex_i)
                bins_str.append("f_{}({},{}{})".format(clean_problog(self.f_names[feature_idx]),
                                                     clean_problog(b_min),
                                                     clean_problog(b_max), ex_i_s))
        # print(bins_str)
        return bins_str

    def _node_stack_to_body(self, node_stack, ex_i=None, use_comparison=False):
        thresholds = {}
        for f in range(len(self.f_names)):
            thresholds[f] = [-np.inf, np.inf]
        for is_smaller, node_idx in node_stack:
            t = self.clf.tree_.threshold[node_idx]
            f = self.clf.tree_.feature[node_idx]
            ts = thresholds[f]
            if is_smaller:
                if t < ts[1]:
                    ts[1] = t
            else:
                if t > ts[0]:
                    ts[0] = t
            thresholds[f] = ts
        # print(thresholds)
        body = []
        for f, (tmin, tmax) in thresholds.items():
            if use_comparison:
                if tmin == -np.inf and tmax == np.inf:
                    continue
                cond = "f_{}(V{})".format(clean_problog(self.f_names[f]), f)
                if tmin != -np.inf:
                    cond += ", {:.2f} < V{}".format(tmin, f)
                if tmax != np.inf:
                    cond += ", V{} =< {:.2f}".format(f, tmax)
                bins = [cond]
            else:
                bins = self._get_true_bins(f, tmin, tmax, ex_i)
            # print("bins for {}({},{}): {}".format(f, tmin, tmax, bins))
            # print(f'bins for {f}({tmin},{tmax}): {bins}')
            body.append(bins)
            # if len(bins) == 1:
            #     body.append(bins[0])
            # else:
            #     body.append("("+"; ".join(bins)+")")
        # return ", ".join(body)
        return body

    def _node_to_problog(self, node_idx, node_stack, use_comparison=False, min_prob=0.0):
        r = ""
        if self.clf.tree_.children_left[node_idx] == -1 and \
           self.clf.tree_.children_right[node_idx] == -1:
            # Leaf
            values = self.clf.tree_.value[node_idx][0]
            total = sum(values)
            # print("Leaf: {} : {} <- {}".format(node_idx, values, node_stack))
            body = self._node_stack_to_body(node_stack, use_comparison=use_comparison)
            body = ", ".join([bins[0] if len(bins) == 1 else "("+"; ".join(bins)+")" for bins in body])
            for i_class in range(self.clf.n_classes_):
                prob = values[i_class]/total
                if prob <= min_prob:
                    continue
                r += "{}::{} :- {}.\n".format(
                    prob,
                    clean_problog(self.c_names[i_class]),
                    body)
        else:
            # Inner node
            r += self._node_to_problog(self.clf.tree_.children_left[node_idx],
                                       node_stack + [(True,  node_idx)],
                                       use_comparison, min_prob)
            r += self._node_to_problog(self.clf.tree_.children_right[node_idx],
                                       node_stack + [(False, node_idx)],
                                       use_comparison, min_prob)

        return r

    def _node_to_problog_2(self, node_idx, node_stack):
        r = ""
        if self.clf.tree_.children_left[node_idx] == -1 and \
           self.clf.tree_.children_right[node_idx] == -1:
            # Leaf
            values = self.clf.tree_.value[node_idx][0]
            total = sum(values)
            # print("Leaf: {} : {} <- {}".format(node_idx, values, node_stack))
            body = self._node_stack_to_body(node_stack)
            for cur_body in product(*body):
                cur_body = ", ".join(cur_body)
                for i_class in range(self.clf.n_classes_):
                    if values[i_class] == 0:
                        continue
                    r += "{}::{} :- {}.\n".format(
                        values[i_class]/total,
                        clean_problog(self.c_names[i_class]),
                        cur_body)
        else:
            # Inner node
            r += self._node_to_problog_2(self.clf.tree_.children_left[node_idx],  node_stack + [(True,  node_idx)])
            r += self._node_to_problog_2(self.clf.tree_.children_right[node_idx], node_stack + [(False, node_idx)])

        return r

    def _node_to_problog_facts(self, node_idx, node_stack, ex_i, only_class_idx=None, min_prob=0.0):
        r = ""
        if self.clf.tree_.children_left[node_idx] == -1 and \
           self.clf.tree_.children_right[node_idx] == -1:
            # Leaf
            values = self.clf.tree_.value[node_idx][0]
            total = sum(values)
            # print("Leaf: {} : {} <- {}".format(node_idx, values, node_stack))
            body = self._node_stack_to_body(node_stack, "{ex_i}")
            for cur_body in product(*body):
                include = False
                for i_class in range(self.clf.n_classes_):
                    # if values[i_class] == 0:
                    #     continue
                    value = values[i_class] / total
                    if only_class_idx is not None and \
                            (i_class != only_class_idx or value <= min_prob):
                        continue
                    if value == 1.0:
                        # value = ""
                        value = "{}::".format(value)
                    else:
                        value = "{}::".format(value)
                    r += "{}{}({}).  % {}/{}\n".format(
                        value,
                        clean_problog(self.c_names[i_class]), ex_i,
                        values[i_class], total)
                    include = True
                if include:
                    for conj_elmt in cur_body:
                        r += "{}.\n".format(conj_elmt.format(ex_i=ex_i))
                        # for disj_elmt in conj_elmt:
                        #     r += "{}.\n".format(disj_elmt)
                    r += "%%\n"
                    ex_i += 1
        else:
            # Inner node
            r1, ex_1 = self._node_to_problog_facts(self.clf.tree_.children_left[node_idx],
                                                   node_stack + [(True,  node_idx)], ex_i, only_class_idx,
                                                   min_prob=min_prob)
            r2, ex_2 = self._node_to_problog_facts(self.clf.tree_.children_right[node_idx],
                                                   node_stack + [(False, node_idx)], ex_1, only_class_idx,
                                                   min_prob=min_prob)
            r += r1 + r2
            ex_i = ex_2
        return r, ex_i


def main(argv=None):
    parser = argparse.ArgumentParser(description='Convert a Scikit-learn decision tree to ProbLog')
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    parser.add_argument('--pickled', required=True, help='Input pickled Decision Tree')
    parser.add_argument('--problog', help='ProbLog output file')
    parser.add_argument('--probfoil', help='Probfoil output files (basename)')
    parser.add_argument('--minprob', type=float, default=0.0, help="Minimum probability of a class in a leaf")
    parser.add_argument('--features', help="Comma-separated list of feature names")
    parser.add_argument('--classes', help="Comma-separated list of class names")
    args = parser.parse_args(argv)

    logger.setLevel(logging.ERROR-10*(0 if args.verbose is None else args.verbose))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    clf = pickle.load(args.pickled)
    if args.features is not None:
        features = args.features.split(",")
    if args.classes is not None:
        classes = args.classes.split(",")
    rules = Rules(clf, features, classes)

    if args.problog is not None:
        program = rules.to_problog(use_comparison=True, min_prob=args.minprob)
        with open(args.problog, 'w') as ofile:
            print(program, file=ofile)
    if args.probfoil is not None:
        settings, data = rules.to_probfoil(min_prob=args.minprob)
        with open(args.problog+".settings", 'w') as ofile:
            print(settings, file=ofile)
        with open(args.problog+".data", 'w') as ofile:
            print(data, file=ofile)


if __name__ == "__main__":
    sys.exit(main())

