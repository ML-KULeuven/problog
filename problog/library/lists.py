from __future__ import print_function

from problog.extern import problog_export_nondet
from problog.logic import term2list, list2term
from collections import defaultdict


@problog_export_nondet("+term", "-term", "-list")
def enum_groups(group_values):
    group_values_l = term2list(group_values, False)

    grouped = defaultdict(list)

    for gv in group_values_l:
        grouped[gv.args[0]].append(gv.args[1])
    return list(grouped.items())


@problog_export_nondet("+term", "+term", "-term", "-list")
def enum_groups(groups, values):
    groups_l = term2list(groups, False)
    values_l = term2list(values, False)

    grouped = defaultdict(list)

    for g, v in zip(groups_l, values_l):
        grouped[g].append(v)
    return list(grouped.items())
