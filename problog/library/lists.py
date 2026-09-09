from collections import defaultdict

from problog.engine_unify import UnifyError, unify_value
from problog.extern import problog_export, problog_export_nondet, problog_export_raw
from problog.logic import Term, is_ground, term2list


def _ground_list(term):
    """The elements of term when it is a ground list, None otherwise."""
    if not is_ground(term):
        return None
    try:
        return term2list(term, deep=False)
    except ValueError:  # not a fixed length list
        return None


@problog_export("+term", "-term", functor="ground_list")
def _ground_list_check(term):
    """Whether memberchk_ground/2 applies to term, as 'true' or 'false'."""
    return Term("true") if _ground_list(term) is not None else Term("false")


@problog_export_raw("+term", "+term", functor="memberchk_ground")
def _memberchk_ground(element, term, **kwargs):
    """memberchk/2 for a ground list, fails for anything else.

    Walking a list with the clauses in lists.pl costs one tabled call per
    element, which dominates any program that carries a list along as a set of
    what it has already visited.  The list being ground is what makes doing it
    here equivalent: unifying with one of its elements can only bind variables
    of the element being looked up, never anything inside the list.
    """
    elements = _ground_list(term)
    if elements is None:
        return []
    if element is None:  # unbound, so it unifies with the first element
        return [(elements[0], term)] if elements else []
    for e in elements:
        try:
            unify_value(element, e, {})
        except UnifyError:
            continue
        return [(e, term)]
    return []


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
