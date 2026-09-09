from problog.extern import problog_export, problog_export_nondet, problog_export_raw
from problog.logic import Term


@problog_export("+str", "+str", "-str")
def concat_str(arg1, arg2):
    return arg1 + arg2


@problog_export("+int", "+int", "-int")
def int_plus(arg1, arg2):
    return arg1 + arg2


@problog_export("+list", "+list", "-list")
def concat_list(arg1, arg2):
    return arg1 + arg2


@problog_export("+int", "+int", "-int", "-int")
def int_plus_times(a, b):
    return a + b, a * b


@problog_export_nondet("+int", "+int", "-int")
def int_between(a, b):
    return list(range(a, b + 1))


@problog_export_raw("+term", "+term")
def pair(a, b, **kwargs):
    """Answers pair(one, two) and nothing else.

    A raw export receives every argument, bound or not, and the ones the
    caller had bound are unified with what it answers, so a call naming any
    other value has to fail.
    """
    return [(Term("one"), Term("two"))]
