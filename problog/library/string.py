from __future__ import print_function

from problog.logic import unquote, make_safe, list2term, Term
from problog.extern import problog_export, problog_export_raw


@problog_export("+list", "-str")
def concat(terms):
    return make_safe("".join(map(lambda x: unquote(str(x)), terms)))


@problog_export("+str", "+list", "-str")
def join(sep, terms):
    return make_safe(unquote(sep).join(map(lambda x: unquote(str(x)), terms)))


@problog_export("+str", "-list")
def str2lst(string):
    return map(Term, map(make_safe, string))


@problog_export("+list", "-str")
def lst2str(terms):
    return join("", terms)
