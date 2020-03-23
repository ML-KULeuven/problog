from problog.extern import problog_export, problog_export_nondet
from collections import OrderedDict
from problog.logic import Constant

recdb_key = "_rec_db"


class LL(object):
    def __init__(self):
        self.first = None
        self.last = None

    def push_left(self, item):
        entry = LLEntry(self, item, None, self.first)
        if self.first is None:  # list is currently empty
            self.first = entry
            self.last = entry
        else:
            self.first.prv = entry
            self.first = entry
        return entry

    def push_right(self, item):
        entry = LLEntry(self, item, self.last, None)
        if self.last is None:  # list if currently empty
            self.first = entry
            self.last = entry
        else:
            self.last.nxt = entry
            self.last = entry
        return entry

    def erase(self, item):
        if item.prv is None and item.nxt is None:  # only element in list
            self.first = None
            self.last = None
        elif item.prv is None:  # first element
            item.nxt.prv = None
            self.first = item.nxt
        elif item.nxt is None:  # last element
            item.prv.nxt = None
            self.last = item.prv
        else:
            item.prv.nxt = item.nxt
            item.nxt.prv = item.prv

    def is_empty(self):
        return self.first is None

    def __iter__(self):
        it = self.first
        while it is not None:
            yield it
            it = it.nxt


class LLEntry(object):

    __slots__ = ("lst", "val", "prv", "nxt")

    def __init__(self, lst, val, prv, nxt):
        self.lst = lst
        self.prv = prv
        self.nxt = nxt
        self.val = val

    def erase(self):
        self.lst.erase(self)


def record_base(key, term, append):
    db = problog_export.database.get_data(recdb_key)
    if db is None:
        db = OrderedDict()
        problog_export.database.set_data(recdb_key, db)
    lst = db.get(key)
    if lst is None:
        lst = LL()
        db[key] = lst
    if append:
        return lst.push_right(term)
    else:
        return lst.push_left(term)


@problog_export_nondet("-term")
def current_key():
    db = problog_export.database.get_data(recdb_key, {})
    return list(db.keys())


@problog_export("+term", "+term", "-term")
def recorda(key, term):
    return Constant(record_base(key, term, False))


@problog_export("+term", "+term")
def recorda(key, term):
    record_base(key, term, False)
    return ()


@problog_export("+term", "+term", "-int")
def recordz(key, term):
    return Constant(record_base(key, term, True))


@problog_export("+term", "+term")
def recordz(key, term):
    record_base(key, term, True)
    return ()


@problog_export("+term")
def erase(ref):
    ref.functor.erase()
    return ()


@problog_export_nondet("+term", "-term")
def recorded(key):
    db = problog_export.database.get_data(recdb_key, {})
    lst = db.get(key)
    if lst is None:
        return ()
    else:
        return [x.val for x in lst]


@problog_export_nondet("+term", "-term", "-term")
def recorded(key):
    db = problog_export.database.get_data(recdb_key, {})
    lst = db.get(key)
    if lst is None:
        return ()
    else:
        return [(x.val, Constant(x)) for x in lst]


@problog_export("+term", "-term")
def instance(ref):
    return ref.functor.val


# @problog_export_nondet('-term', '-term', '-term')
# def recorded():
#     db = problog_export.database.get_data(recdb_key)
#
#     for key, lst in db.items():
#         if lst is None:
#             return ()
#         else:
#             return [(key, x.val, Constant(x)) for x in lst]
