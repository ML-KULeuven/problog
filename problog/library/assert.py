from problog.extern import problog_export, problog_export_nondet, problog_export_raw


@problog_export_raw("+term")
def assertz(term, target=None, **kwargs):
    problog_export.database += term
    target._cache.reset()  # reset tabling cache
    return [(term,)]


@problog_export_raw("+term")
def retract(term, target=None, **kwargs):
    db = problog_export.database
    nodekey = db.find(term)
    node = db.get_node(nodekey)
    to_erase = node.children.find(term.args)
    if to_erase:
        item = next(to_erase.__iter__())
        node.children.erase((item,))
        target._cache.reset()  # reset tabling cache
        return [(term,)]
    else:
        return []


@problog_export_raw("+term")
def retractall(term, target=None, **kwargs):
    db = problog_export.database
    nodekey = db.find(term)
    node = db.get_node(nodekey)
    to_erase = node.children.find(term.args)
    node.children.erase(to_erase)
    target._cache.reset()  # reset tabling cache
    return [(term,)]
