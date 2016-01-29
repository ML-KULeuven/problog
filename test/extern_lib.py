from problog.extern import problog_export_nondet, problog_export

@problog_export('+str', '+str', '-str')
def concat_str(arg1, arg2):
    return arg1 + arg2


@problog_export('+int', '+int', '-int')
def int_plus(arg1, arg2):
    return arg1 + arg2


@problog_export('+list', '+list', '-list')
def concat_list(arg1, arg2):
    return arg1 + arg2


@problog_export('+int', '+int', '-int', '-int')
def int_plus_times(a, b):
    return a + b, a * b


@problog_export_nondet('+int', '+int', '-int')
def int_between(a, b):
    return list(range(a, b + 1))
