from problog.extern import problog_export, problog_export_nondet, problog_export_raw

@problog_export('+str', '+str', '-int')
def str_sum(a, b):
    """Computes the sum of two numbers."""
    return int(a) + int(b)
