from __future__ import print_function

from problog.logic import unquote
from problog.extern import problog_export, problog_export_raw


@problog_export('+list', '-str')
def concat(terms):
    return ''.join(map(lambda x: unquote(str(x)), terms))
    
    
@problog_export('+str', '+list', '-str')
def join(sep, terms):
    return unquote(sep).join(map(lambda x: unquote(str(x)), terms))
