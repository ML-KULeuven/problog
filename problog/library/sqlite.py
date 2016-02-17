from __future__ import print_function

from problog.extern import problog_export, problog_export_nondet, problog_export_raw

from problog.logic import Term, Constant

import sqlite3


def convert_value(dbvalue):
    if type(dbvalue) == str:
        return Term("'" + dbvalue + "'")
    else:
        return Constant(dbvalue)


def get_colnames(conn, tablename):
    cur = conn.cursor()
    cur.execute('SELECT * FROM %s WHERE 0;' % tablename)
    res = [x[0] for x in cur.description]
    cur.close()
    return res


@problog_export('+str')
def sqlite_load(filename):

    filename = problog_export.database.resolve_filename(filename)
    conn = sqlite3.connect(filename)

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [x[0] for x in cursor.fetchall()]
    cursor.close()

    for table in tables:
        columns = get_colnames(conn, table)
        types = ['+term'] * len(columns)
        problog_export_raw(*types)(QueryFunc(conn, table, columns), funcname=table)

    return ()


class QueryFunc(object):

    def __init__(self, db, tablename, columns):
        self.db = db
        self.tablename = tablename
        self.columns = columns

    def __call__(self, *args):
        where = []
        for c, a in zip(self.columns, args):
            if a is not None:
                where.append('%s = %s' % (c, a))
        where = ' AND '.join(where)
        if where:
            where = ' WHERE ' + where

        query = 'SELECT %s FROM %s%s' % (', '.join(self.columns), self.tablename, where)
        cur = self.db.cursor()
        cur.execute(query)
        res = [tuple(map(convert_value, r)) for r in cur.fetchall()]
        cur.close()
        return res

# @problog_export_nondet('+str', '+str', '-term')
# def db_query(dbname, table):
#     colname = get_colnames(dbname, table)
#     print (colname)


@problog_export_raw('+str', '+str', '?int', '?str', '?str')
def db_query(dbname, table, arg1, arg2, arg3):

    return [(dbname, table, 3, 'aaa', '333')]

    # return [((arg_vals), p)]
    # return [((arg_vals))]



@problog_export_nondet('+str', '+str', '+str', '-term')
def db_query2(dbname, table, arg1, arg2):
    pass
#
#
# @problog_export_nondet('+str')
# def db_query3(dbname, table, arg1, arg2, arg3):
#     pass
#
#
# @problog_export_nondet('+str')
# def db_query4(dbname, table, arg1, arg2, arg3, arg4):
#     pass
#
#
# @problog_export_nondet('+str')
# def db_query5(dbname, table, arg1, arg2, arg3, arg4, arg5):
#     pass
#
#
# @problog_export_nondet('+str')
# def db_query6(dbname, table, arg1, arg2, arg3, arg4, arg5, arg6):
#     pass
