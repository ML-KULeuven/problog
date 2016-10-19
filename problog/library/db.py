from __future__ import print_function

from problog.extern import problog_export, problog_export_nondet, problog_export_raw

from problog.logic import Term, Constant
from problog.errors import UserError, InvalidValue

import sqlite3
import os
import logging

logger = logging.getLogger('problog')

def db2pl(dbvalue):
    if type(dbvalue) == str:
        return Term("'" + dbvalue + "'")
    else:
        return Constant(dbvalue)


def pl2db(t):
    if isinstance(t, Constant):
        return t.value
    else:
        return str(t).strip("'")


def get_colnames(conn, tablename):
    cur = conn.cursor()
    cur.execute('SELECT * FROM %s WHERE 0;' % tablename)
    res = [x[0] for x in cur.description]
    cur.close()
    return res


@problog_export('+str')
def sqlite_load(filename):

    filename = problog_export.database.resolve_filename(filename)
    if not os.path.exists(filename):
        raise UserError('Can\'t find database \'%s\'' % filename)

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


@problog_export('+str', '+str')
def csv_load(filename, predicate):
    import tempfile
    import csv
    import re

    filename = problog_export.database.resolve_filename(filename)
    if not os.path.exists(filename):
        raise UserError('Can\'t find csv file \'%s\'' % filename)

    csvfile = open(filename, 'r')
    line = 0
    reader = csv.reader(csvfile)
    # Column names
    row = next(reader)
    line += 1
    invalid_chars = re.compile(r'''[^a-zA-Z0-9_]''')
    columns = [invalid_chars.sub('',field) for field in row]

    filepath, ext = os.path.splitext(os.path.basename(filename))
    sql_dir = tempfile.mkdtemp()
    sql_filename = os.path.join(sql_dir, filepath+'.sqlite')
    idx = 1
    while os.path.exists(sql_filename):
        sql_filename = os.path.join(sql_dir, filepath+'_'+str(idx)+'.sqlite')
        idx += 1
    logger.debug('CSV->SQLite: '+sql_filename)
    conn = sqlite3.connect(sql_filename)

    cursor = conn.cursor()

    row = next(reader)
    line += 1
    column_types_sql = []
    column_types_py = []
    for field in row:
        try:
            value = int(field)
            column_types_sql.append('INTEGER')
            column_types_py.append(int)
            continue
        except ValueError:
            pass
        try:
            value = float(field)
            column_types_sql.append('REAL')
            column_types_py.append(float)
            continue
        except ValueError:
            pass
        column_types_sql.append('TEXT')
        column_types_py.append(str)

    coldefs = [n+" "+t for n,t in zip(columns,column_types_sql)]
    cursor.execute("CREATE TABLE "+predicate+"("+",".join(coldefs)+");")

    def insert_statement(row):
        values = []
        for value, sqltype, pytype in zip(row, column_types_sql, column_types_py):
            try:
                if sqltype == 'TEXT':
                    values.append("'"+pytype(value.strip())+"'")
                else:
                    values.append(str(pytype(value)))
            except ValueError:
                raise InvalidValue('{}:{}. Expected type {}, found value {}'.format(filename, line, pytype.__name__, value))
        return "INSERT INTO "+predicate+"("+",".join(columns)+") VALUES ("+",".join(values)+");"

    cursor.execute(insert_statement(row))
    for row in reader:
        line += 1
        if len(row) == 0:
            continue
        cursor.execute(insert_statement(row))

    conn.commit()
    cursor.close()
    csvfile.close()

    types = ['+term'] * len(columns)
    problog_export_raw(*types)(QueryFunc(conn, predicate, columns), funcname=predicate)

    return ()


class QueryFunc(object):

    def __init__(self, db, tablename, columns):
        self.db = db
        self.tablename = tablename
        self.columns = columns

    def __call__(self, *args, **kwargs):
        where = []
        values = []
        for c, a in zip(self.columns, args):
            if a is not None:
                where.append('%s = ?' % c)
                values.append(pl2db(a))
        where = ' AND '.join(where)
        if where:
            where = ' WHERE ' + where

        query = 'SELECT %s FROM %s%s' % (', '.join(self.columns), self.tablename, where)
        cur = self.db.cursor()

        cur.execute(query, values)
        res = [tuple(map(db2pl, r)) for r in cur.fetchall()]
        cur.close()
        return res

