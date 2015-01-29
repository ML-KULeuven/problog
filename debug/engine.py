#! /usr/bin/env python

from __future__ import print_function

import os
import sys

problog_path =os.path.abspath(os.path.join(os.path.dirname(__file__),'../') )
sys.path.insert(0, problog_path)

import problog

def test(filename, debug=None, trace=None) :
    import problog
    
    pl = problog.program.PrologFile(filename)
    
    target = problog.formula.LogicFormula()
    
    eng = problog.engine.DefaultEngine()
    db = eng.prepare(pl)
    
    context = [None]
    
    print ('== Database ==')
    print (db)
    
    print ()
    print ('== Results ==')
        
    query_node = db.find(problog.logic.Term('query',None) )
    queries = eng.execute( query_node, database=db, target=target, context=[None])
    
    # print (queries)
    #
    # sys.exit()
    env = { 'database': db, 'target': target }
    
    i = 0
    for query in queries :
        i += 1
        query = query[0][0]
        print ("Query %s" % query)
        target = eng.ground( db, query, gp=target, label='query', debug=debug,trace=trace) 
        results = target.queries()
        if results :
            for q, node in results :
                print ('\t', q, { 0: 'true' }.get(node,node)  )
        else :
            print ('\t', 'fail')
    
    print ()
    print ("== Ground program ==")
    print (target)
    
    
    with open('engine_debug.dot', 'w') as f :
        print (target.toDot(), file=f)
    
    
if __name__ == '__main__' :
    import argparse
    
    p = argparse.ArgumentParser()
    p.add_argument('filename')
    p.add_argument('-d', '--debug', action='store_true')
    p.add_argument('-t', '--trace', action='store_true')    
    args = p.parse_args()
    test(**vars(args))