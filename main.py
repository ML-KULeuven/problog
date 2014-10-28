#! /usr/bin/env python

from __future__ import print_function

import sys, os, time, subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

sys.setrecursionlimit(10000)

from problog.logic import Term, Var, Constant
from problog.logic.program import PrologFile, ClauseDB
from problog.logic.prolog import PrologEngine
from problog.logic.engine import GroundProgram, Debugger

def main( filename, trace=False ) :
    
    basename = os.path.splitext(filename)[0]
    
    lp = PrologFile(os.path.abspath(filename))
    
    print ('======= INITIALIZE DATA ======')
    t = time.time()
    pl = PrologEngine()
    print ('Completed in %.4fs' % (time.time() - t))
    
    print ('========= PARSE DATA =========')
    t = time.time()
    db = ClauseDB.createFrom(lp, builtins=pl.getBuiltIns())
    print ('Completed in %.4fs' % (time.time() - t))
    
    print ('======= LOGIC PROGRAM =======')

    t = time.time()
    print ('\n'.join(map(str,db)))
    print ('Completed in %.4fs' % (time.time() - t))
    print ()
    print ('====== CLAUSE DATABASE ======')
    
    print (db)
    
    print ()
    print ('========== QUERIES ==========')
    t = time.time()
    queries = pl.query(db, Term( 'query', Var('Q') ))
    evidence = pl.query(db, Term( 'evidence', Var('Q'), Var('V') ))
    
    print ('Number of queries:', len(queries))
    print ('Completed in %.4fs' % (time.time() - t))
    
    t = time.time()
    query_nodes = []
    
    if trace :
        pl = PrologEngine(debugger=Debugger(trace=True))
    gp = GroundProgram()
    for query in queries :
        print ("Grounding for query '%s':" % query[0])
        gp, ground = pl.ground(db, query[0], gp)
        print (gp)
        print (ground)
        query_nodes += [ (n,query[0].withArgs(*x)) for n,x in ground ]

    for query in evidence :
        gp, ground = pl.ground(db, query[0], gp)
        print ("Grounding for evidence '%s':" % query[0])
        print (gp)
        print (ground)
        query_nodes += [ (n,query[0].withArgs(*x)) for n,x in ground ]
    
    print ('Completed in %.4fs' % (time.time() - t))
    print ()
    print ('========== GROUND PROGRAM ==========')
    with open(basename + '.dot', 'w') as f :
        f.write(gp.toDot(query_nodes))
    print ('See \'%s.dot\'.' % basename)
    
    print ('========== CNF ==========')
    
    cnf, facts, choices = gp.toCNF()
    
    cnf_file = basename + '.cnf'
    with open(cnf_file, 'w') as f :
        print('\n'.join(cnf), file=f)
    print ('See \'%s\'.' % cnf_file)
    
    print ('========== NNF ==========')
    nnf_file = basename +'.nnf'
    cmd = ['../version2.0/assist/darwin/dsharp', '-Fnnf', nnf_file, '-smoothNNF','-disableAllLits', cnf_file ] # 
    
    subprocess.check_call(cmd)
   
    print (facts)
    qn = dict(query_nodes)
    qns = []
    nnf = GroundProgram()
    with open(nnf_file) as f :
        for line in f :
            line = line.strip().split()
            if line[0] == 'nnf' :
                pass
            elif line[0] == 'L' :
                name = int(line[1])
                probs = (1.0,1.0)
                if name < 0 :
                    prob = facts.get(-name, probs)[1]
                else :
                    prob = facts.get(name, probs)[0]
                if name in qn :
                    prob = str(prob) + '::' + str(qn[name])
                elif -name in qn :
                    prob = str(prob) + '::-' + str(qn[-name])
                # else :
                #     prob = str(prob) + '::fact_' + str(name)
                nnf._addNode( nnf._fact( None, None, prob ) )
                
            elif line[0] == 'A' :
                children = map(lambda x : int(x) + 1 , line[2:])
                nnf._addNode( nnf._conj( tuple(children) ))
            elif line[0] == 'O' :
                children = map(lambda x : int(x) + 1, line[3:])
                nnf._addNode( nnf._disj( tuple(children) ))
            else :
                print ('Unknown line type')

    print (nnf) 
    with open(basename + '.nnf.dot', 'w') as f :
        f.write(nnf.toDot())
    print (choices.values())
    print (query_nodes)
    
                
    
if __name__ == '__main__' :
    main(*sys.argv[1:])
    
    
    