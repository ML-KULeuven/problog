#! /usr/bin/env python

from __future__ import print_function

import sys, os, time, subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

sys.setrecursionlimit(10000)

from problog.logic import Term, Var, Constant
from problog.logic.program import PrologFile, ClauseDB, PrologString
from problog.logic.engine import DefaultEngine
from problog.logic.formula import LogicFormula, CNF, NNFFile


def ground(model) :
    
    engine = DefaultEngine()
    db = engine.prepare(model)
    
    queries = engine.query(db, Term( 'query', None ))
    evidence = engine.query(db, Term( 'evidence', None, None ))
        
    gp = LogicFormula()
    for query in queries :
        gp = engine.ground(db, query[0], gp, label='query')

    for query in evidence :
        if str(query[1]) == 'true' :
            gp = engine.ground(db, query[0], gp, label='evidence')
        else :
            gp = engine.ground(db, query[0], gp, label='-evidence')

    return gp
    
def convert(gp) :
    
    return CNF(gp)
    
def compile(cnf) :
    
    cnf_file = '/tmp/pl.cnf'
    with open(cnf_file, 'w') as f :
        f.write(cnf.toDimacs())

    nnf_file = '/tmp/pl.nnf'
    cmd = ['../version2.0/assist/darwin/dsharp', '-Fnnf', nnf_file, '-smoothNNF','-disableAllLits', cnf_file ] #

    attempts_left = 10
    success = False
    while attempts_left and not success :
        try :
            subprocess.check_call(cmd)
            success = True
        except subprocess.CalledProcessError as err :
            print (err)
            print ("dsharp crashed, retrying", file=sys.stderr)
            attempts_left -= 1
            if attempts_left == 0 :
                raise err
        
    return NNFFile(nnf_file, cnf.facts, cnf.names)
    
def evaluate(nnf) :
    pass

def main( filename ) :
    
    with open(filename) as f :
        model = f.read()
    
    lp = PrologString(model)
        
    gp = ground(lp)

    gp = gp.makeAcyclic()

    print ('========== GROUND PROGRAM ==========')
    print (gp)
    filename = '/tmp/pl.dot'
    with open(filename, 'w') as f :
        f.write(gp.toDot())
    print ('See \'%s\'.' % filename)
    
    print (gp.getNames())
    
    cnf = convert(gp)
    
    nnf = compile(cnf)
    
    print (nnf)
    
    filename = '/tmp/pl.nnf.dot'
    with open(filename, 'w') as f :
        f.write(nnf.toDot())
    print ('See \'%s\'.' % filename)
    
    print (nnf.getNamesWithLabel())

    # Probability of evidence
    for n_ev, node_ev in nnf.getNames('evidence') :
        print ("Positive evidence:", node_ev, n_ev)
        nnf.setRTrue(node_ev)
    for n_ev, node_ev in nnf.getNames('-evidence') :
        print ("Negative evidence:", node_ev, n_ev)
        nnf.setRFalse(node_ev) 
    prob_evidence = nnf.getProbability( len(nnf) )
    nnf.resetProbabilities()
    
    # Probability of query given evidence
    for name, node in nnf.getNames('query') :
        print ("Query:", node, name)
        nnf.setTrue(node)
        for n_ev, node_ev in nnf.getNames('evidence') :
            print ("Positive evidence:", node_ev, n_ev)
            nnf.setRTrue(node_ev)
        for n_ev, node_ev in nnf.getNames('-evidence') :
            print ("Negative evidence:", node_ev, n_ev)
            nnf.setRFalse(node_ev)            

        print ('==>', name, ':', nnf.getProbability( len(nnf) ) / prob_evidence )    
        nnf.resetProbabilities()

    
    
#
#     basename = os.path.splitext(filename)[0]
#
#     lp = PrologFile(os.path.abspath(filename))
#
#     print ('======= INITIALIZE DATA ======')
#     t = time.time()
#     pl = EventBasedEngine()
#     addBuiltins(pl)
#     print ('Completed in %.4fs' % (time.time() - t))
#
#     print ('========= PARSE DATA =========')
#     t = time.time()
#     db = ClauseDB.createFrom(lp, builtins=pl.getBuiltIns())
#     print ('Completed in %.4fs' % (time.time() - t))
#
#     print ('======= LOGIC PROGRAM =======')
#
#     t = time.time()
#     print ('\n'.join(map(str,db)))
#     print ('Completed in %.4fs' % (time.time() - t))
#     print ()
#     print ('====== CLAUSE DATABASE ======')
#
#     print (db)
#
#
#     print ()
#     print ('========== QUERIES ==========')
#     t = time.time()
#     queries = pl.query(db, Term( 'query', None ))
#     evidence = pl.query(db, Term( 'evidence', None,None ))
#
#     print ('Number of queries:', len(queries))
#     print ('Completed in %.4fs' % (time.time() - t))
#
#     t = time.time()
#     query_nodes = []
#
#     if trace :
#         pl = PrologEngine(debugger=Debugger(trace=True))
#     gp = LogicFormula()
#     for query in queries :
#         print ("Grounding for query '%s':" % query[0])
#         gp, ground = pl.ground(db, query[0], gp)
#         print (gp)
#         print (ground)
#         query_nodes += [ (n,query[0].withArgs(*x)) for n,x in ground ]
#
#     for query in evidence :
#         gp, ground = pl.ground(db, query[0], gp)
#         print ("Grounding for evidence '%s':" % query[0])
#         print (gp)
#         print (ground)
#         query_nodes += [ (n,query[0].withArgs(*x)) for n,x in ground ]
#
#     print ('Completed in %.4fs' % (time.time() - t))
#     print ()
#
#     qn_index, qn_name = zip(*query_nodes)
#     gp, qn_index = gp.makeAcyclic( qn_index )
#     query_nodes = zip(qn_index, qn_name)
#
#
#     #new_query_nodes = gp.breakCycles( query_nodes )
#
#
#     print ('========== GROUND PROGRAM ==========')
#     with open(basename + '.dot', 'w') as f :
#         f.write(gp.toDot(query_nodes))
#     print ('See \'%s.dot\'.' % basename)
#
#
#
#
#     print ('========== CNF ==========')
#
#     cnf, facts = gp.toCNF()
#
#     cnf_file = basename + '.cnf'
#     with open(cnf_file, 'w') as f :
#         print('\n'.join(cnf), file=f)
#     print ('See \'%s\'.' % cnf_file)
#
#     print (facts)
#
#     sys.exit()
#
#
#     print ('========== NNF ==========')
#     nnf_file = basename +'.nnf'
#     cmd = ['../version2.0/assist/darwin/dsharp', '-Fnnf', nnf_file, '-smoothNNF','-disableAllLits', cnf_file ] #
#
#     subprocess.check_call(cmd)
#
#     print (facts)
#     qn = dict(query_nodes)
#     qns = []
#     nnf = GroundProgram()
#     with open(nnf_file) as f :
#         for line in f :
#             line = line.strip().split()
#             if line[0] == 'nnf' :
#                 pass
#             elif line[0] == 'L' :
#                 name = int(line[1])
#                 probs = (1.0,1.0)
#                 if name < 0 :
#                     prob = facts.get(-name, probs)[1]
#                 else :
#                     prob = facts.get(name, probs)[0]
#                 if name in qn :
#                     prob = str(prob) + '::' + str(qn[name])
#                 elif -name in qn :
#                     prob = str(prob) + '::-' + str(qn[-name])
#                 # else :
#                 #     prob = str(prob) + '::fact_' + str(name)
#                 nnf._addNode( nnf._fact( None, None, prob ) )
#
#             elif line[0] == 'A' :
#                 children = map(lambda x : int(x) + 1 , line[2:])
#                 nnf._addNode( nnf._conj( tuple(children) ))
#             elif line[0] == 'O' :
#                 children = map(lambda x : int(x) + 1, line[3:])
#                 nnf._addNode( nnf._disj( tuple(children) ))
#             else :
#                 print ('Unknown line type')
#
#     print (nnf)
#     with open(basename + '.nnf.dot', 'w') as f :
#         f.write(nnf.toDot())
#     print (choices.values())
#     print (query_nodes)
    
                
    
if __name__ == '__main__' :
    main(*sys.argv[1:])
    
    
    