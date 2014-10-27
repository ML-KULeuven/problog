from __future__ import print_function

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))


from problog.logic import Term, Var, Clause, And, Or

from problog.logic.program import ClauseDB

from problog.logic.engine import Engine, Debugger

from problog.logic.prolog import PrologEngine

    
def test1() :
    
    db = ClauseDB()
    
    anc1 = Term('anc1')
    anc2 = Term('anc2')
    anc3 = Term('anc3')
    par = Term('par')  
    tst = Term('tst')  
    
    f = Term('f')
    
    X = Var('X')
    Y = Var('Y')
    Z = Var('Z')
    
    erik = Term('erik')
    katrien = Term('katrien')
    liese = Term('liese')
    
    true = Term('true')
    
    db += anc1(X,Y) << ( par(X,Y) | par(X,Z) & anc1(Z,Y) )
    
    db += anc2(X,Y) << ( anc2(X,Z) & par(Z,Y) | par(X,Y) )
    
    db += anc3(X,Y) << ( anc3(X,Z) & par(Z,Y) )
    db += anc3(X,Y) << ( par(X,Y) )
    
    #db.addClause(  )
    
    
    db += par(erik, katrien)
    db += par(katrien, liese)
    
    
    #db.addDef(par(X,Y).term, db.addOr( f1, f2 ))
    #
    print ('===== Compiled =====')
    
    print (db)
    
    import sys
    #sys.exit()
    
    print ('==== Evaluations ====')
    A = Var('A')
    B = Var('B')
    C = Var('C')
   
    
    dbg = Debugger() 
    #dbg = None
   
    print ('Cycle free')
    print (PrologEngine().query( db, anc1(A,B) ))
    
    print ('With cycle - or')
    print (PrologEngine().query( db, anc2(A,B) ))

    print ('With cycle - clause')
    print (PrologEngine(dbg).query( db, anc3(A,B) ))

    
    print ('.\n'.join(map(str,db)) + '.')

def test2() :
    
    eng = PrologEngine()
    
    db = ClauseDB(builtins=eng.getBuiltIns())
    
    tst1 = Term('tst1')  
    tst2 = Term('tst2')  
    _is = Term('is')
    _plus = Term("'+'")
    eq = Term("'='")
    call = Term('call')
        
    X = Var('X')
    Y = Var('Y')
    Z = Var('Z')
        
    true = Term('true')
    fail = Term('fail')
    
    f = Term('f')
    
    #c1 = db.addFact( eq(X,X) )
    db += tst1(X,Y) << eq(Y,f(X))
    #db += tst1(X,Y) << eq(Y,f(X))
    db += tst2(X,f(X)) << true
    
    
    print ('===== Compiled =====')
    
    print (db)
    
    print ('==== Evaluations ====')
    A = Var('A')
    B = Var('B')
    C = Var('C')
       
    print ('Version 1')
    print (eng.query( db, tst1(A,B) ))

    print ('Version 2')
    print (eng.query( db, tst2(A,B) ))  
    
    
def test3() :
    
    eng = PrologEngine()
    db = ClauseDB(builtins=eng.getBuiltIns())
    
    p = Term('p')
    q = Term('q')
    
    X = Var('X')
    Y = Var('Y')
    Z = Var('Z')
    
    a = Term('a')
    b = Term('b')
    
    db += (p(X,Y) << ( ~q(X,Y) ))
    db += q(a,b)
   
    print ('===== Compiled =====')    
    print (db)
    
    print ('==== Evaluations ====')
    
    print ('?- p(a,b)')
    print (eng.query( db, p(a,b) ))

    print ('?- p(b,a)')
    print (eng.query( db, p(b,a) ))

    print ('?- p(b,Y)')
    print (eng.query( db, p(b,Y) ))

    
    print ('?- p(a,X)')
    print (eng.query( db, p(a,X) ))
    
    print ('?- q(a,X)')
    print (eng.query( db, q(a,b) ))

def test4() :
    from problog.logic.program import PrologFile
    
    pl = PrologFile( os.path.join(os.path.dirname(__file__),'family.pl') )
    
    print (PrologEngine().query(pl, Term('ancestor',Var('X'),Var('Y'))))

def test5() :
    
    print ('====== TEST 5 ======')
    from problog.logic.program import PrologFile
    
    pl = PrologFile( os.path.join(os.path.dirname(__file__),'holidays.pl') )

    for cl in pl : print (str(cl))

    db = ClauseDB.createFrom(pl)
    
    print (db)
    
    for cl in db :
        print (cl)
    
if __name__ == '__main__' :
    test1()
    test2()
    test3()
    test4()
    test5()