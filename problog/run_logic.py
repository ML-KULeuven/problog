from __future__ import print_function

from logic.basic import Term, Var

from logic.program import Clause, And, Or, Lit, ClauseDB

from logic.engine import Engine, Debugger

from logic.prolog import PrologEngine

debug_call = print


    

    
def test1() :
    
    db = ClauseDB()
    
    anc1 = Lit.create('anc1')
    anc2 = Lit.create('anc2')
    par = Lit.create('par')  
    tst = Lit.create('tst')  
    
    f = Term.create('f')
    
    X = Var('X')
    Y = Var('Y')
    Z = Var('Z')
    
    erik = Term('erik')
    katrien = Term('katrien')
    liese = Term('liese')
    
    true = Lit('true')
    
    db += anc1(X,Y) << ( par(X,Y) | par(X,Z) & anc1(Z,Y) )
    
    db += anc2(X,Y) << ( anc2(X,Z) & par(Z,Y) | par(X,Y) )
    
    
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
    print (PrologEngine(dbg).query( db, anc1(A,B) ))
    
    print ('With cycle')
    print (PrologEngine().query( db, anc2(A,B) ))
    
    print ('.\n'.join(map(str,db)) + '.')

def test2() :
    
    db = ClauseDB()
    
    tst1 = Lit.create('tst1')  
    tst2 = Lit.create('tst2')  
    _is = Lit.create('is')
    _plus = Term.create('+')
    eq = Lit.create('=')
    call = Lit.create('call')
        
    X = Var('X')
    Y = Var('Y')
    Z = Var('Z')
        
    true = Lit('true')
    fail = Lit('fail')
    
    f = Term.create('f')
    
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
    print (PrologEngine().query( db, tst1(A,B) ))

    print ('Version 2')
    print (PrologEngine().query( db, tst2(A,B) ))  
    
    
def test3() :
    
    db = ClauseDB()
    
    p = Lit.create('p')
    q = Lit.create('q')
    
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
    print (PrologEngine().query( db, p(a,b) ))

    print ('?- p(b,a)')
    print (PrologEngine().query( db, p(b,a) ))

    print ('?- p(b,Y)')
    print (PrologEngine().query( db, p(b,Y) ))

    
    print ('?- p(a,X)')
    print (PrologEngine().query( db, p(a,X) ))
    
    print ('?- q(a,X)')
    print (PrologEngine().query( db, q(a,b) ))


    
if __name__ == '__main__' :
    test1()
    test2()
    test3()