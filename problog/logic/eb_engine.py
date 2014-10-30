from __future__ import print_function

from .basic import Term, Constant
from .program import ClauseDB, PrologString
from .engine import DummyGroundProgram, BaseEngine, GroundProgram, UnknownClause
from .prolog import addPrologBuiltins

""" 
Assumptions
-----------
Assumption 1: range-restricted clauses (after clause evaluation, head is ground)
Assumption 2: functor-free
Assumption 3: conjunction nodes have exactly two children
Assumption 8: no prolog builtins

-- REMOVED: Assumption 4: no OR
-- REMOVED: Assumption 5: no NOT
-- REMOVED: Assumption 7: no probabilistic grounding
-- REMOVED: Assumption 6: no CHOICE


"""

class EventBasedEngine(BaseEngine) :
    
    def __init__(self) :
        BaseEngine.__init__(self)
        self.nodes = {}
    
    def query(self, db, term, level=0) :
        gp = DummyGroundProgram()
        gp, result = self.ground(db, term, gp, level)
        
        return [ y for x,y in result ]        
    
    def ground(self, db, term, gp=None, level=0) :
        db = ClauseDB.createFrom(db, builtins=self.getBuiltIns())
        
        if gp == None :
            gp = GroundProgram()
        
        # tdb = TermDB()
        # args = [ tdb.add(x) for x in term.args ]
        clause_node = db.find(term)
        
        if clause_node == None : return gp, []
        
        call_node = ClauseDB._call( term.functor, range(0,len(term.args)), clause_node )
        
        res = ResultCollector()
        
        self._eval_call(db, gp, None, call_node, term.args, res )
        return gp, res.results
    
    
    def _eval(self, db, gp, node_id, context, parent) :
        node = db.getNode( node_id )
        ntype = type(node).__name__
        
        if node == () :
            raise UnknownClause()
        elif ntype == 'fact' :
            f = self._eval_fact 
        elif ntype == 'choice' :
            f = self._eval_choice
        elif ntype == 'define' :
            f = self._eval_define
        elif ntype == 'clause' :
            f = self._eval_clause
        elif ntype == 'conj' :
            f = self._eval_conj
        elif ntype == 'disj' :
            f = self._eval_disj
        elif ntype == 'call' :
            f = self._eval_call
        elif ntype == 'neg' :
            f = self._eval_neg
        else :
            raise ValueError(ntype)
        f(db, gp, node_id, node, context, parent)
        
    def _eval_fact( self, db, gp, node_id, node, call_args, parent ) :
        trace( node, call_args )
        # Unify fact arguments with call arguments
        result = unify(node.args, call_args)

        # Notify parent
        if result != None :
            parent.newResult( result, ground_node=gp.addFact(node_id, (), node.probability) )    

    def _eval_choice( self, db, gp, node_id, node, call_args, parent ) :
        trace( node, call_args )
        # Unify fact arguments with call arguments
        result = tuple(call_args)
        #result = unify(node.args, call_args)

        # Notify parent
        if result != None :
            origin = (node.group, result)
            ground_node = gp.addChoice(origin, node.choice, node.probability)
            parent.newResult( result, ground_node )    
    
    def _eval_call( self, db, gp, node_id, node, context, parent ) :
        trace( node, context )
        # Extract call arguments from context
        call_args = []
        for call_arg in node.args :
            if type(call_arg) == int :
                call_args.append( context[call_arg] )
            else :
                call_args.append( call_arg )
        # create a context-switching node that extracts the head arguments
        #  from results from the body context
        # output should be send to the given parent
        context_switch = ProcessCallReturn( node.args, context  )
        context_switch.addListener(parent)
        
        if node.defnode < 0 :
            #sub = builtin( engine=self, clausedb=db, args=call_args, tdb=tdb, functor=node.functor, arity=len(node.args), level=level, **extra)
            raise RuntimeError('Builtins are not yet supported')        
        else :
            self._eval( db, gp, node.defnode, call_args, context_switch )
            
    def _eval_clause( self, db, gp, node_id, node, call_args, parent ) :
        try :
            context = [None] * node.varcount
            for head_arg, call_arg in zip(node.args, call_args) :
                if type(head_arg) == int : # head_arg is a variable
                    context[head_arg] = call_arg
                else : # head arg is a constant => make sure it unifies with the call arg
                    unify_value( ( head_arg, call_arg ) )
                    
            # create a context-switching node that extracts the head arguments
            #  from results from the body context
            # output should be send to the given parent
            context_switch = ProcessBodyReturn( node.args  )
            context_switch.addListener(parent)
            
            # evaluate the body, output should be send to the context-switcher
            self._eval( db, gp, node.child, context, context_switch )
        except UnifyError :
            pass    # head and call are not unifiable
            
    def _eval_conj( self, db, gp, node_id, node, context, parent ) :
        # Assumption: node has exactly two children
        child1, child2 = node.children
        
        # Use a link node that evaluates the second child based on input from the first.
        node2 = ProcessLink( self, db, gp, child2, parent )     # context passed through from node1
        self._eval( db, gp, child1, context, node2 )    # evaluate child1 and make it activate node2

    def _eval_disj( self, db, gp, node_id, node, context, parent ) :
        for child in node.children :
            self._eval( db, gp, child, context, parent )    # evaluate child1 and make it activate node2
        parent.complete()

    def _eval_neg(self, db, gp, node_id, node, context, parent) :
        
        process = ProcessNot( gp, context )
        process.addListener(parent)

        self._eval( db, gp, node.child, context, process )
        #process.complete()
    
    def _eval_define( self, db, gp, node_id, node, call_args, parent ) :
        key = (node_id, tuple(call_args))
        pnode = self.nodes.get(key)
        if pnode == None :
            pnode = ProcessDefine( self, db, gp, node_id, node, call_args )
            self.nodes[key] = pnode
            pnode.addListener(parent)
            pnode.execute()
        else :
            pnode.addListener(parent)

class ProcessNode(object) :

    def __init__(self) :
        self.listeners = []
        self.isComplete = False
    
    def notifyListeners(self, result, ground_node) :
        for listener in self.listeners :
            listener.newResult(result, ground_node)
    
    def notifyComplete(self) :
        if not self.isComplete :
            self.isComplete = True
            for listener in self.listeners :
                listener.complete()
        
    def addListener(self, listener) :
        # Add the listener such that it receives future events.
        self.listeners.append(listener)
        
    def complete(self) :
        self.notifyComplete()
        
class ProcessNot(ProcessNode) :
    
    def __init__(self, gp, context) :
        ProcessNode.__init__(self)
        self.context = context
        self.ground_nodes = []
        self.gp = gp
        
    def newResult(self, result, ground_node) :
        print ('NOT_NEW', result, ground_node)
        if ground_node != None :
            self.ground_nodes.append(ground_node)
        
    def complete(self) :
        print ('NOT', self.ground_nodes)
        if self.ground_nodes :
            or_node = self.gp.negate(self.gp.addOrNode( self.ground_nodes ))
            self.notifyListeners(self.context, ground_node=or_node)
        self.notifyComplete()

class ProcessLink(object) :
    
    def __init__(self, engine, db, gp, node_id, parent) :
        self.engine = engine
        self.db = db
        self.gp = gp
        self.node_id = node_id
        self.parent = parent
        
    def newResult(self, result, ground_node) :
        process = ProcessAnd(self.gp, ground_node)
        process.addListener(self.parent)
        self.engine._eval( self.db, self.gp, self.node_id, result, process)
        
    def complete(self) :
        pass
        
class ProcessAnd(ProcessNode) :
    
    def __init__(self, gp, first_node ) :
        ProcessNode.__init__(self)
        self.gp = gp
        self.first_node = first_node
        
    def newResult(self, result, ground_node) :
        and_node = self.gp.addAndNode( (self.first_node, ground_node) )
        self.notifyListeners(result, and_node)
     
class ProcessDefine(ProcessNode) :
    
    def __init__(self, engine, db, gp, node_id, node, args) :
        ProcessNode.__init__(self)
        self.results = {}
        self.engine = engine
        self.db = db
        self.gp = gp
        self.node_id = node_id
        self.node = node
        self.args = args
        
    def notifyListeners(self, result, ground_node) :
        for listener in self.listeners :
            listener.newResult(result, ground_node)
        
    def addListener(self, listener) :
        # Add the listener such that it receives future events.
        ProcessNode.addListener(self, listener)
        
        # If the node was already active, notify listener of past events.
        for result, ground_node in list(self.results.items()) :
            listener.newResult(result, ground_node)
            
        if self.isComplete :
            listener.complete()
        
    def execute(self) :
        # Get the appropriate children
        children = self.node.children.find( self.args )
        # Evaluate the children
        for child in children :
            self.engine._eval( self.db, self.gp, child, self.args, parent=self )        
        
        self.notifyComplete()
    
    def newResult(self, result, ground_node) :
        res = (tuple(result))
        if res in self.results :
            gn = self.results[res]
            if ground_node != gn and gn != 0 :
                gn_node = self.gp.getNode(gn)
                if type(gn_node).__name__ == 'disj' :
                    self.gp._setNode( gn, self.gp._disj(  (ground_node,) + gn_node.children ))
                else :
                    moved_gn = self.gp._addNode( self.gp.getNode(gn) )
                    self.gp._setNode( gn, self.gp._disj( (moved_gn, ground_node) ))
        else :
            self.results[ res ] = ground_node
            self.notifyListeners(result, ground_node)

        
            
class ProcessBodyReturn(ProcessNode) :
    
    def __init__(self, head_args) :
        ProcessNode.__init__(self)
        self.head_args = head_args
                    
    def newResult(self, result, ground_node) :
        output = []
        for head_arg in self.head_args :
            if type(head_arg) == int : # head_arg is a variable
                output.append( result[head_arg] )
            else : # head arg is a constant => make sure it unifies with the call arg
                output.append( head_arg )
        self.notifyListeners(output, ground_node)  
        

class ProcessCallReturn(ProcessNode) :
    
    def __init__(self, call_args, context) :
        ProcessNode.__init__(self)
        self.call_args = call_args
        self.context = context
                    
    def newResult(self, result, ground_node) :
        output = list(self.context)
        for call_arg, res_arg in zip(self.call_args,result) :
            if type(call_arg) == int : # head_arg is a variable
                output[call_arg] = res_arg
            else : # head arg is a constant => make sure it unifies with the call arg
                pass
        self.notifyListeners(output, ground_node)    


#def trace(*args) : print(*args)
def trace(*args) : pass

class UnifyError(Exception) : pass

def unify_value( v12 ) :
    v1, v2 = v12
    if v1 == None :
        return v2
    elif v2 == None :
        return v1
    elif v1 == v2 :
        return v1
    else :
        raise UnifyError()
        
def unify( v1, v2 ) :
    if v1 == None :
        return v2
    elif v2 == None :
        return v1
    else :
        try :
            return tuple( map(unify_value, zip(v1,v2) ) )
        except UnifyError :
            return None



class ResultCollector(object) :
    
    def __init__(self) :
        self.results = []
    
    def newResult( self, result, ground_result ) :
        self.results.append( (ground_result, result  ))
        
    def complete(self) :
        pass

def test1() :
    
    program = """
    parent(erik,katrien). 
    parent(katrien,liese). 
    0.3::parent(erik,anton). 
    0.5::parent(erik,pieter).
    0.2::parent(maria,erik).
    parent(maria,francinne).
    
    married(francinne,erik).
    parent(X,Y) :- married(X,Z), parent(Z,Y).

    ancestor1(X,Y) :- parent(X,Y).
    ancestor1(X,Y) :- parent(Z,Y), ancestor1(X,Z).

    
    ancestor2(X,Y) :- parent(X,Y).
    ancestor2(X,Y) :- ancestor2(X,Z), parent(Z,Y).

    ancestor3(X,Y) :- ancestor3(X,Z), parent(Z,Y).
    ancestor3(X,Y) :- parent(X,Y).
    """
    
    pl = PrologString( program )
    
    db = ClauseDB.createFrom(pl)
    
    eng = EventBasedEngine()

    gp, res = eng.ground( db,  Term('ancestor1',None,None ) )

    print (len(res)) 
    for r in res :
        print (r)
        
    print (gp)

    
    # processor = eng.createProcessor( def_index, res )
    #
    #
    # print ('======')
    # processor = eng.createProcessor( def_index, res )
    # processor.evaluate( (Term('erik'),None) )
    #
    # print ('======')
    #
    # def_index = db._getHead(Term('ancestor',None,None))
    # facts = db.getNode(def_index).children
    #
    # processor = eng.createProcessor( def_index, res )
    # processor.evaluate( (None,Term('liese')) )

def test2() :
    
    program = """
        0.3::stress(X) :- person(X).
        0.2::influences(X,Y) :- person(X), person(Y).

        smokes(X) :- stress(X).
        smokes(X) :- friend(X,Y), influences(Y,X), smokes(Y).

        0.4::asthma(X) <- smokes(X).

        person(1).
        person(2).
        person(3).
        person(4).

        friend(1,2).
        friend(2,1).
        friend(2,4).
        friend(3,2).
        friend(4,2).
    """
    
    pl = PrologString( program )
    
    db = ClauseDB.createFrom(pl)
    
    
    eng = EventBasedEngine()
    res = ResultCollector() 
    
    print ('====')
    
    
    gp, ress = eng.ground( db, Term('smokes',Constant(1)))
    
    print (ress)
    
    print (gp)

def test3() :
    
    program = """
        
        p(X) :- r(X).
        p(X) :- \+ p(X).
        
        q(1).
        q(2).
        
        r(1).

    """
    
    pl = PrologString( program )
    
    db = ClauseDB.createFrom(pl)
    
    def_index = db._getHead(Term('p',None))
    facts = db.getNode(def_index).children
    
    eng = EventBasedEngine()
    res = ResultCollector() 
    
    print ('====')
    
    print ('== 1 ==')
    eng._eval( db, def_index, (Constant(1),), res)
    print ('== 2 ==')
    eng._eval( db, def_index, (Constant(2),), res)
    print ('== 3 ==')
    eng._eval( db, def_index, (Constant(3),), res)

if __name__ == '__main__' :
    #test1()
    
    test2()
    #
    #test3()