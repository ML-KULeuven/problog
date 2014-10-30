from __future__ import print_function

from .basic import Term, Constant
from .program import ClauseDB, PrologString
from .engine import DummyGroundProgram, BaseEngine, GroundProgram, UnknownClause, _UnknownClause, Debugger

""" 
Assumptions
-----------
Assumption 1: range-restricted clauses (after clause evaluation, head is ground)
Assumption 2: functor-free
    - added support for functors in clause head arguments
Assumption 3: conjunction nodes have exactly two children
Assumption 8: no prolog builtins 
    - added some builtins (needs better framework)

-- REMOVED: Assumption 4: no OR
-- REMOVED: Assumption 5: no NOT
-- REMOVED: Assumption 7: no probabilistic grounding
-- REMOVED: Assumption 6: no CHOICE

Known issues
------------

Table for query() may make ground() invalid.

"""

class UnifyError(Exception) : pass

def unify_value( v1, v2, context1=None, context2=None ) :
    if v1 == None :
        return v2
    elif v2 == None :
        return v1
    elif v1 == v2 :
        # TODO functor
        return v1
    else :
        raise UnifyError()
        
def unify( v1, v2, context1=None, context2=None ) :
    if v1 == None :
        return v2
    elif v2 == None :
        return v1
    else :
        try :
            return tuple( map( lambda xy : unify_value(xy[0],xy[1], context1, context2), zip(v1,v2) ) )
        except UnifyError :
            return None


class PrologInstantiationError(Exception) : pass

class PrologTypeError(Exception) : pass

def computeFunction(func, args, context) :
    if func == "'+'" :
        return Constant(args[0].value + args[1].value)
    elif func == "'-'" :
        return Constant(args[0].value - args[1].value)
    elif func == "'*'" :
        return Constant(args[0].value * args[1].value)
    elif func == "'/'" :
        return Constant(args[0].value / args[1].value)

    else :
        raise ValueError("Unknown function: '%s'" % func)

def compute( value, context ) :
    if type(value) == int :
        return compute(context[value], context)
    elif value == None :
        raise PrologInstantiationError(value)        
    elif value.isConstant() :
        if type(value.value) == str :
            raise PrologTypeError('number', value)
        else :
            return value
    else :
        args = [ compute(arg, context) for arg in value.args ]
        return computeFunction( value.functor, args, context )

def builtin_true( context, callback ) :
    callback.newResult(context)
    callback.complete()    

def builtin_fail( context, callback ) :
    callback.complete()

def builtin_eq( A, B, context, callback ) :
    """A = B
        A and B not both variables
    """
    if A == None and B == None :
        raise RuntimeError('Operation not supported!')
    else :
        try :
            R = unify_value(A,B)
            callback.newResult( ( R, R ) )
        except UnifyError :
            pass
        callback.complete()

def builtin_neq( A, B, context, callback ) :
    """A = B
        A and B not both variables
    """
    if A == None and B == None :
        raise RuntimeError('Operation not supported!')
    else :
        try :
            R = unify_value(A,B)
        except UnifyError :
            callback.newResult( ( A, B ) )
        callback.complete()
            
def builtin_notsame( A, B, context, callback ) :
    """A \== B"""
    if A == None and B == None :
        raise RuntimeError('Operation not supported!')
    else :
        if A != B :
            callback.newResult( (A,B) )
        callback.complete()    

def builtin_same( A, B, context, callback ) :
    """A \== B"""
    if A == None and B == None :
        raise RuntimeError('Operation not supported!')
    else :
        if A == B :
            callback.newResult( (A,B) )
        callback.complete()    

def builtin_gt( A, B, context, callback ) :
    """A > B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA > vB) :
        callback.newResult( (A,B) )
    callback.complete()

def builtin_lt( A, B, context, callback ) :
    """A > B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA < vB) :
        callback.newResult( (A,B) )
    callback.complete()

def builtin_le( A, B, context, callback ) :
    """A =< B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA <= vB) :
        callback.newResult( (A,B) )
    callback.complete()

def builtin_ge( A, B, context, callback ) :
    """A >= B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA >= vB) :
        callback.newResult( (A,B) )
    callback.complete()

def builtin_val_neq( A, B, context, callback ) :
    """A =/= B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA != vB) :
        callback.newResult( (A,B) )
    callback.complete()

def builtin_val_eq( A, B, context, callback ) :
    """A =:= B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA == vB) :
        callback.newResult( (A,B) )
    callback.complete()



def builtin_is( A, B, context, callback ) :
    """A is B
        B is ground
    """
    vB = compute(B, context).value
    try :
        R = Constant(vB)
        unify_value(A,R)
        callback.newResult( (R,B) )
    except UnifyError :
        pass
    callback.complete()
        
def addBuiltins(engine) :
    engine.addBuiltIn('true', 0, builtin_true)
    engine.addBuiltIn('fail', 0, builtin_fail)
    # engine.addBuiltIn('call/1', _builtin_call_1)

    engine.addBuiltIn('=', 2, builtin_eq)
    engine.addBuiltIn('\=', 2, builtin_eq)
    engine.addBuiltIn('==', 2, builtin_same)
    engine.addBuiltIn('\==', 2, builtin_notsame)

    engine.addBuiltIn('is', 2, builtin_is)

    engine.addBuiltIn('>', 2, builtin_gt)
    engine.addBuiltIn('<', 2, builtin_lt)
    engine.addBuiltIn('>', 2, builtin_ge)
    engine.addBuiltIn('=<', 2, builtin_le)
    engine.addBuiltIn('>=', 2, builtin_gt)
    engine.addBuiltIn('=\=', 2, builtin_val_neq)
    engine.addBuiltIn('=:=', 2, builtin_val_eq)

    

class EventBasedEngine(BaseEngine) :
    
    def __init__(self) :
        BaseEngine.__init__(self)
        self.nodes = {}
        
        self.debugger = None
        #self.debugger = Debugger(trace=True)
    
    def query(self, db, term, level=0) :
        gp = DummyGroundProgram()
        gp, result = self.ground(db, term, gp, level)
        
        return [ y for x,y in result ]
        
    def enter_call(self, node, context) :
        if self.debugger :
            self.debugger.enter(0, node, context)
        
    def exit_call(self, node, context) :
        if self.debugger :
            self.debugger.exit(0, node, context, None)
                
    
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
        
        self.enter_call( node, context )
        
        if node == () :
            raise _UnknownClause()
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
        
        self.exit_call( node, context )
        
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

        if type(node.probability) == int :
            probability = call_args[node.probability]
        else :
            probability = node.probability.apply(call_args)

        # Notify parent
        if result != None :
            origin = (node.group, result)
            ground_node = gp.addChoice(origin, node.choice, probability)
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
        #print ('CALL', call_args, node.args, context)        
        
        # create a context-switching node that extracts the head arguments
        #  from results from the body context
        # output should be send to the given parent
        context_switch = ProcessCallReturn( node.args, context  )
        context_switch.addListener(parent)
        
        if node.defnode < 0 :
            #sub = builtin( engine=self, clausedb=db, args=call_args, tdb=tdb, functor=node.functor, arity=len(node.args), level=level, **extra)

            builtin = self._getBuiltIn( node.defnode )
            builtin( *call_args, context=context, callback=context_switch )                
        else :
            try :
                self._eval( db, gp, node.defnode, call_args, context_switch )
            except _UnknownClause :
                sig = '%s/%s' % (node.functor, len(node.args))
                raise UnknownClause(sig)
            
    def _eval_clause( self, db, gp, node_id, node, call_args, parent ) :
        try :
            # context = target context
            # node.args = target values
            # call_args = source values
            
            def unify( source_value, target_value, target_context ) :
                if type(target_value) == int :
                    target_context[target_value] = source_value
                else :
                    assert( isinstance(target_value, Term) )
                    if source_value == None :  # a variable
                        return True # unification successful
                    else :
                        assert( isinstance( source_value, Term ) )
                        if target_value.signature == source_value.signature :
                            for s_arg, t_arg in zip(source_value.args, target_value.args) :
                                unify( s_arg, t_arg, target_context )
                        else :
                            raise UnifyError()
            
            context = [None] * node.varcount
            for head_arg, call_arg in zip(node.args, call_args) :
                unify( call_arg, head_arg, context)                
                # if type(head_arg) == int : # head_arg is a variable
                #     context[head_arg] = call_arg
                # else : # head arg is a constant => make sure it unifies with the call arg
                #     unify_value( head_arg, call_arg )
                    
            # create a context-switching node that extracts the head arguments
            #  from results from the body context
            # output should be send to the given parent
            context_switch = ProcessBodyReturn( node.args  )
            context_switch.addListener(parent)
            
            # evaluate the body, output should be send to the context-switcher
            self._eval( db, gp, node.child, context, context_switch )
        except UnifyError :
            #print ('unification failed', node.args, call_args, context)
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
        
    def newResult(self, result, ground_node=0) :
        if ground_node != None :
            self.ground_nodes.append(ground_node)
        
    def complete(self) :
        if self.ground_nodes :
            or_node = self.gp.negate(self.gp.addOrNode( self.ground_nodes ))
            if or_node != None :
                self.notifyListeners(self.context, ground_node=or_node)
        else :
            self.notifyListeners(self.context, ground_node=0)
        self.notifyComplete()

class ProcessLink(object) :
    
    def __init__(self, engine, db, gp, node_id, parent) :
        self.engine = engine
        self.db = db
        self.gp = gp
        self.node_id = node_id
        self.parent = parent
        
    def newResult(self, result, ground_node=0) :
        self.engine.exit_call( self.node_id, result )    
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
        
    def newResult(self, result, ground_node=0) :
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
    
    def newResult(self, result, ground_node=0) :
        res = (tuple(result))
        if res in self.results :
            self.gp.updateRedirect( self.results[res], ground_node )
        else :
            self.engine.exit_call( self.node, result )
            result_node = self.gp.addRedirect( ground_node )
            self.results[ res ] = result_node
            self.notifyListeners(result, result_node)

        
            
class ProcessBodyReturn(ProcessNode) :
    
    def __init__(self, head_args) :
        ProcessNode.__init__(self)
        self.head_args = head_args
                    
    def newResult(self, result, ground_node=0) :
        output = []
        for head_arg in self.head_args :
            if type(head_arg) == int : # head_arg is a variable
                output.append( result[head_arg] )
            else : # head arg is a constant => make sure it unifies with the call arg
                output.append( head_arg )
        #print ('BODY_RETURN', result, self.head_args, output, result)        
        self.notifyListeners(output, ground_node)  
        

class ProcessCallReturn(ProcessNode) :
    
    def __init__(self, call_args, context) :
        ProcessNode.__init__(self)
        self.call_args = call_args
        self.context = context
                    
    def newResult(self, result, ground_node=0) :
        output = list(self.context)
        for call_arg, res_arg in zip(self.call_args,result) :
            if type(call_arg) == int : # head_arg is a variable
                output[call_arg] = res_arg
            else : # head arg is a constant => make sure it unifies with the call arg
                pass
        #print ('CALL_RETURN', self.context, self.call_args, output, result)        
        self.notifyListeners(output, ground_node)    


#def trace(*args) : print(*args)
def trace(*args) : pass


    


class ResultCollector(object) :
    
    def __init__(self) :
        self.results = []
    
    def newResult( self, result, ground_result ) :
        self.results.append( (ground_result, result  ))
        
    def complete(self) :
        pass

try:
    input = raw_input
except NameError:
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