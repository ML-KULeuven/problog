from __future__ import print_function

from .program import ClauseDB, PrologString
from .logic import Term, Constant, Var
from .formula import LogicFormula

"""
Assumptions
-----------
Assumption 1: range-restricted clauses (after clause evaluation, head is ground)
Assumption 2: functor-free
    - added support for functors in clause head arguments
    - still missing:
            - functors in facts -> OK
            - functors in calls -> OK
            - unify in body return?
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

def unify_value( v1, v2 ) :
    if v1 == None :
        return v2
    elif v2 == None :
        return v1
    elif v1 == v2 :
        # TODO functor
        return v1
    else :
        raise UnifyError()
        
def unify_simple( v1, v2 ) :
    if v1 == None :
        return v2
    elif v2 == None :
        return v1
    else :
        try :
            return tuple( map( lambda xy : unify_value(xy[0],xy[1]), zip(v1,v2) ) )
        except UnifyError :
            return None

def unify( source_value, target_value, target_context=None ) :
    if type(target_value) == int :
        if target_context != None :
            current_value = target_context[target_value]
            if current_value == None :
                target_context[target_value] = source_value
            else :
                unify( current_value, source_value )
    elif target_value == None :
        pass
    else :
        assert( isinstance(target_value, Term) )
        if source_value == None :  # a variable
            return True # unification successful
        else :
            # print (source_value, type(source_value))
            # print (target_value, type(target_value))
            assert( isinstance( source_value, Term ) )
            if target_value.signature == source_value.signature :
                for s_arg, t_arg in zip(source_value.args, target_value.args) :
                    unify( s_arg, t_arg, target_context )
            else :
                raise UnifyError()


class _UnknownClause(Exception) : pass

class UnknownClause(Exception) : pass


class EventBasedEngine(object) :
    
    def __init__(self, builtins=True, debugger=None) :
        self.__builtin_index = {}
        self.__builtins = []
        
        if builtins :
            addBuiltins(self)
        
        self.debugger = debugger
        
    def _getBuiltIn(self, index) :
        real_index = -(index + 1)
        return self.__builtins[real_index]
        
    def addBuiltIn(self, pred, arity, func) :
        sig = '%s/%s' % (pred, arity)
        self.__builtin_index[sig] = -(len(self.__builtin_index) + 1)
        self.__builtins.append( func )
        
    def getBuiltIns(self) :
        return self.__builtin_index
    
    def enter_call(self, node, context) :
        if self.debugger :
            self.debugger.enter(0, node, context)
        
    def exit_call(self, node, context) :
        if self.debugger :
            self.debugger.exit(0, node, context, None)
    
    def query(self, db, term, level=0) :
        gp = LogicFormula()
        gp, result = self._ground(db, term, gp, level)
        return [ y for x,y in result ]
            
    def prepare(self, db) :
        return ClauseDB.createFrom(db, builtins=self.getBuiltIns())
    
    def ground(self, db, term, gp=None, label=None) :
        #self.debugger = Debugger(trace=True)
        gp, results = self._ground(db, term, gp)
        
        for node_id, args in results :
            gp.addName( term.withArgs(*args), node_id, label )
        if not results :
            gp.addName( term, None, label )
        
        # print ('Ground result')
        # print (gp)
        return gp
    
    def _ground(self, db, term, gp=None, level=0) :
        db = self.prepare(db)
        # print ('Ground program', db)

        if gp == None : gp = LogicFormula()
        
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
        
        try :
            for a,b in zip(call_args, node.args) :
                unify(a, b)
            # Notify parent
            parent.newResult( node.args, ground_node=gp.addAtom(node_id, node.probability) )
        except UnifyError :
            #print ('FACT unify', node.args, call_args)
            pass
        parent.complete()    

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
            ground_node = gp.addAtom( (node.group, result, node.choice) , probability, group=(node.group, result) ) 
            parent.newResult( result, ground_node )
        parent.complete()
    
    def _eval_call( self, db, gp, node_id, node, context, parent ) :
        trace( node, context )
        # Extract call arguments from context
        # TODO functors???
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
            context_switch = ProcessBodyReturn( node.args, node  )
            context_switch.addListener(parent)
            
            # evaluate the body, output should be send to the context-switcher
            self._eval( db, gp, node.child, context, context_switch )
        except UnifyError :
            #print ('unification failed', node.args, call_args, context)
            pass    # head and call are not unifiable
            
    def _eval_conj( self, db, gp, node_id, node, context, parent ) :
        # Assumption: node has exactly two children
        child1, child2 = node.children
        
        # Processor for sending out complete signal.
        process_complete = ProcessCompleteAll( len(node.children), parent )
        
        # Use a link node that evaluates the second child based on input from the first.
        node2 = ProcessLink( self, db, gp, child2, process_complete, parent )     # context passed through from node1
        self._eval( db, gp, child1, context, node2 )    # evaluate child1 and make it activate node2    

    def _eval_disj( self, db, gp, node_id, node, context, parent ) :
        
        process = ProcessOr( len(node.children), parent )
        for child in node.children :
            self._eval( db, gp, child, context, process )    # evaluate child1 and make it activate node2

    def _eval_neg(self, db, gp, node_id, node, context, parent) :
        process = ProcessNot( gp, context )
        process.addListener(parent)
        self._eval( db, gp, node.child, context, process )
    
    def _eval_define( self, db, gp, node_id, node, call_args, parent ) :
        key = (node_id, tuple(call_args))
        
        if not hasattr(gp, '_def_nodes') :
            gp._def_nodes = {}
        def_nodes = gp._def_nodes
        
        pnode = def_nodes.get(key)
        if pnode == None :
            pnode = ProcessDefine( self, db, gp, node_id, node, call_args )
            def_nodes[key] = pnode
            pnode.addListener(parent)
            pnode.execute()
        else :
            pnode.addListener(parent)
            
class ProcessCompleteAll( object ) :
    
    def __init__(self, count, parent) :
        self.count = count
        self.parent = parent
        
    def newResult(self, result, ground_node=0, source=None) :
        pass
        
    def complete(self, source=None) :
        # Assumption: children are well-behaved
        #   -> each child sends out exactly one 'complete' event.
        #   => after 'count' events => all children are complete
        self.count -= 1
        if self.count == 0 :
            self.parent.complete()

class ProcessNode(object) :
    """Generic class for representing *process nodes*."""
    
    EVT_COMPLETE = 1
    EVT_RESULT = 2
    EVT_ALL = 3
    
    def __init__(self) :
        #print ('CREATE', id(self), type(self))
        self.listeners = []
        self.isComplete = False
    
    def notifyListeners(self, result, ground_node=0, source=None) :
        """Send the ``newResult`` event to all the listeners of this node.
            The arguments are used as the arguments of the event.
        """
        for listener, evttype in self.listeners :
            if evttype & self.EVT_RESULT :
                #print ('SEND', 'result', id(self), '->', id(listener))
                listener.newResult(result, ground_node, source)
    
    def notifyComplete(self, source=None) :
        """Send the ``complete`` event to all listeners of this node."""
        
        if not self.isComplete :
            # print ('COMPLETE', self)
            self.isComplete = True
            for listener, evttype in self.listeners :
                if evttype & self.EVT_COMPLETE :
                    #print ('SEND', 'complete', id(self), '->', id(listener))
                    listener.complete(source)
        
    def addListener(self, listener, eventtype=EVT_ALL) :
        """Add the given listener."""
        # Add the listener such that it receives future events.
        self.listeners.append((listener,eventtype))
        
    def complete(self, source=None) :
        self.notifyComplete()
        
    def newResult(self, result, ground_node=0, source=None) :
        raise NotImplementedError('ProcessNode.newResult is an abstract method!')

class ProcessOr(object) :

    def __init__(self, count, parent) :
        self.parent = parent
        self.count = count
    
    def newResult(self, result, ground_node=0, source=None) :
        self.parent.newResult(result, ground_node, source)
    
    def complete(self, source=None) :
        
        #print ('OR complete', self.count, self.parent)
        
        # Assumption: children are well-behaved
        #   -> each child sends out exactly one 'complete' event.
        #   => after 'count' events => all children are complete
        self.count -= 1
        if self.count == 0 :
            self.parent.complete()
            
class ProcessNot(ProcessNode) :
    
    def __init__(self, gp, context) :
        ProcessNode.__init__(self)
        self.context = context
        self.ground_nodes = []
        self.gp = gp
        
    def newResult(self, result, ground_node=0, source=None) :
        #print('NOT RECEIVES:', result, ground_node)
        if ground_node != None :
            self.ground_nodes.append(ground_node)
        
    def complete(self, source=None) :
        #print('NOT COMPLETE', self.ground_nodes)
        if self.ground_nodes :
            or_node = self.gp.addNot(self.gp.addOr( self.ground_nodes ))
            if or_node != None :
                self.notifyListeners(self.context, ground_node=or_node)
        else :
            self.notifyListeners(self.context, ground_node=0)
        self.notifyComplete()

class ProcessLink(object) :
    
    def __init__(self, engine, db, gp, node_id, andc, parent) :
        self.engine = engine
        self.db = db
        self.gp = gp
        self.node_id = node_id
        self.parent = parent
        self.andc = andc
        
    def newResult(self, result, ground_node=0, source=None) :
        self.engine.exit_call( self.node_id, result )    
        process = ProcessAnd(self.gp, ground_node)
        process.addListener(self.parent, ProcessNode.EVT_RESULT)
        process.addListener(self.andc, ProcessNode.EVT_COMPLETE)
        self.engine._eval( self.db, self.gp, self.node_id, result, process)
        
    def complete(self, source=None) :
        self.andc.complete()
        
class ProcessAnd(ProcessNode) :
    
    def __init__(self, gp, first_node ) :
        ProcessNode.__init__(self)
        self.gp = gp
        self.first_node = first_node
        
    def newResult(self, result, ground_node=0, source=None) :
        and_node = self.gp.addAnd( (self.first_node, ground_node) )
        self.notifyListeners(result, and_node, source)
        
    def complete(self, source=None) :
        self.notifyComplete()
     
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
                
    def addListener(self, listener, eventtype=ProcessNode.EVT_ALL) :
        
        # Add the listener such that it receives future events.
        ProcessNode.addListener(self, listener, eventtype)
        
        # If the node was already active, notify listener of past events.
        for result, ground_node in list(self.results.items()) :
            if eventtype & ProcessNode.EVT_RESULT :
                listener.newResult(result, ground_node)
            
        if self.isComplete :
            if eventtype & ProcessNode.EVT_COMPLETE :
                listener.complete()
        
    def execute(self) :
        # Get the appropriate children
        children = self.node.children.find( self.args )
        
        process = ProcessOr( len(children), self )
        # Evaluate the children
        for child in children :
            self.engine._eval( self.db, self.gp, child, self.args, parent=process )
        
        self.notifyComplete()
    
    def newResult(self, result, ground_node=0, source=None) :
        res = (tuple(result))
        #debug = (self.node.functor in ('stress','smokes'))
        if res in self.results :
            res_node = self.results[res]
            # Also matches siblings: p::have_one(1). have_two(X,Y) :- have_one(X), have_one(X). query(have_two(1,1)). 
            # if self.gp._deref(res_node) == self.gp._deref(ground_node) : print ('CYCLE!!?!?!?!?!?!? =>', res_node)
            #if debug: print ('\t UPDATE NODE:', res_node)     
            self.gp.addDisjunct( res_node, ground_node )
        else :
            self.engine.exit_call( self.node, result )
            result_node = self.gp.addOr( (ground_node,), readonly=False )
            self.results[ res ] = result_node
            
            #if debug: print ('\t STORED AS:', result_node)        
            self.notifyListeners(result, result_node, source )
            
    def complete(self, source=None) :
        self.notifyComplete()
        
    def __repr__(self) :
        return str(self.node_id) + ': ' + str(self.node) + str(id(self))
        
        
            
class ProcessBodyReturn(ProcessNode) :
    
    def __init__(self, head_args, node) :
        ProcessNode.__init__(self)
        self.head_args = head_args
        self.node = node
                    
    def newResult(self, result, ground_node=0, source=None) :
        output = []
        for head_arg in self.head_args :
            if type(head_arg) == int : # head_arg is a variable
                output.append( result[head_arg] )
            else : # head arg is a constant => make sure it unifies with the call arg
                output.append( head_arg )
        #print ('BODY_RETURN', result, self.head_args, output, result)        
        self.notifyListeners(output, ground_node, source)
    
    # def complete(self) :
    #     pass
        
    def __repr__(self) :
        return str(self.node)

class ProcessCallReturn(ProcessNode) :
    
    def __init__(self, call_args, context) :
        ProcessNode.__init__(self)
        self.call_args = call_args
        self.context = context
                    
    def newResult(self, result, ground_node=0, source=None) :
        output = list(self.context)
        try :
            for call_arg, res_arg in zip(self.call_args,result) :
#                unify( res_arg, call_arg, output )


                if type(call_arg) == int : # head_arg is a variable
                    output[call_arg] = res_arg
                else : # head arg is a constant => make sure it unifies with the call arg
                    pass
            #print ('CALL_RETURN', self.context, self.call_args, output, result)        
            self.notifyListeners(output, ground_node, source)    
        except UnifyError :
            pass
            #print ('CALL unify', result, self.call_args)
    

#def trace(*args) : print(*args)
def trace(*args) : pass


    


class ResultCollector(object) :
    
    def __init__(self) :
        self.results = []
    
    def newResult( self, result, ground_result, source=None ) :
        self.results.append( (ground_result, result  ))
        
    def complete(self, source=None) :
        pass

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
        return Constant(float(args[0].value) / float(args[1].value))

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


DefaultEngine = EventBasedEngine


class UserAbort(Exception) : pass

class UserFail(Exception) : pass

class NonGroundQuery(Exception) : pass


# Input python 2 and 3 compatible input
try:
    input = raw_input
except NameError:
    pass
        
class Debugger(object) :
        
    def __init__(self, debug=True, trace=False) :
        self.__debug = debug
        self.__trace = trace
        self.__trace_level = None
        
    def enter(self, level, node_id, call_args) :
        if self.__trace :
            print ('  ' * level, '>', node_id, call_args, end='')
            self._trace(level)  
        elif self.__debug :
            print ('  ' * level, '>', node_id, call_args)
        
    def exit(self, level, node_id, call_args, result) :
        
        if not self.__trace and level == self.__trace_level :
            self.__trace = True
            self.__trace_level = None
            
        if self.__trace :
            if result == 'USER' :
                print ('  ' * level, '<', node_id, call_args, result)
            else :
                print ('  ' * level, '<', node_id, call_args, result, end='')
                self._trace(level, False)
        elif self.__debug :
            print ('  ' * level, '<', node_id, call_args, result)
    
    def _trace(self, level, call=True) :
        try : 
            cmd = input('? ')
            if cmd == '' or cmd == 'c' :
                pass    # Do nothing special
            elif cmd.lower() == 's' :
                if call :
                    self.__trace = False
                    if cmd == 's' : self.__debug = False                
                    self.__trace_level = level
            elif cmd.lower() == 'u' :
                self.__trace = False
                if cmd == 'u' : self.__debug = False
                self.__trace_level = level - 1
            elif cmd.lower() == 'l' :
                self.__trace = False
                if cmd == 'l' : self.__debug = False
                self.__trace_level = None
            elif cmd.lower() == 'a' :
                raise UserAbort()
            elif cmd.lower() == 'f' :
                if call :
                    raise UserFail()
            else : # help
                prefix = '  ' * (level) + '    '
                print (prefix, 'Available commands:')
                print (prefix, '\tc\tcreep' )
                print (prefix, '\ts\tskip     \tS\tskip (with debug)' )
                print (prefix, '\tu\tgo up    \tU\tgo up (with debug)' )
                print (prefix, '\tl\tleap     \tL\tleap (with debug)' )
                print (prefix, '\ta\tabort' )
                print (prefix, '\tf\tfail')
                print (prefix, end='')
                self._trace(level,call)
        except EOFError :
            raise UserAbort()
