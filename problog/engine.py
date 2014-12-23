from __future__ import print_function

from .program import ClauseDB, PrologString
from .logic import Term, Constant, Var
from .formula import LogicFormula

from collections import defaultdict

"""
Assumptions
-----------
Assumption 1: range-restricted clauses (after clause evaluation, head is ground, ALSO: no unification of two uninstantiated variables)
Assumption 3: conjunction nodes have exactly two children   (not a real restriction)

Assumption 8: no prolog builtins 
    - added some builtins (needs better framework)

-- REMOVED: Assumption 4: no OR
-- REMOVED: Assumption 5: no NOT
-- REMOVED: Assumption 7: no probabilistic grounding
-- REMOVED: Assumption 6: no CHOICE
-- REMOVED: Assumption 2: functor-free


Known issues
------------

Table for query() may make ground() invalid. (Solution: don't reuse engine if you don't want to reuse tables)

"""

class UnifyError(Exception) : pass

def unify_value( v1, v2 ) :
    if v1 == None or type(v1) == int :
        return v2
    elif v2 == None or type(v2) == int :
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


def instantiate( term, context ) :
    if type(term) == int :
        return context[term]
    else :
        return term.apply(context)


class EngineLogger(object) :

    instance = None
    instance_class = None
    
    @classmethod
    def get(self) :
        if EngineLogger.instance == None :
            if EngineLogger.instance_class == None :
                EngineLogger.instance = EngineLogger()
            else :
                EngineLogger.instance = EngineLogger.instance_class()
        return EngineLogger.instance
    
    @classmethod
    def setClass(cls, instance_class) :
        EngineLogger.instance_class = instance_class
        EngineLogger.instance = None
        
    def __init__(self) :
        pass
                
    def receiveResult(self, source, result, node, *extra) :
        pass
        
    def receiveComplete(self, source, *extra) :
        pass
        
    def sendResult(self, source, result, node, *extra) :
        pass

    def sendComplete(self, source, *extra) :
        pass
        
    def create(self, node) :
        pass
        
    def connect(self, source, listener, evt_type) :
        pass
        
class SimpleEngineLogger(EngineLogger) :
        
    def __init__(self) :
        pass
                
    def receiveResult(self, source, result, node, *extra) :
        print (type(source).__name__, id(source), 'receive', result, node, source)
        
    def receiveComplete(self, source, *extra) :
        print (type(source).__name__, id(source), 'receive complete', source)
        
    def sendResult(self, source, result, node, *extra) :
        print (type(source).__name__, id(source), 'send', result, node, source)

    def sendComplete(self, source, *extra) :
        print (type(source).__name__, id(source), 'send complete', source)
        
    def create(self, source) :
        print (type(source).__name__, id(source), 'create', source)
        
    def connect(self, source, listener, evt_type) :
        print (type(source).__name__, id(source), 'connect', type(listener).__name__, id(listener))

class Context(object) :
    
    def __init__(self, lst, define) :
        self.__lst = lst
        self.define = define
        
    def __getitem__(self, index) :
        return self.__lst[index]
        
    def __setitem__(self, index, value) :
        self.__lst[index] = value
        
    def __len__(self) :
        return len(self.__lst)
        
    def __iter__(self) :
        return iter(self.__lst)
        
    def __str__(self) :
        return str(self.__lst)
    

class EventBasedEngine(object) :
    """An event-based ProbLog grounding engine. It supports cyclic programs."""
    
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
        self.__builtin_index[sig] = -(len(self.__builtins) + 1)
        self.__builtins.append( func )
        
    def getBuiltIns(self) :
        return self.__builtin_index
    
    def enter_call(self, node, context) :
        if self.debugger :
            self.debugger.enter(0, node, context)
        
    def exit_call(self, node, context) :
        if self.debugger :
            self.debugger.exit(0, node, context, None)
    
    def create_context(self, lst=[], size=None, define=None) :
        if size != None :
            assert( not lst )
            lst = [None] * size
        return Context(lst, define=define)
    
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
        
        try :
            self._eval_call(db, gp, None, call_node, self.create_context(term.args,define=None), res )
        except RuntimeError as err :
            if str(err).startswith('maximum recursion depth exceeded') :
                raise UnboundProgramError()
            else :
                raise
        
        return gp, res.results
    
    def _eval(self, db, gp, node_id, context, parent) :
        node = db.getNode( node_id )
        ntype = type(node).__name__
        
        assert(isinstance(context,Context))
        
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
            for a,b in zip(node.args, call_args) :
                unify(a, b)
            # Notify parent
            parent.newResult( node.args, ground_node=gp.addAtom(node_id, node.probability) )
        except UnifyError :
            pass
        parent.complete(node_id)    

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
        parent.complete(node_id)
    
    def _eval_call( self, db, gp, node_id, node, context, parent ) :
        trace( node, context )
        # Extract call arguments from context
        # TODO functors???
        
        call_args = [ instantiate(arg, context) for arg in node.args ]
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
                self._eval( db, gp, node.defnode, self.create_context(call_args, define=context.define), context_switch )
            except _UnknownClause :
                sig = '%s/%s' % (node.functor, len(node.args))
                raise UnknownClause(sig)
            
    def _eval_clause( self, db, gp, node_id, node, call_args, parent ) :
        try :
            context = self.create_context(size=node.varcount,define=call_args.define)
            for head_arg, call_arg in zip(node.args, call_args) :
                if type(call_arg) == int : call_arg = None
                unify( call_arg, head_arg, context)                

            # create a context-switching node that extracts the head arguments
            #  from results from the body context
            # output should be send to the given parent
            context_switch = ProcessBodyReturn( node.args, node, node_id )
            context_switch.addListener(parent)
            
            # evaluate the body, output should be send to the context-switcher
            self._eval( db, gp, node.child, context, context_switch )
        except UnifyError :
            parent.complete(None) # head and call are not unifiable
            
    def _eval_conj( self, db, gp, node_id, node, context, parent ) :
        # Assumption: node has exactly two children
        child1, child2 = node.children
        
        # Processor for sending out complete signal.
        # Use a link node that evaluates the second child based on input from the first.
        node2 = ProcessLink( self, db, gp, child2, parent, context.define )     # context passed through from node1
        self._eval( db, gp, child1, context, node2 )    # evaluate child1 and make it activate node2    

    def _eval_disj( self, db, gp, node_id, node, context, parent ) :
        
        process = ProcessOr( node.children, parent )
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
            pnode = ProcessDefine( self, db, gp, node_id, node, call_args, call_args.define )
            def_nodes[key] = pnode
            pnode.addListener(parent)
            pnode.execute()
        else :
            if call_args.define and call_args.define.hasAncestor(pnode) :
                cnode = ProcessDefineCycle(pnode, call_args.define, parent)
            else :
                pnode.addListener(parent)
            

class ProcessNode(object) :
    """Generic class for representing *process nodes*."""
    
    EVT_COMPLETE = 1
    EVT_RESULT = 2
    EVT_ALL = 3
    
    def __init__(self) :
        EngineLogger.get().create(self)
        self.listeners = []
        self.isComplete = False
    
    def notifyListeners(self, result, ground_node=0, source=None) :
        """Send the ``newResult`` event to all the listeners of this node.
            The arguments are used as the arguments of the event.
        """
        EngineLogger.get().sendResult(self, result, ground_node)
        for listener, evttype in self.listeners :
            if evttype & self.EVT_RESULT :
                #print ('SEND', 'result', id(self), '->', id(listener))
                listener.newResult(result, ground_node, source)
    
    def notifyComplete(self, source) :
        """Send the ``complete`` event to all listeners of this node."""
        
        EngineLogger.get().sendComplete(self)
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
        EngineLogger.get().connect(self,listener,eventtype)
        self.listeners.append((listener,eventtype))
        
    def complete(self, source=None) :
        EngineLogger.get().receiveComplete(self)
        self.notifyComplete(source)
        
    def newResult(self, result, ground_node=0, source=None) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        self.notifyListeners(result, ground_node, source)

class ProcessOr(ProcessNode) :

    def __init__(self, count, parent, notify_complete=True) :
        ProcessNode.__init__(self)
        self.parent = parent
        self.count = len(count)
        self.notify_complete = notify_complete
        self.is_complete = False
        if self.count == 0 :
            self.complete(None)
        
    
    def newResult(self, result, ground_node=0, source=None) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        self.parent.newResult(result, ground_node, source)
    
    def complete(self, source) :        
        EngineLogger.get().receiveComplete(self)
        # Assumption: children are well-behaved
        #   -> each child sends out exactly one 'complete' event.
        #   => after 'count' events => all children are complete
        # assert(type(source) == int)
        # if self.is_complete : return
        #
        # if source in self.count :
        #     self.count.remove(source)

        self.count -= 1
        if self.count <= 0 and self.notify_complete :
            self.is_complete = True
            EngineLogger.get().sendComplete(self)
            self.parent.complete(self)
            
class ProcessNot(ProcessNode) :
    
    def __init__(self, gp, context) :
        ProcessNode.__init__(self)
        self.context = context
        self.ground_nodes = []
        self.gp = gp
        
    def newResult(self, result, ground_node=0, source=None) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        if ground_node != None :
            self.ground_nodes.append(ground_node)
        
    def complete(self, source) :
        EngineLogger.get().receiveComplete(self)
        if self.ground_nodes :
            or_node = self.gp.addNot(self.gp.addOr( self.ground_nodes ))
            if or_node != None :
                self.notifyListeners(self.context, ground_node=or_node)
        else :
            self.notifyListeners(self.context, ground_node=0)
        self.notifyComplete(source)

class ProcessLink(object) :
    
    def __init__(self, engine, db, gp, node_id, parent, define) :
        self.engine = engine
        self.db = db
        self.gp = gp
        self.node_id = node_id
        self.parent = parent
        self.define = define
        
    def newResult(self, result, ground_node=0, source=None) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        self.engine.exit_call( self.node_id, result )    
        process = ProcessAnd(self.gp, ground_node)
        process.addListener(self.parent, ProcessNode.EVT_RESULT)
        self.engine._eval( self.db, self.gp, self.node_id, self.engine.create_context(result,define=self.define), process)
        
    def complete(self, source) :
        EngineLogger.get().receiveComplete(self)
        EngineLogger.get().sendComplete(self)
        self.parent.complete(None)
        
class ProcessAnd(ProcessNode) :
    
    def __init__(self, gp, first_node ) :
        ProcessNode.__init__(self)
        self.gp = gp
        self.first_node = first_node
        
    def newResult(self, result, ground_node=0, source=None) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        and_node = self.gp.addAnd( (self.first_node, ground_node) )
        self.notifyListeners(result, and_node, source)
        
    def complete(self, source) :
        EngineLogger.get().receiveComplete(self)
        self.notifyComplete(source)

class ProcessDefineCycle(ProcessNode) :
    
    def __init__(self, parent, context, listener) :
        self.parent = parent
        
        while context != self.parent :
            context.cyclic = True
            context = context.parent

        self.parent.cyclic = True
        self.parent.addCycleChild(self)
        ProcessNode.__init__(self)
        self.addListener(listener)
        self.parent.addListener(self)
    
    def __repr__(self) :
        return 'cycle child of %s' % (self.parent)
        
     
class ProcessDefine(ProcessNode) :
    
    def __init__(self, engine, db, gp, node_id, node, args, parent) :
        self.node = node
        self.results = {}
        self.engine = engine
        self.db = db
        self.gp = gp
        self.node_id = node_id
        self.args = args
        self.parent = parent
        self.__is_cyclic = False
        self.__buffer = defaultdict(list)
        self.children = []
        ProcessNode.__init__(self)
        
    @property
    def cyclic(self) :
        return self.__is_cyclic
        
    @cyclic.setter
    def cyclic(self, value) :
        if self.__is_cyclic != value :
            self.__is_cyclic = value
            self._cycle_detected()
        
    def addCycleChild(self, cnode ) :
        self.children.append(cnode)
        
    def hasAncestor(self, anc) :
        ancestor = self
        while ancestor != None :
            if ancestor == anc : return True
            ancestor = ancestor.parent
        return False
                        
    def addListener(self, listener, eventtype=ProcessNode.EVT_ALL) :
        
        # Add the listener such that it receives future events.
        ProcessNode.addListener(self, listener, eventtype)
        
        # If the node was already active, notify listener of past events.
        for result, ground_node in list(self.results.items()) :
            if eventtype & ProcessNode.EVT_RESULT :
                listener.newResult(result, ground_node)
            
        if self.isComplete :
            if eventtype & ProcessNode.EVT_COMPLETE :
                listener.complete(None)
        
    def execute(self) :
        # Get the appropriate children
        children = self.node.children.find( self.args )
        
        process = ProcessOr( children, self, True)
        # Evaluate the children
        for child in children :
            self.engine._eval( self.db, self.gp, child, self.engine.create_context(self.args,define=self), parent=process )
        
        for c in self.children :
            c.complete()
    
    def newResult(self, result, ground_node=0, source=None) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        if self.cyclic :
            self.newResultUnbuffered(result, ground_node, source)
        else :
            self.newResultBuffered(result, ground_node, source)
    
    def newResultBuffered(self, result, ground_node=0, source=None) :
        res = (tuple(result))
        self.__buffer[res].append( ground_node )
                        
    def newResultUnbuffered(self, result, ground_node=0, source=None) :
        res = (tuple(result))
        if res in self.results :
            res_node = self.results[res]
            self.gp.addDisjunct( res_node, ground_node )
        else :
            self.engine.exit_call( self.node, result )
            result_node = self.gp.addOr( (ground_node,), readonly=False )
            self.results[ res ] = result_node
            
            self.notifyListeners(result, result_node, source )
            
    def complete(self, source) :
        EngineLogger.get().receiveComplete(self)
        self._flush_buffer()
        self.notifyComplete(self.node_id)
    
    def _cycle_detected(self) :
        self._flush_buffer(True)
    
    def _flush_buffer(self, cycle=False) :
        for result, nodes in self.__buffer.items() :
            if len(nodes) > 1 or cycle :
                # Must make an 'or' node
                node = self.gp.addOr( nodes, readonly=(not cycle) )
            else :
                node = nodes[0]
            self.results[result] = node
            self.notifyListeners(result, node, self)
        self.__buffer.clear()
    
        
    def __repr__(self) :
        return '%s %s(%s)' % (id(self), self.node.functor, ', '.join(map(str,self.args)))
        
        
            
class ProcessBodyReturn(ProcessNode) :
    
    def __init__(self, head_args, node, node_id) :
        ProcessNode.__init__(self)
        self.head_args = head_args
        self.node_id = node_id
        self.node = node
                    
    def newResult(self, result, ground_node=0, source=None) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        output = [ instantiate(arg, result) for arg in self.head_args ]
        self.notifyListeners(output, ground_node, source)
    
    def complete(self, source) :
        self.notifyComplete(self.node_id)
            
class ProcessCallReturn(ProcessNode) :
    
    def __init__(self, call_args, context) :
        ProcessNode.__init__(self)
        self.call_args = call_args
        self.context = context
                    
    def newResult(self, result, ground_node=0, source=None) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        output = list(self.context)
        try :
            for call_arg, res_arg in zip(self.call_args,result) :
                unify( res_arg, call_arg, output )
            self.notifyListeners(output, ground_node, source)    
        except UnifyError :
            pass
    

#def trace(*args) : print(*args)
def trace(*args) : pass


    


class ResultCollector(object) :
    
    def __init__(self) :
        self.results = []
    
    def newResult( self, result, ground_result, source=None ) :
        self.results.append( (ground_result, result  ))
        
    def complete(self, source) :
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
    callback.complete(None)    

def builtin_fail( context, callback ) :
    callback.complete(None)

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
        callback.complete(None)

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
        callback.complete(None)
            
def builtin_notsame( A, B, context, callback ) :
    """A \== B"""
    if A == None and B == None :
        raise RuntimeError('Operation not supported!')
    else :
        if A != B :
            callback.newResult( (A,B) )
        callback.complete(None)    

def builtin_same( A, B, context, callback ) :
    """A \== B"""
    if A == None and B == None :
        raise RuntimeError('Operation not supported!')
    else :
        if A == B :
            callback.newResult( (A,B) )
        callback.complete(None)    

def builtin_gt( A, B, context, callback ) :
    """A > B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA > vB) :
        callback.newResult( (A,B) )
    callback.complete(None)

def builtin_lt( A, B, context, callback ) :
    """A > B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA < vB) :
        callback.newResult( (A,B) )
    callback.complete(None)

def builtin_le( A, B, context, callback ) :
    """A =< B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA <= vB) :
        callback.newResult( (A,B) )
    callback.complete(None)

def builtin_ge( A, B, context, callback ) :
    """A >= B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA >= vB) :
        callback.newResult( (A,B) )
    callback.complete(None)

def builtin_val_neq( A, B, context, callback ) :
    """A =/= B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA != vB) :
        callback.newResult( (A,B) )
    callback.complete(None)

def builtin_val_eq( A, B, context, callback ) :
    """A =:= B 
        A and B are ground
    """
    vA = compute(A, context).value
    vB = compute(B, context).value
    
    if (vA == vB) :
        callback.newResult( (A,B) )
    callback.complete(None)

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
    callback.complete(None)
        
def addBuiltins(engine) :
    engine.addBuiltIn('true', 0, builtin_true)
    engine.addBuiltIn('fail', 0, builtin_fail)
    # engine.addBuiltIn('call/1', _builtin_call_1)

    engine.addBuiltIn('=', 2, builtin_eq)
    engine.addBuiltIn('\=', 2, builtin_neq)
    engine.addBuiltIn('==', 2, builtin_same)
    engine.addBuiltIn('\==', 2, builtin_notsame)

    engine.addBuiltIn('is', 2, builtin_is)

    engine.addBuiltIn('>', 2, builtin_gt)
    engine.addBuiltIn('<', 2, builtin_lt)
    engine.addBuiltIn('=<', 2, builtin_le)
    engine.addBuiltIn('>=', 2, builtin_ge)
    engine.addBuiltIn('=\=', 2, builtin_val_neq)
    engine.addBuiltIn('=:=', 2, builtin_val_eq)


DefaultEngine = EventBasedEngine


class UserAbort(Exception) : pass

class UserFail(Exception) : pass

class NonGroundQuery(Exception) : pass

class UnboundProgramError(Exception) : pass


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
