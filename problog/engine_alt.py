from __future__ import print_function
from collections import defaultdict 

import sys, os

from .formula import LogicFormula
from .program import ClauseDB, PrologFile
from .logic import Term
from .engine import unify, UnifyError, instantiate, extract_vars, is_ground, UnknownClause, _UnknownClause
from .engine_builtins import addStandardBuiltIns, check_mode, GroundingError

# TODO fix bug:
#  Detect fake cycles. The active define node is not always an ancestor of the cycle child.
#   In that case, the cycle finder message ('o') reaches the top level and the child is not registered at its parent.
#


# New engine: notable differences
#  - keeps its own call stack -> no problem with Python's maximal recursion depth 
#  - should support skipping calls / tail-recursion optimization

# TODO:
#  - make clause and call skippable (only transform results) -> tail recursion optimization
#
# DONE:
#  - add choice node (annotated disjunctions) 
#  - add builtins
#  - add caching of define nodes
#  - add cycle handling
#  - add interface
#  - process directives

class NegativeCycle(GroundingError) : 
    """The engine does not support negative cycles."""
    
    def __init__(self, location=None) :
        self.location = location
        
        msg = 'Negative cycle detected'
        if self.location : msg += ' at position %s:%s' % self.location
        msg += '.'
        GroundingError.__init__(self, msg)
        
class InvalidEngineState(GroundingError): pass

class StackBasedEngine(object) :
    
    def __init__(self, builtins=True) :
        self.stack = []
        
        self.__builtin_index = {}
        self.__builtins = []
        
        if builtins :
            addBuiltIns(self)
        
        self.node_types = {}
        self.node_types['fact'] = EvalFact
        self.node_types['conj'] = EvalAnd
        self.node_types['disj'] = EvalOr
        self.node_types['neg'] = EvalNot
        self.node_types['define'] = EvalDefine
        self.node_types['call'] = EvalCall
        self.node_types['clause'] = EvalClause
        self.node_types['choice'] = EvalChoice
        self.node_types['builtin'] = EvalBuiltIn
        
        self.cycle_root = None
    
    def prepare(self, db) :
        """Convert given logic program to suitable format for this engine."""
        result = ClauseDB.createFrom(db, builtins=self.getBuiltIns())
        self._process_directives( result )
        return result

    def _process_directives( self, db) :
        term = Term('_directive')
        directive_node = db.find( term )
        if directive_node == None : return True    # no directives
        # Create a new call.
        
        node = db.getNode(directive_node)        
        gp = LogicFormula()
        directives = db.getNode(directive_node).children
        
        results = []
        while directives :
            current = directives.pop(0)
            results += self.execute(current, database=db, target=gp, context=[])
    
    def _getBuiltIn(self, index) :
        real_index = -(index + 1)
        return self.__builtins[real_index]
        
    def addBuiltIn(self, pred, arity, func) :
        """Add a builtin."""
        sig = '%s/%s' % (pred, arity)
        self.__builtin_index[sig] = -(len(self.__builtins) + 1)
        self.__builtins.append( func )
        
    def getBuiltIns(self) :
        """Get the list of builtins."""
        return self.__builtin_index
        
    def create_node_type(self, node_type) :
        try :
            return self.node_types[node_type]
        except KeyError :
            raise _UnknownClause()
    
    def create(self, node_id, database, **kwdargs ) :
        if node_id < 0 :
            node_type = 'builtin'
            node = self._getBuiltIn(node_id)
        else :
            node = database.getNode(node_id)
            node_type = type(node).__name__
        
        exec_node = self.create_node_type( node_type )
        
        pointer = len(self.stack) 
        self.stack.append(exec_node(pointer=pointer, engine=self, database=database, node_id=node_id, node=node, **kwdargs))
        return pointer
    
    def query(self, db, term, level=0) :
        """Perform a non-probabilistic query."""
        gp = LogicFormula()
        gp, result = self._ground(db, term, gp, level)
        return [ x for x,y in result ]
        
    def ground(self, db, term, gp=None, label=None, trace=None, debug=None) :
        """Ground a query on the given database.
        
        :param db: logic program
        :type db: LogicProgram
        :param term: query term
        :type term: Term
        :param gp: output data structure (for incremental grounding)
        :type gp: LogicFormula
        :param label: type of query (e.g. ``query``, ``evidence`` or ``-evidence``)
        :type label: str
        """
        gp, results = self._ground(db, term, gp, silent_fail=False, allow_vars=False, trace=trace, debug=debug)
        
        for args, node_id in results :
            gp.addName( term.withArgs(*args), node_id, label )
        if not results :
            gp.addName( term, None, label )
        
        return gp
    
    def _ground(self, db, term, gp=None, level=0, silent_fail=True, allow_vars=True, trace=None, debug=None) :
        # Convert logic program if needed.
        db = self.prepare(db)
        # Create a new target datastructure if none was given.
        if gp == None : gp = LogicFormula()
        # Find the define node for the given query term.
        clause_node = db.find(term)
        # If term not defined: fail query (no error)    # TODO add error to make it consistent?
        if clause_node == None :
            # Could be builtin?
            clause_node = db._getBuiltIn(term.signature)
        if clause_node == None : 
            if silent_fail :
                return gp, []
            else :
                raise UnknownClause(term.signature, location=db.lineno(term.location))
                
        # return self.execute( node_id, database=database, target=target, context=query.args, **kwdargs)
        # eng.execute( query_node, database=db, target=target, context=[None] )
        results = self.execute( clause_node, database=db, target=gp, context=list(term.args), trace=trace, debug=debug)
        
        return gp, results
                
    def isAncestor( self, parent, child ) :
        
        # TODO this is not entirely correct: 
        #  the negation check should happen only up to the common ancestor of the two
        # Question: could we stop when we reach a node that is already marked cyclic or cycleroot?
        current = self.stack[child]
        while current and current.pointer != parent and current.parents[0] != None :
            if type(current.node).__name__ == 'neg' :
                raise NegativeCycle(location=current.database.lineno(current.node.location))
            current = self.stack[current.parents[0]]
        return current.pointer == parent 
        
    def checkState(self, actions) :
        for act, obj, args, kwargs in actions :
            if act == 'e' :
                ps = kwargs['parents']
                for p in ps :
                    if p != None and (p >= len(self.stack) or self.stack[p] == None) :
                        print (act, obj, args, kwargs)
                        raise Exception('Invalid state: referencing non-existing object. [parent]')
            elif act == 'r' and obj != None :                
                obj = self.stack[obj]
                if obj.__class__.__name__ == 'EvalDefine'  :
                    for p in obj.cycle_children :
                        if p != None and (p >= len(self.stack) or self.stack[p] == None) :
                            print (p,obj,obj.pointer)
                            raise Exception('Invalid state: referencing non-existing obj [child]')
                
                
    def execute(self, node_id, **kwdargs ) :
        trace = kwdargs.get('trace')
        debug = kwdargs.get('debug') or trace
        
        target = kwdargs['target']
        if not hasattr(target, '_cache') : target._cache = DefineCache()
        
        pointer = self.create( node_id, parents=[None], **kwdargs )
        cleanUp, actions = self.stack[pointer]()
        actions = list(reversed(actions))
        max_stack = len(self.stack)
        solutions = []
        while actions :
            #self.checkState(actions)
            act, obj, args, kwdargs = actions.pop(-1)
            if obj == None :
                if act == 'r' :
                    solutions.append( (args[0], args[1]) )
                    if args[3] :    # Last result received
                        return solutions
                elif act == 'c' :
                    if debug : print ('Maximal stack size:', max_stack)
                    # if self.active_cycles :
                    #     self.printStack()
                    #     print ('Active cycles:', self.active_cycles)
                    #     raise InvalidEngineState('The engine did not complete successfully!')
                    return solutions
                elif act == 'o' :
                    raise InvalidEngineState('Unexpected state: cycle detected at top-level.')
                    
                    exec_node = self.stack[args[1]]
                    cleanUp, next_actions = exec_node.complete()
                    actions += list(reversed(next_actions))
                    if cleanUp : self.cleanUp(args[1])
                else :
                    raise InvalidEngineState('Unknown message!')
            else:
                if act == 'e' :
                    try :
                        obj = self.create( obj, *args, **kwdargs )
                    except _UnknownClause : 
                        # TODO set right parameters
                        raise UnknownClause('signature', location=None)
                    exec_node = self.stack[obj]
                    cleanUp, next_actions = exec_node()
                elif act == 'C' :
                    if args[0] == True and self.cycle_root and obj == self.cycle_root.pointer :
                        exec_node = self.stack[obj]
                        cleanUp, next_actions = exec_node.closeCycle(*args,**kwdargs)
                    elif args[0] == False :
                        exec_node = self.stack[obj]
                        cleanUp, next_actions = exec_node.closeCycle(*args,**kwdargs)
                else:
                    if obj >= len(self.stack) :
                        self.printStack()
                        print (act, obj)
                        raise InvalidEngineState('Non-existing pointer')
                    exec_node = self.stack[obj]
                    if exec_node == None :
                        print (act, obj)
                        raise InvalidEngineState('Invalid node at given pointer')
                    if act == 'r' :
                        cleanUp, next_actions = exec_node.newResult(*args,**kwdargs)
                    elif act == 'c' :
                        cleanUp, next_actions = exec_node.complete(*args,**kwdargs)
                    elif act == 'o' :
                        cleanUp, next_actions = exec_node.createCycle(*args,**kwdargs)
                    else :
                        raise InvalidEngineState('Unknown message')
                actions += list(reversed(next_actions))
                if debug :
                    # if type(exec_node).__name__ in ('EvalDefine',) :
                    self.printStack(obj)
                    if act in 'rco' : print (obj, act, args)
                    print ( [ (a,o,x) for a,o,x,t in actions ])
                if len(self.stack) > max_stack : max_stack = len(self.stack)
                if trace : sys.stdin.readline()
                if cleanUp :
                    self.cleanUp(obj)
            # if not actions and self.active_cycles :
            #     # Engine stalled -> reactivate it by completing a cycle
            #     cycle = self.active_cycles.pop(-1)
            #     if debug : print ('Activating cycle', cycle)
            #     exec_node = self.stack[cycle]
            #     cleanUp, next_actions = exec_node.complete()
            #     actions += list(reversed(next_actions))
            #     if cleanUp : self.cleanUp(cycle)
                        
        self.printStack()
        print ('Collected results:', solutions)
        raise InvalidEngineState('Engine did not complete correctly!')
    
    def cleanUp(self, obj) :
        self.stack[obj] = None
        while self.stack and self.stack[-1] == None :
            self.stack.pop(-1)
        
        
    def call(self, query, database, target, **kwdargs ) :
        node_id = database.find(query)
        return self.execute( node_id, database=database, target=target, context=query.args, **kwdargs) 
                
    def printStack(self, pointer=None) :
        print ('===========================')
        for i,x in enumerate(self.stack) :
            if i == pointer :
                print ('>>> %s: %s' % (i,x) )
            elif self.cycle_root != None and i == self.cycle_root.pointer  :
                print ('ccc %s: %s' % (i,x) )
            else :
                print ('    %s: %s' % (i,x) )        
        

NODE_TRUE = 0
NODE_FALSE = None

def call(obj, args, kwdargs) :
    return ( 'e', obj, args, kwdargs ) 

def newResult(obj, result, ground_node, source, is_last) :
    return ( 'r', obj, (result, ground_node, source, is_last), {} )

def complete(obj, source) :
    return ( 'c', obj, (source,), {} )
    
def cyclic(obj, child) :
    return ( 'o', obj, (child,), {} )


class EvalNode(object):

    def __init__(self, engine, database, target, node_id, node, context, parents, pointer, identifier=None, transform=None, call=None, **extra ) :
        self.engine = engine
        self.database = database
        self.target = target
        self.node_id = node_id
        self.node = node
        self.context = context
        self.parents = parents
        self.identifier = identifier
        self.pointer = pointer
        self.transform = transform
        self.call = call
        
    def notifyResult(self, arguments, node=0, is_last=False, parents=None ) :
        if type(arguments) != tuple : raise Exception()
        if parents == None : parents = self.parents
        if self.transform == None :
            return [ newResult( parent, arguments, node, self.identifier, is_last ) for parent in parents ]
        else :
            return [ newResult( parent, self.transform(arguments), node, self.identifier, is_last ) for parent in parents ]
        
    def notifyComplete(self, parents=None) :
        if parents == None : parents = self.parents
        return [ complete( parent, self.identifier ) for parent in parents  ]
        
    def notifyCycle(self, child, parents=None) :
        if parents == None : parents = self.parents
        return [ cyclic( parent, child ) for parent in parents ]
        
    def createCall(self, node_id, *args, **kwdargs) :
        base_args = {}
        base_args['database'] = self.database
        base_args['target'] = self.target 
        base_args['context'] = self.context
        base_args['parents'] = [ self.pointer ]
        base_args['identifier'] = self.identifier
        base_args['transform'] = self.transform
        base_args['call'] = self.call
        base_args.update(kwdargs)
        return call( node_id, args, base_args )
        
    def createCycle(self,child) :
        return False, self.notifyCycle( child )
        
    def __repr__(self) :
        return '%s: %s %s [%s] %s {%s}' % (self.__class__.__name__, self.node, self.context, self.call, self.parents, self.pointer)
        

class EvalFact(EvalNode) :
    # Has exactly one listener.
    # Behaviour:
    # - call returns 0 or 1 'newResult' and 1 'complete'
    # - can always be cleaned up immediately
    # - does not support 'newResult' and 'complete'
    
    def __call__(self) :
        actions = []
        try :
            # Verify that fact arguments unify with call arguments.
            for a,b in zip(self.node.args, self.context) :
                unify(a, b)
            # Successful unification: notify parent callback.
            target_node = self.target.addAtom(self.node_id, self.node.probability)
            actions += self.notifyResult( self.node.args, target_node, True )
        except UnifyError :
            # Failed unification: don't send result.
            # Send complete message.
            actions += self.notifyComplete()
        return True, actions        # Clean up, actions
        
class EvalChoice(EvalNode) :
    # Has exactly one listener.
    # Behaviour:
    # - call returns 0 or 1 'newResult' and 1 'complete'
    # - can always be cleaned up immediately
    # - does not support 'newResult' and 'complete'
    
    def __call__(self) :
        actions = []
        
        result = tuple(self.context)
        
        if not is_ground(*result) : raise NonGroundProbabilisticClause(location=self.database.lineno(self.node.location))
        
        probability = instantiate( self.node.probability, result )
        # Create a new atom in ground program.
        origin = (self.node.group, result)
        ground_node = self.target.addAtom( origin + (self.node.choice,) , probability, group=origin ) 
        # Notify parent.
        
        return True, self.notifyResult(result, ground_node, True)

class EvalOr(EvalNode) :
    # Has exactly one listener (parent)
    # Has C children.
    # Behaviour:
    # - 'call' creates child nodes and request calls to them
    # - 'call' calls complete if there are no children (C = 0)
    # - 'newResult' is forwarded to parent
    # - 'complete' waits until it is called C times, then sends signal to parent
    # Can be cleanup after 'complete' was sent
    
    def __init__(self, **parent_args ) : 
        EvalNode.__init__(self, **parent_args)
        
        self.on_cycle = False
        self.__buffer = defaultdict(list)
        self.results = None
        
    def __call__(self) :
        children = self.node.children
        self.to_complete = len(children)
        if self.to_complete == 0 :
            # No children, so complete immediately.
            return True, self.notifyComplete()
        else :
            return False, [ self.createCall( child ) for child in children ]
    
    def isOnCycle(self) :
        return self.on_cycle
    
    def flushBuffer(self, cycle=False) :
        if self.results == None :
            self.results = {}
            for result, nodes in self.__buffer.items() :
                if len(nodes) > 1 or cycle :
                    # Must make an 'or' node
                    node = self.target.addOr( nodes, readonly=(not cycle) )
                else :
                    node = nodes[0]
                self.results[result] = node
            self.__buffer.clear()
            
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
        if self.isOnCycle() :
            res = (tuple(result))
            if res in self.results :
                res_node = self.results[res]
                self.target.addDisjunct( res_node, node )
                actions = []
            else :
                result_node = self.target.addOr( (node,), readonly=False )
                self.results[ res ] = result_node
                actions = []
                if self.isOnCycle() : actions += self.notifyResult(res, result_node)
                if is_last : 
                    a, act = self.complete(source)
                    actions += act
            return False, actions
        else :
            res = (tuple(result))
            self.__buffer[res].append( node )
            if is_last :
                return self.complete(source)
            else :
                return False, []
    
    def complete(self, source=None) :
        self.to_complete -= 1
        if self.to_complete == 0:
            self.flushBuffer()
            actions = []
            if not self.isOnCycle() : 
                for result, node in self.results.items() :
                    actions += self.notifyResult(result,node)
            actions += self.notifyComplete()
            return True, actions
        # elif self.to_complete == len(self.cycle_children) :
        #     return False, self.notifyComplete(parents=self.cycle_children)
        else :
            return False, []
            
    def createCycle(self, child) :
        self.on_cycle = True
        self.flushBuffer(True)
        actions = []
        for result, node in self.results.items() :
            actions += self.notifyResult(result,node) 
        return False, actions + self.notifyCycle(child)


class DefineCache(object) : 
    
    # After a node is finished:
    #   - store it in cache (part of ground program)
    #   - also make subgoals available, e.g. after compute p(X)
    #        we have p(1), p(2), p(3), ...
    #       p(X) -> [(1,), (2,), (3,), (4,), (5,) ... ]
    #       p(1) -> [ 1, 3, 6, 10 ]
    #   - also reversed? -> harder
    #
    
    def __init__(self) :
        self.__non_ground = {}
        self.__ground = {}
        self.__active = {}
    
    def activate(self, goal, node) :
        self.__active[goal] = node
        
    def deactivate(self, goal) :
        del self.__active[goal]
        
    def is_active(self, goal) :
        return self.getEvalNode(goal) != None
        
    def printEvalNode(self) :
        print ('EVALNODES:',self.__active)
        
    def getEvalNode(self, goal) :
        return self.__active.get(goal)
        
    def __setitem__(self, goal, results) :
        # Results
        functor, args = goal
        if is_ground(*args) :
            if results :
                assert(len(results) == 1)
                res_key = next(iter(results.keys()))
                key = (functor, res_key)
                self.__ground[ key ] = results[res_key]
            else :
                key = (functor, args)
                self.__ground[ key ] = NODE_FALSE  # Goal failed
        else :
            res_keys = list(results.keys())
            self.__non_ground[ goal ] = res_keys
            for res_key in res_keys :
                key = (functor, res_key)
                self.__ground[ key ] = results[res_key]
                
    def __getitem__(self, goal) :
        functor, args = goal
        if is_ground(*args) :
            return { args : self.__ground[goal] }
        else :
            res_keys = self.__non_ground[goal]
            result = {}
            for res_key in res_keys :
                result[res_key] = self.__ground[(functor,res_key)]
            return result
            
    def __contains__(self, goal) :
        functor, args = goal
        if is_ground(*args) :
            return goal in self.__ground
        else :
            return goal in self.__non_ground
            
    def __str__(self) :
        return '%s\n%s' % (self.__non_ground, self.__ground)
        
class EvalDefine(EvalNode) :
    
    
    # A buffered Define node.
    def __init__(self, call=None, **parent_args ) : 
        EvalNode.__init__(self, **parent_args)
        self.__buffer = defaultdict(list)
        self.results = None
        
        self.subcycles = set()
        self.cycle_children = []
        self.cycle_close = []
        self.on_cycle = False
        self.is_cycle_root = False
        self.is_cycle_child = False
        
        self.call = ( self.node.functor, tuple(self.context) )
        self.to_complete = None
    
    
    def __call__(self) :
        goal = (self.node.functor, tuple(self.context))
        if self.target._cache.is_active(goal) :
            return False, self.cycleDetected()
        elif goal in self.target._cache :
            results = self.target._cache[goal]
            actions = []
            for result, node in results.items() :
                if node != NODE_FALSE :
                    actions += self.notifyResult(result, node )
            actions += self.notifyComplete()
            return True, actions
        else :
            children = self.node.children.find( self.context )
            self.to_complete = len(children)
            
            if self.to_complete == 0 :
                # No children, so complete immediately.
                return True, self.notifyComplete()
            # elif len(children) == 1 :     # This case skips caching
            #     return True, [ self.createCall( child, parents=self.parents ) for child in children ] 
            else :
                self.target._cache.activate(goal, self)
                actions = [ self.createCall( child) for child in children ]
                return False,  actions + [ ('C', self.pointer, (True,), {} ) ]
    
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
        if self.is_cycle_child :
            if is_last :
                return True, self.notifyResult(result, node, is_last=is_last)
            else :
                return False, self.notifyResult(result, node, is_last=is_last)
        else :
            if self.isOnCycle() or self.isCycleRoot() :
                res = (tuple(result))
                if res in self.results :
                    res_node = self.results[res]
                    self.target.addDisjunct( res_node, node )
                    actions = []
                else :
                    result_node = self.target.addOr( (node,), readonly=False )
                    self.results[ res ] = result_node
                    actions = []
                    # Send results to cycle children
                    actions += self.notifyResult(res, result_node, parents=self.cycle_children)
                    if self.isOnCycle() : actions += self.notifyResult(res, result_node)
                    if is_last : 
                        a, act = self.complete(source)
                        actions += act
                return False, actions
            else :
                res = (tuple(result))
                self.__buffer[res].append( node )
                if is_last :
                    return self.complete(source)
                else :
                    return False, []
    
    def complete(self, source=None) :
        if self.is_cycle_child :
            return True, self.notifyComplete()
        else :
            self.to_complete -= 1
            if self.to_complete == 0:
                cache_key = (self.node.functor, tuple(self.context))
                self.flushBuffer()
                self.target._cache[ cache_key ] = self.results
                self.target._cache.deactivate(cache_key)
                actions = []
                if not self.isOnCycle() : 
                    for result, node in self.results.items() :
                        actions += self.notifyResult(result,node)
                actions += self.notifyComplete()
                return True, actions
            # elif self.to_complete == len(self.cycle_children) :
            #     return False, self.notifyComplete(parents=self.cycle_children)
            else :
                return False, []
            
    def flushBuffer(self, cycle=False) :
        if self.results == None :
            self.results = {}
            for result, nodes in self.__buffer.items() :
                if len(nodes) > 1 or cycle :
                    # Must make an 'or' node
                    node = self.target.addOr( nodes, readonly=(not cycle) )
                else :
                    node = nodes[0]
                self.results[result] = node
            self.__buffer.clear()
        
    def isOnCycle(self) :
        return self.on_cycle
        
    def isCycleRoot(self) :
        return bool(self.cycle_children)
            
    def cycleDetected(self) :
        queue = []
        goal = (self.node.functor, tuple(self.context))
        # Get the top node of this cycle.
        cycle_parent = self.target._cache.getEvalNode(goal)
        cycle_root = self.engine.cycle_root
        # Mark this node as a cycle child
        self.is_cycle_child = True
        # Register this node as a cycle child of cycle_parent
        cycle_parent.cycle_children.append(self.pointer)
        
        cycle_parent.flushBuffer(True)
        for result, node in cycle_parent.results.items() :
            queue += self.notifyResult(result,node,parents=[self.pointer]) 
        
        if cycle_root != None and cycle_parent.pointer < cycle_root.pointer :
            # New parent is earlier in call stack as current cycle root
            # Unset current root
            # Unmark current cycle root
            cycle_root.is_cycle_root = False
            cycle_root.on_cycle = True
            # Copy subcycle information from old root to new root
            cycle_parent.subcycles = cycle_root.subcycles
            cycle_root.subcycles = set()
            
            cycle_parent.cycle_close = cycle_root.cycle_close
            cycle_root.cycle_close = []
            
            cycle_parent.subcycles.add(cycle_root.pointer)
            cycle_root = None        
        if cycle_root == None : # No active cycle
            # Register cycle root with engine
            self.engine.cycle_root = cycle_parent
            # Mark parent as cycle root
            cycle_parent.is_cycle_root = True
            #
            cycle_parent.cycle_close.append(self.pointer)
            # Queue a create cycle message that will be passed up the call stack.
            queue += self.notifyCycle(self.pointer)
            # Send a close cycle message to the cycle root.
        else :  # A cycle is already active
            # The new one is a subcycle
            cycle_root.cycle_close.append(self.pointer)
            
            if cycle_root.pointer != cycle_parent.pointer :
                cycle_root.subcycles.add(cycle_parent.pointer)
            # Queue a create cycle message that will be passed up the call stack.
            queue += self.notifyCycle(self.pointer)
        return queue
    
    def closeCycle(self, toplevel) :
        if self.is_cycle_root and toplevel :
            self.engine.cycle_root = None
            return False, self.notifyComplete(parents=self.cycle_close)
            
            
            #
            # # Forward message to subclauses and complete cycle children
            # return False, [ ('C', sub, (False,), {}) for sub in sorted(self.subcycles) ] + self.notifyComplete(parents=self.cycle_children)
        # elif not toplevel :
        #     return False, self.notifyComplete(parents=self.cycle_children)
        # else :
        else :
            return False, []
    
    def createCycle(self, child) :
        if self.on_cycle :  # Already on cycle
            # Don't do anything
            return False, []
        elif self.is_cycle_root :
            # self.flushBuffer(True)
            # actions = []
            # for result, node in self.results.items() :
            #     actions += self.notifyResult(result,node, parents=[child])
            return False, []
        else :
            # Define node
            self.on_cycle = True
            self.flushBuffer(True)
            actions = []
            for result, node in self.results.items() :
                actions += self.notifyResult(result,node) 
            return False, actions + self.notifyCycle(child)

    def __repr__(self) :
         return EvalNode.__repr__(self) + ' CC:' + str(self.cycle_children) + ' CS:' + str(self.subcycles) + ' ' + str(self.to_complete) + ' %s%s%s' % (self.is_cycle_child,self.isCycleRoot(),self.isOnCycle())


class EvalNot(EvalNode) :
    # Has exactly one listener (parent)
    # Has 1 child.
    # Behaviour:
    # - 'newResult' stores results and does not request actions
    # - 'complete: sends out newResults and complete signals
    # Can be cleanup after 'complete' was sent
    
    def __init__(self, **parent_args ) : 
        EvalNode.__init__(self, **parent_args)
        self.nodes = set()  # Store ground nodes

    def __call__(self) :
        return False, [ self.createCall( self.node.child ) ]
        
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
        if node != NODE_FALSE :
            self.nodes.add( node )
        if is_last :
            return self.complete(source)
        else :
            return False, []
    
    def complete(self, source=None) :
        actions = []
        if self.nodes :
            or_node = self.target.addNot(self.target.addOr( self.nodes ))
            if or_node != NODE_FALSE :
                actions += self.notifyResult(tuple(self.context), or_node)
        else :
            actions += self.notifyResult(tuple(self.context), NODE_TRUE)
        actions += self.notifyComplete()
        return True, actions
        
    def createCycle(self,child) :
        raise NegativeCycle(location=self.database.lineno(self.node.location))
    
        
class EvalAnd(EvalNode) :
    
    def __init__(self, **parent_args) :
        EvalNode.__init__(self, **parent_args)
        self.to_complete = 1
        
    def __call__(self) :
        # Create a node for child 1 and call it.
        return False, [ self.createCall(  self.node.children[0], identifier=None ) ]
        
    def newResult(self, result, node=0, source=None, is_last=False ) :
        # Different depending on whether result comes from child 1 or child 2.
        if source != None : # Child 2 -> pass to parent
            target_node = self.target.addAnd( (source, node) )
            if is_last : 
                a, acts = self.complete()
            else :
                a = False
                acts = []
            return a, self.notifyResult(result, target_node) + acts
        else :  # Child 1 -> create child 2 and call it
            if not is_last : 
                self.to_complete += 1
                return False, [ self.createCall( self.node.children[1], context=result, identifier=node ) ]
            else :
                if node == NODE_TRUE :
                    # TODO Doesn't work if node != 0! Forgets the first node in probabilistic programs!
                    return True, [ self.createCall( self.node.children[1], context=result, parents=self.parents ) ]
                else :
                    return False, [ self.createCall( self.node.children[1], context=result, identifier=node ) ]
            
    def complete(self, source=None) :
        self.to_complete -= 1
        if self.to_complete == 0 :
            return True, self.notifyComplete()
        else :
            assert(self.to_complete > 0)
            return False, []
    
class EvalCall(EvalNode) :
    
    def __init__(self, **parent_args ) : 
        EvalNode.__init__(self, **parent_args)
        
    def __call__(self) :
        call_args = [ instantiate(arg, self.context) for arg in self.node.args ]
        return False, [ self.createCall( self.node.defnode, context=call_args ) ]
    
    def getResultTransform(self) :
        context = list(self.context)
        node_args = list(self.node.args)
        def result_transform(result) :
            output = context
            actions = []
            try :
                assert(len(result) == len(node_args))
                for call_arg, res_arg in zip(node_args,result) :
                    unify( res_arg, call_arg, output )
                return tuple(output)
            except UnifyError :
                pass
        return result_transform
            
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
        result = self.getResultTransform()( result )
        if result == None :
            if is_last :
                return self.complete(source)
            else :
                return False, []
        else :
            return is_last, self.notifyResult(result, node, is_last)
        
    def complete(self, source=None) :
        return True, self.notifyComplete()
    
class EvalBuiltIn(EvalNode) : 
    
    def __call__(self) :
        return self.node(*self.context, engine=self.engine, database=self.database, target=self.target, callback=self)

class EvalClause(EvalNode) :
    
    def __init__(self, **parent_args ) : 
        EvalNode.__init__(self, **parent_args)
        self.head_vars = extract_vars(*self.node.args)  # For variable unification check
        
    def __call__(self) :
        context = [None]*self.node.varcount
        # self._create_context(size=node.varcount,define=call_args.define)
        try :
            assert(len(self.node.args) == len(self.context))
            # Fill in the context by unifying clause head arguments with call arguments.
            for head_arg, call_arg in zip(self.node.args, self.context) :
                # Remove variable identifiers from calling context.
                if type(call_arg) == int : call_arg = None
                # Unify argument and update context (raises UnifyError if not possible)
                unify( call_arg, head_arg, context)
                #
            return False, [ self.createCall( self.node.child, context=context) ]
        except UnifyError :
            # Call and clause head are not unifiable, just fail (complete without results).
            return True, self.notifyComplete()
    
    def getResultTransform(self) :
        location = self.node.location
        node_args = list(self.node.args)
        def result_transform(result) :
            for i, res in enumerate(result) :
                if not is_ground(res) and self.head_vars[i] > 1 :
                    raise VariableUnification(location=location)
            output = [ instantiate(arg, result) for arg in node_args ]
            return tuple(output)
            
        return result_transform
        
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
        result = self.getResultTransform()( result )
        if result == None :
            return self.complete(source)
        else :
            return is_last, self.notifyResult(result, node, is_last)
        
    def complete(self, source=None) :
        return True, self.notifyComplete()



class BooleanBuiltIn(object) :
    """Simple builtin that consist of a check without unification. (e.g. var(X), integer(X), ... )."""
    
    def __init__(self, base_function) :
        self.base_function = base_function
    
    def __call__( self, *args, **kwdargs ) :
        callback = kwdargs.get('callback')
        if self.base_function(*args, **kwdargs) :
            return True, callback.notifyResult(args,NODE_TRUE,True)
        else :
            return True, callback.notifyComplete()
        
class SimpleBuiltIn(object) :
    """Simple builtin that does cannot be involved in a cycle or require engine information and has 0 or more results."""

    def __init__(self, base_function) :
        self.base_function = base_function
    
    def __call__(self, *args, **kwdargs ) :
        callback = kwdargs.get('callback')
        results = self.base_function(*args, **kwdargs)
        output = []
        if results :
            for i,result in enumerate(results) :
                output += callback.notifyResult(result,NODE_TRUE,i==len(results)-1)
            return True, output
        else :
            return True, callback.notifyComplete()

def atom_to_filename(atom) :
    atom = str(atom)
    if atom[0] == atom[-1] == "'" :
        atom = atom[1:-1]
    return atom
    

def builtin_consult_as_list( op1, op2, **kwdargs ) :
    check_mode( (op1,op2), ['*L'], functor='consult', **kwdargs )
    builtin_consult(op1, **kwdargs)
    if is_list_nonempty(op2) :
        builtin_consult_as_list(op2.args[0], op2.args[1], **kwdargs)
    
    
def builtin_consult( arg, callback=None, database=None, engine=None, context=None, location=None, **kwdargs ) :
    check_mode( (arg,), 'a', functor='consult' )
    filename = os.path.join(database.source_root, atom_to_filename( arg ))
    if not os.path.exists( filename ) :
        filename += '.pl'
    if not os.path.exists( filename ) :
        # TODO better exception
        raise ConsultError(location=database.lineno(location), message="Consult: file not found '%s'" % filename)
    
    # Prevent loading the same file twice
    if not filename in database.source_files : 
        database.source_files.append(filename)
        pl = PrologFile( filename )
        for clause in pl :
            database += clause
    return True, callback.notifyResult((arg,), is_last=True)



class CallProcessNode(object) :
    
    def __init__(self, term, args, parent) :
        self.term = term
        self.num_args = len(args)
        self.parent = parent
    
    def newResult(self, result, ground_node=0) :
        if self.num_args > 0 :
            res1 = result[:-self.num_args]
            res2 = result[-self.num_args:]
        else :
            res1 = result
            res2 = []
        self.parent.newResult( [self.term(*res1)] + list(res2), ground_node )

    def complete(self) :
        self.parent.complete()


def builtin_call( term, args=(), engine=None, callback=None, **kwdargs ) :
    # TODO does not support cycle through call!
    check_mode( (term,), 'c', functor='call' )
    # Find the define node for the given query term.
    term_call = term.withArgs( *(term.args + args ))
    results = engine.call( term_call, **kwdargs )
    
    actions = []
    n = len(term.args)
    for res, node in results :
        res1 = res[:n]
        res2 = res[n:]
        res_pass = (term.withArgs(*res1),) + res2
        actions += callback.notifyResult( res_pass, node)
    actions += callback.notifyComplete()
    return True, actions

def builtin_callN( term, *args, **kwdargs ) :
    return builtin_call(term, args, **kwdargs)

def addBuiltIns(engine) :
    
    addStandardBuiltIns(engine, BooleanBuiltIn, SimpleBuiltIn )
    
    #These are special builtins
    engine.addBuiltIn('call', 1, builtin_call)
    for i in range(2,10) :
        engine.addBuiltIn('call', i, builtin_callN)
    engine.addBuiltIn('consult', 1, builtin_consult)
    engine.addBuiltIn('.', 2, builtin_consult_as_list)



