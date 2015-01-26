from __future__ import print_function
from collections import defaultdict 

import sys, os

from .formula import LogicFormula
from .program import ClauseDB, PrologFile
from .logic import Term
from .core import LABEL_NAMED
from .engine import unify, UnifyError, instantiate, extract_vars, is_ground, UnknownClause, _UnknownClause
from .engine_builtins import addStandardBuiltIns, check_mode, GroundingError, NonGroundProbabilisticClause


# New engine: notable differences
#  - keeps its own call stack -> no problem with Python's maximal recursion depth 
#  - should support skipping calls / tail-recursion optimization

# TODO:
#  - make clause skippable (only transform results) -> tail recursion optimization
#
# DONE:
#  - add choice node (annotated disjunctions) 
#  - add builtins
#  - add caching of define nodes
#  - add cycle handling
#  - add interface
#  - process directives
#  - make call skippable

class NegativeCycle(GroundingError) : 
    """The engine does not support negative cycles."""
    
    def __init__(self, location=None) :
        self.location = location
        
        msg = 'Negative cycle detected'
        if self.location : msg += ' at position %s:%s' % self.location
        msg += '.'
        GroundingError.__init__(self, msg)
        
        self.trace = False
        self.debug = False
        
class InvalidEngineState(GroundingError): pass

class StackBasedEngine(object) :
    
    def __init__(self, builtins=True) :
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
        self.pointer = 0
        self.stack_size = 128
        self.stack = [None] * self.stack_size
        
        self.stats = [0,0,0,0]
    
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
        
        pointer = self.pointer
        record = self.create_record(exec_node, pointer, database, node_id, node, **kwdargs)
        self.add_record(record)
        return pointer
        
    def create_record(self, exec_node, pointer, database, node_id, node, **kwdargs ) :
        return exec_node(pointer=pointer, engine=self, database=database, node_id=node_id, node=node, **kwdargs)
        
    def grow_stack(self) :
        self.stack += [None] * self.stack_size
        self.stack_size = self.stack_size * 2
            
    def add_record(self, record) :
        if self.pointer >= self.stack_size : self.grow_stack()
        self.stack[self.pointer] = record
        self.pointer += 1
    
    def query(self, db, term, level=0, **kwdargs) :
        """Perform a non-probabilistic query."""
        gp = LogicFormula()
        gp, result = self._ground(db, term, gp, level, **kwdargs)
        return [ x for x,y in result ]
        
    def ground(self, db, term, gp=None, label=None, **kwdargs) :
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
        gp, results = self._ground(db, term, gp, silent_fail=False, allow_vars=False, **kwdargs)
        
        for args, node_id in results :
            gp.addName( term.withArgs(*args), node_id, label )
        if not results :
            gp.addName( term, None, label )
        
        return gp
    
    def _ground(self, db, term, gp=None, level=0, silent_fail=True, allow_vars=True, **kwdargs) :
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
        results = self.execute( clause_node, database=db, target=gp, context=list(term.args), **kwdargs)
        
        return gp, results
                
    def isAncestor( self, parent, child ) :
        
        # TODO this is not entirely correct: 
        #  the negation check should happen only up to the common ancestor of the two
        # Question: could we stop when we reach a node that is already marked cyclic or cycleroot?
        current = self.stack[child]
        while current and current.pointer != parent and current.parent != None :
            if type(current.node).__name__ == 'neg' :
                raise NegativeCycle(location=current.database.lineno(current.node.location))
            current = self.stack[current.parent]
        return current.pointer == parent
        
    def notifyCycle(self, child) :
        # Optimization: we can usually stop when we reach a node on_cycle.
        #   However, when we swap the cycle root we need to also notify the old cycle root up to the new cycle root.
        assert(self.cycle_root != None)
        root = self.cycle_root.pointer
        childnode = self.stack[child]
        current = childnode.parent
        actions = []
        while current != root :
            exec_node = self.stack[current]
            if exec_node.on_cycle : 
                break
            new_actions = exec_node.createCycle()
            actions += new_actions
            current = exec_node.parent
            self.stats[0] += 1
        return actions
        
    def execute(self, node_id, **kwdargs ) :
        trace = kwdargs.get('trace')
        debug = kwdargs.get('debug') or trace
        stats = kwdargs.get('stats')    # Should support stats[i] += 1 with i=0..5
        self.trace = trace
        self.debug = debug
        
        target = kwdargs['target']
        database = kwdargs['database']
        if not hasattr(target, '_cache') : target._cache = DefineCache()
        n = 6
        pointer = self.create( node_id, parent=None, **kwdargs )
        cleanUp, actions = self.stack[pointer]()
        actions = list(reversed(actions))
        max_stack = len(self.stack)
        solutions = []
        while actions :
            act, obj, args, kwdargs = actions.pop(-1)
            if obj == None :
                if act == 'r' :
                    solutions.append( (args[0], args[1]) )
                    if stats != None : stats[0] += 1
                    if args[3] :    # Last result received
                        return solutions
                elif act == 'c' :
                    if stats != None : stats[1] += 1
                    if debug : print ('Maximal stack size:', max_stack)
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
                    if self.cycle_root != None and kwdargs['parent'] < self.cycle_root.pointer :
                        cleanUp = False
                        next_actions = self.cycle_root.closeCycle(True) + [ (act,obj,args,kwdargs) ]
                    else :
                        if stats != None : stats[2] += 1
                        try :
                            obj = self.create( obj, *args, **kwdargs )
                        except _UnknownClause : 
                            # TODO set right parameters
                            call_origin = kwdargs.get('call_origin')
                            if call_origin == None :
                                sig = 'unknown'
                                raise UnknownClause(sig, location=None)
                            else :
                                loc = database.lineno(call_origin[1])
                                raise UnknownClause(call_origin[0], location=loc)
                        exec_node = self.stack[obj]
                        cleanUp, next_actions = exec_node()
                else:
                    try :
                        exec_node = self.stack[obj]
                    except IndexError :
                        self.printStack()
                        raise InvalidEngineState('Non-existing pointer: %s' % obj )
                    if exec_node == None :
                        print (act, obj)
                        raise InvalidEngineState('Invalid node at given pointer: %s' % obj)
                    if act == 'r' :
                        if stats != None : stats[0] += 1
                        cleanUp, next_actions = exec_node.newResult(*args,**kwdargs)
                    elif act == 'c' :
                        if stats != None : stats[1] += 1
                        cleanUp, next_actions = exec_node.complete(*args,**kwdargs)
                    else :
                        raise InvalidEngineState('Unknown message')
                if not actions and not next_actions and self.cycle_root != None :
                    next_actions = self.cycle_root.closeCycle(True)
                actions += list(reversed(next_actions))
                if debug :
                    # if type(exec_node).__name__ in ('EvalDefine',) :
                    self.printStack(obj)
                    if act in 'rco' : print (obj, act, args)
                    print ( [ (a,o,x) for a,o,x,t in actions[-10:] ])
                    if len(self.stack) > max_stack : max_stack = len(self.stack)
                if trace : 
                    a = sys.stdin.readline()
                    if a.strip() == 'gp' :
                        print (target)
                    elif a.strip() == 'l' :
                        trace = False
                        self.trace = False
                        debug = False
                        self.debug = False
                if cleanUp :
                    self.cleanUp(obj)
                
                        
        self.printStack()
        print ('Collected results:', solutions)
        raise InvalidEngineState('Engine did not complete correctly!')
    
    def cleanUp(self, obj) :
        self.stack[obj] = None
        while self.pointer > 0 and self.stack[self.pointer-1] == None :
            #self.stack.pop(-1)
            self.pointer -= 1
        
        
    def call(self, query, database, target, transform=None, **kwdargs ) :
        node_id = database.find(query)
        return self.execute( node_id, database=database, target=target, context=query.args, **kwdargs) 
                
    def printStack(self, pointer=None) :
        print ('===========================')
        for i,x in enumerate(self.stack) :
            if pointer == None or pointer - 20 < i < pointer + 20 :
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

    def __init__(self, engine, database, target, node_id, node, context, parent, pointer, identifier=None, transform=None, call=None, **extra ) :
        self.engine = engine
        self.database = database
        self.target = target
        self.node_id = node_id
        self.node = node
        self.context = context
        self.parent = parent
        self.identifier = identifier
        self.pointer = pointer
        self.transform = transform
        self.call = call
        self.on_cycle = False
        
    def notifyResult(self, arguments, node=0, is_last=False, parent=None ) :
        if parent == None : parent = self.parent
        if self.transform : arguments = self.transform(arguments)
        if arguments == None :
            if is_last :
                return self.notifyComplete()
            else :
                return []
        else :
            return [ newResult( parent, arguments, node, self.identifier, is_last ) ]
        
    def notifyComplete(self, parent=None) :
        if parent == None : parent = self.parent
        return [ complete( parent, self.identifier ) ]
        
    def createCall(self, node_id, *args, **kwdargs) :
        base_args = {}
        base_args['database'] = self.database
        base_args['target'] = self.target 
        base_args['context'] = self.context
        base_args['parent'] = self.pointer
        base_args['identifier'] = self.identifier
        base_args['transform'] = None
        base_args['call'] = self.call
        base_args.update(kwdargs)
        return call( node_id, args, base_args )
        
    def createCycle(self) :
        self.on_cycle = True
        return []
        
    def __repr__(self) :
        return '%s %s: %s %s [%s] {%s}' % (self.parent, self.__class__.__name__, self.node, self.context, self.call, self.pointer)
        
    def node_str(self) :
        return str(self.node)
        
    def __str__(self) :
        if hasattr(self.node, 'location') :
            pos = self.database.lineno(self.node.location)
        else :
            pos = None
        if pos == None : pos = '??'
        node_type = self.__class__.__name__[4:]
        return '%s %s %s [at %s:%s] | Context: %s' % (self.parent, node_type, self.node_str(), pos[0], pos[1], self.context )    

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
        
    def node_str(self) :
        return '%s' % (Term(self.node.functor, *self.node.args, p=self.node.probability), )
    
        
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
        
    def node_str(self) :
        return '%s [%s,%s]' % (Term(self.node.functor, *self.context, p=self.node.probability), self.node.group, self.node.choice)

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
        
        self.results = ResultSet()
        
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
        func = lambda result, nodes : self.target.addOr( nodes, readonly=(not cycle) )
        self.results.collapse(func)
            
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
        if self.isOnCycle() :
            res = (tuple(result))
            assert(self.results.collapsed)
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
            assert(not self.results.collapsed)
            res = (tuple(result))
            self.results[res] = node
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
                for result, node in self.results :
                    actions += self.notifyResult(result,node)
            actions += self.notifyComplete()
            return True, actions
        else :
            return False, []
            
    def createCycle(self) :
        self.on_cycle = True
        self.flushBuffer(True)
        actions = []
        for result, node in self.results :
            actions += self.notifyResult(result,node) 
        return actions
        
    def node_str(self) :
        return ''
    
    def __str__(self) :
        return EvalNode.__str__(self) + ' tc: ' + str(self.to_complete)


class NestedDict(object) :
    
    def __init__(self) :
        self.__base = {}
        
    def __getitem__(self, key) :
        p_key, s_key = key
        elem = self.__base[p_key]
        for s in s_key :
            elem = elem[s]
        return elem
        
    def get(self, key, default=None) :
        try :
            return self[key]
        except KeyError :
            return default
        
    def __contains__(self, key) :
        p_key, s_key = key
        try :
            elem = self.__base[p_key]
            for s in s_key :
                elem = elem[s]
            return True
        except KeyError :
            return False
            
    def __setitem__(self, key, value) :
        p_key, s_key = key
        if s_key :
            elem = self.__base.get(p_key)
            if elem == None :
                elem = {}
                self.__base[p_key] = elem
            for s in s_key[:-1] :
                elemN = elem.get(s)
                if elemN == None :
                    elemN = {}
                    elem[s] = elemN
                elem = elemN
            elem[s_key[-1]] = value
        else :
            self.__base[p_key] = value
        
    def __delitem__(self, key) :
        p_key, s_key = key
        if s_key :
            elem = self.__base[p_key]
            elems = []
            elems.append((p_key,self.__base,elem))
            for s in s_key[:-1] :
                elemN = elem[s]
                elems.append((s,elem,elemN))
                elem = elemN
            del elem[s_key[-1]] # Remove last element
            for s,e,ec in reversed(elems) :
                if len(ec) == 0 :
                    del e[s]
                else :
                    break
        else :
            del self.__base[p_key]

# class DefineCache(object) :
#
#     def __init__(self) :
#         self.__active = {}
#
#
#

class DefineCache(object) : 
    
    def __init__(self) :
        self.__non_ground = NestedDict()
        self.__ground = NestedDict()
        self.__active = NestedDict()
    
    def activate(self, goal, node) :
        self.__active[goal] = node
        
    def deactivate(self, goal) :
        del self.__active[goal]
        
    def is_active(self, goal) :
        return self.getEvalNode(goal) != None
        
    def getEvalNode(self, goal) :
        return self.__active.get(goal)
        
    def __setitem__(self, goal, results) :
        # Results
        functor, args = goal
        if is_ground(*args) :
            if results :
                # assert(len(results) == 1)
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
                
    def get(self, key, default=None) :
        try :
            return self[key]
        except KeyError :
            return default
                
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

class ResultSet(object) :
    
    def __init__(self) :
        self.results = []
        self.index = {}
        self.collapsed = False
    
    def __setitem__(self, result, node) :
        index = self.index.get(result)
        if index == None :
            index = len(self.results)
            self.index[result] = index
            if self.collapsed :
                self.results.append( (result,node) )
            else :
                self.results.append( (result,[node]) )
        else :
            assert(not self.collapsed)
            self.results[index][1].append( node )
        
    def __getitem__(self, result) :
        index = self.index[result]
        result, node = self.results[index]
        return node
        
    def get(self, key, default=None) :
        try :
            return self[key]
        except KeyError :
            return None
        
    def keys(self) :
        return [ result for result, node in self.results ] 
        
    def __len__(self) :
        return len(self.results)
        
    def collapse(self, function) :
        if not self.collapsed :
            for i,v in enumerate(self.results) :
                result, node = v
                collapsed_node = function(result, node)
                self.results[i] = (result,collapsed_node)
            self.collapsed = True
    
    def __contains__(self, key) :
        return key in self.index
    
    def __iter__(self) :
        return iter(self.results)
        
        
class EvalDefine(EvalNode) :
    
    # A buffered Define node.
    def __init__(self, call=None, **parent_args ) : 
        EvalNode.__init__(self, **parent_args)
        # self.__buffer = defaultdict(list)
        # self.results = None
        
        self.results = ResultSet()
        
        self.cycle_children = []
        self.cycle_close = []
        self.is_cycle_root = False
        self.is_cycle_child = False
        
        self.call = ( self.node.functor, tuple(self.context) )
        self.to_complete = None
    
    
    def __call__(self) :
        goal = (self.node.functor, tuple(self.context))
        active_node = self.target._cache.getEvalNode(goal)
        if active_node != None :
            return False, self.cycleDetected(active_node)
        else:
            results = self.target._cache.get(goal)
            if results != None :
                actions = []
                n = len(results)
                if n > 0 :
                    for result, node in results.items() :
                        n -= 1
                        if node != NODE_FALSE :
                            actions += self.notifyResult(result, node, is_last=(n==0))
                        elif n == 0 :
                            actions += self.notifyComplete()
                else :
                    actions += self.notifyComplete()
                return True, actions
            else :
                children = self.node.children.find( self.context )
                self.to_complete = len(children)
            
                if self.to_complete == 0 :
                    # No children, so complete immediately.
                    return True, self.notifyComplete()
                elif len(children) == 1 :
                    # We could clean up this node here, but:
                    #   - that would skip caching
                    #   - child should apply the transform function
                    #   - effect on cycles?
                    #       -> there is no alternative, so there should no be an effect? 
                
                    self.target._cache.activate(goal, self)
                    actions = [ self.createCall( child) for child in children ]
                    return False,  actions # + [ ('C', self.pointer, (True,), {} ) ]
                else :
                    self.target._cache.activate(goal, self)
                    actions = [ self.createCall( child) for child in children ]
                    return False,  actions # + [ ('C', self.pointer, (True,), {} ) ]
    
    def notifyResultMe(self, arguments, node=0, is_last=False ) :
        parent = self.pointer
        return [ newResult( parent, arguments, node, self.identifier, is_last ) ]
        
    def notifyResultChildren(self, arguments, node=0, is_last=False ) :
        parents = self.cycle_children
        return [ newResult( parent, arguments, node, self.identifier, is_last ) for parent in parents ]
    
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
        #if self.transform : result = self.transform(result)
        if result == None :
            if is_last :
                return self.complete(source)
            else :
                return False, []
        
        if self.is_cycle_child :
            if is_last :
                return True, self.notifyResult(result, node, is_last=is_last)
            else :
                return False, self.notifyResult(result, node, is_last=is_last)
        else :
            if self.isOnCycle() or self.isCycleParent() :
                assert(self.results.collapsed)
                res = (tuple(result))
                res_node = self.results.get(res)
                if res_node != None :
                    self.target.addDisjunct( res_node, node )
                    actions = []
                    if is_last :
                        a, act = self.complete(source)
                        actions += act
                else :
                    cache_key = (self.node.functor, res)
                    if cache_key in self.target._cache :
                        result_node = self.target._cache[ cache_key ][ res ]
                    else :
                        result_node = self.target.addOr( (node,), readonly=False )
                    name = str(Term(self.node.functor, *res))
                    self.target.addName(name, result_node, LABEL_NAMED)
                    self.results[res] = result_node
                    actions = []
                    # Send results to cycle children
                    actions += self.notifyResultChildren(res, result_node)
                    if self.isOnCycle() : actions += self.notifyResult(res, result_node)
                    if is_last : 
                        a, act = self.complete(source)
                        actions += act
                return False, actions
            else :
                assert(not self.results.collapsed)
                res = (tuple(result))
                self.results[res] = node
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
                #assert (not cache_key in self.target._cache)
                self.flushBuffer()
                self.target._cache[ cache_key ] = self.results
                self.target._cache.deactivate(cache_key)
                actions = []
                if not self.isOnCycle() : 
                    n = len(self.results)
                    if n :
                        for result, node in self.results :
                            n -= 1
                            actions += self.notifyResult(result,node,is_last=(n==0))
                    else :
                        actions += self.notifyComplete()
                else :
                    actions += self.notifyComplete()
                return True, actions
            else :
                return False, []
            
    def flushBuffer(self, cycle=False) :
        def func( res, nodes ) :
            cache_key = (self.node.functor, res)
            if cache_key in self.target._cache :
                node = self.target._cache[ cache_key ][ res ]
            else :
                node = self.target.addOr( nodes, readonly=(not cycle) )
            #node = self.target.addOr( nodes, readonly=(not cycle) )
            name = str(Term(self.node.functor, *res))
            self.target.addName(name, node, LABEL_NAMED)
            return node
        self.results.collapse(func)
                
    def isOnCycle(self) :
        return self.on_cycle
        
    def isCycleParent(self) :
        return bool(self.cycle_children)
            
    def cycleDetected(self, cycle_parent) :
        queue = []
        goal = (self.node.functor, tuple(self.context))
        # Get the top node of this cycle.
        cycle_root = self.engine.cycle_root
        # Mark this node as a cycle child
        self.is_cycle_child = True
        # Register this node as a cycle child of cycle_parent
        cycle_parent.cycle_children.append(self.pointer)
        
        cycle_parent.flushBuffer(True)
        for result, node in cycle_parent.results :
            queue += self.notifyResultMe(result,node) 
        
        if cycle_root != None and cycle_parent.pointer < cycle_root.pointer :
            # New parent is earlier in call stack as current cycle root
            # Unset current root
            # Unmark current cycle root
            cycle_root.is_cycle_root = False
            # Copy subcycle information from old root to new root
            cycle_parent.cycle_close = cycle_root.cycle_close
            cycle_root.cycle_close = []
            self.engine.cycle_root = cycle_parent
            queue += cycle_root.createCycle()
            queue += self.engine.notifyCycle(cycle_root.pointer) # Notify old cycle root up to new cycle root
            cycle_root = None
        if cycle_root == None : # No active cycle
            # Register cycle root with engine
            self.engine.cycle_root = cycle_parent
            # Mark parent as cycle root
            cycle_parent.is_cycle_root = True
            #
            cycle_parent.cycle_close.append(self.pointer)
            # Queue a create cycle message that will be passed up the call stack.
            queue += self.engine.notifyCycle(self.pointer)
            # Send a close cycle message to the cycle root.
        else :  # A cycle is already active
            # The new one is a subcycle
            cycle_root.cycle_close.append(self.pointer)            
            # Queue a create cycle message that will be passed up the call stack.
            queue += self.engine.notifyCycle(self.pointer)
        return queue
    
    def closeCycle(self, toplevel) :
        if self.is_cycle_root and toplevel :
            self.engine.cycle_root = None
            actions = []
            for cc in self.cycle_close :
                actions += self.notifyComplete(parent=cc) 
            return actions
        else :
            return []
    
    def createCycle(self) :
        if self.on_cycle :  # Already on cycle
            # Pass message to parent
            return []
        elif self.is_cycle_root :
            return []
        else :
            # Define node
            self.on_cycle = True
            self.flushBuffer(True)
            actions = []
            for result, node in self.results :
                actions += self.notifyResult(result,node) 
            return actions

    def node_str(self) :
        return str(Term(self.node.functor, *self.context))
        
    def __str__(self) :
         extra = ['tc: %s' % self.to_complete]
         if self.is_cycle_child : extra.append('CC')
         if self.is_cycle_root : extra.append('CR')
         if self.isCycleParent() : extra.append('CP') 
         if self.on_cycle : extra.append('*')
         if self.cycle_children : extra.append('c_ch: %s' % (self.cycle_children,))
         if self.cycle_close : extra.append('c_cl: %s' % (self.cycle_close,))
         return EvalNode.__str__(self) + ' ' + ' '.join(extra)

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
        
    def createCycle(self) :
        raise NegativeCycle(location=self.database.lineno(self.node.location))
        
    def node_str(self) :
        return ''
        
    
        
class EvalAnd(EvalNode) :
    
    def __init__(self, **parent_args) :
        EvalNode.__init__(self, **parent_args)
        self.to_complete = 1
        
    def __call__(self) :
        # Create a node for child 1 and call it.
        return False, [ self.createCall(  self.node.children[0], identifier=None ) ]
        
    def newResult(self, result, node=0, source=None, is_last=False) :
        if source == None :     # Result from the first conjunct.
            # We will create a second conjunct, which needs to send a 'complete' signal.
            self.to_complete += 1
            if is_last :
                # Notify self that this conjunct is complete. ('all_complete' will always be False)
                all_complete, complete_actions = self.complete()
                if False and node == NODE_TRUE :
                    # TODO THERE IS A BUG HERE 
                    # If there is only one node to complete (the new second conjunct) then
                    #  we can clean up this node, but then we would lose the ground node of the first conjunct.
                    # This is ok when it is deterministically true.  TODO make this always ok!
                    # We can redirect the second conjunct to our parent.
                    return (self.to_complete==1), [ self.createCall( self.node.children[1], context=result, parent=self.parent ) ]
                else :
                    return False, [ self.createCall( self.node.children[1], context=result, identifier=node ) ]
            else :
                # Not the last result: default behaviour
                return False, [ self.createCall( self.node.children[1], context=result, identifier=node ) ]
        else :  # Result from the second node
            # Make a ground node
            target_node = self.target.addAnd( (source, node) )
            if is_last :
                # Notify self of child completion
                all_complete, complete_actions = self.complete()
            else :
                all_complete, complete_actions = False, []
            if all_complete :
                return True, self.notifyResult(result, target_node, is_last=True)
            else :
                return False, self.notifyResult(result, target_node, is_last=False)
    
    def complete(self, source=None) :
        self.to_complete -= 1
        if self.to_complete == 0 :
            return True, self.notifyComplete()
        else :
            assert(self.to_complete > 0)
            return False, []
    
    def node_str(self) :
        return ''
    
class EvalCall(EvalNode) :
    
    def __init__(self, **parent_args ) : 
        EvalNode.__init__(self, **parent_args)
        
    def __call__(self) :
        call_args = [ instantiate(arg, self.context) for arg in self.node.args ]
        origin = '%s/%s' % (self.node.functor,len(self.node.args))
        return True, [ self.createCall( self.node.defnode, call_origin=(origin,self.node.location), context=call_args, transform=self.getResultTransform(), parent=self.parent ) ]
    
    def getResultTransform(self) :
        context = list(self.context)
        node_args = list(self.node.args)
        def result_transform(result) :
            output = context[:]
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
        
    def node_str(self) :
        call_args = [ instantiate(arg, self.context) for arg in self.node.args ]
        return '%s' %  (Term(self.node.functor, *call_args),)

    
class EvalBuiltIn(EvalNode) : 
    
    def __init__(self, call_origin=None, **kwdargs) :
        EvalNode.__init__(self, **kwdargs)
        if call_origin != None :
            self.location = call_origin[1]
        else :
            self.location = None
    
    def __call__(self) :
        return self.node(*self.context, engine=self.engine, database=self.database, target=self.target, location=self.location, callback=self, transform=self.transform)
        

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
            if is_last :
                return self.complete(source)
            else :
                return False, []
        else :
            return is_last, self.notifyResult(result, node, is_last)
        
    def complete(self, source=None) :
        return True, self.notifyComplete()
        
    def node_str(self) :
        return '%s :- ...' %  (Term(self.node.functor, *self.node.args, p=self.node.probability),)
        

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
            
    def __str__(self) :
        return str(self.base_function)
        
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
            
    def __str__(self) :
        return str(self.base_function)

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
        actions += callback.notifyResult( res_pass, node, False)
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



