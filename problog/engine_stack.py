from __future__ import print_function
from collections import defaultdict 

import sys, os
import imp, inspect # For load_external

from .formula import LogicFormula
from .program import ClauseDB, PrologFile
from .logic import Term
from .core import LABEL_NAMED
from .engine import unify, UnifyError, instantiate, extract_vars, is_ground, UnknownClause, _UnknownClause, ConsultError
from .engine import addStandardBuiltIns, check_mode, GroundingError, NonGroundProbabilisticClause
from .engine import ClauseDBEngine


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


class StackBasedEngine(ClauseDBEngine) :
    
    def __init__(self, *args, **kwdargs) :
        ClauseDBEngine.__init__(self,*args,**kwdargs)
        
        self.node_types = {}
        self.node_types['fact'] = self.eval_fact
        self.node_types['conj'] = self.eval_conj
        self.node_types['disj'] = self.eval_disj
        self.node_types['neg'] = self.eval_neg
        self.node_types['define'] = self.eval_define
        self.node_types['call'] = self.eval_call
        self.node_types['clause'] = self.eval_clause
        self.node_types['choice'] = self.eval_choice
        self.node_types['builtin'] = self.eval_builtin
        
        self.cycle_root = None
        self.pointer = 0
        self.stack_size = 128
        self.stack = [None] * self.stack_size
        
        self.stats = [0,0,0,0]
        
        self.debug = False
        self.trace = False
    
    def eval(self, node_id, **kwdargs) :
        database = kwdargs['database']
        if node_id < 0 :
            node_type = 'builtin'
            node = self._getBuiltIn(node_id)
        else :
            node = database.getNode(node_id)
            node_type = type(node).__name__
        
        exec_func = self.create_node_type( node_type )        
        return exec_func(node_id=node_id, node=node, **kwdargs)
        
    def create_node_type(self, node_type) :
        try :
            return self.node_types[node_type]
        except KeyError :
            raise _UnknownClause()
            
    def loadBuiltIns(self) :
        addBuiltIns(self)
    
    def create_record(self, exec_node, pointer, database, node_id, node, **kwdargs ) :
        return exec_node(pointer=pointer, engine=self, database=database, node_id=node_id, node=node, **kwdargs)
        
    def grow_stack(self) :
        self.stack += [None] * self.stack_size
        self.stack_size = self.stack_size * 2
        
    def shrink_stack(self) :
        self.stack_size = 128
        self.stack = [None] * self.stack_size
            
    def add_record(self, record) :
        if self.pointer >= self.stack_size : self.grow_stack()
        self.stack[self.pointer] = record
        self.pointer += 1
    
    def notifyCycle(self, childnode) :
        # Optimization: we can usually stop when we reach a node on_cycle.
        #   However, when we swap the cycle root we need to also notify the old cycle root up to the new cycle root.
        assert(self.cycle_root != None)
        root = self.cycle_root.pointer
        #childnode = self.stack[child]
        current = childnode.parent
        actions = []
        while current != root :
            exec_node = self.stack[current]
            if exec_node.on_cycle : 
                break
            new_actions = exec_node.createCycle()
            actions += new_actions
            current = exec_node.parent
        return actions
        
    def execute(self, node_id, subcall=False, **kwdargs ) :
        self.trace = kwdargs.get('trace')
        self.debug = kwdargs.get('debug') or self.trace
        stats = kwdargs.get('stats')    # Should support stats[i] += 1 with i=0..5
        
        target = kwdargs['target']
        database = kwdargs['database']
        if not hasattr(target, '_cache') : target._cache = DefineCache()
        actions = self.eval( node_id, parent=None, **kwdargs)
        cleanUp = False
        
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
                        if not subcall and self.pointer != 0 :
                            self.printStack()
                            raise InvalidEngineState('Stack not empty at end of execution!')
                        if not subcall : self.shrink_stack()
                        return solutions
                elif act == 'R' :
                    solutions += args[0]
                    if args[2] :
                        return solutions
                elif act == 'c' :
                    if stats != None : stats[1] += 1
                    if self.debug : print ('Maximal stack size:', max_stack)
                    return solutions
                else :
                    raise InvalidEngineState('Unknown message!')
            else:
                if act == 'e' :
                    if self.cycle_root != None and kwdargs['parent'] < self.cycle_root.pointer :
                        cleanUp = False
                        next_actions = self.cycle_root.closeCycle(True) + [ (act,obj,args,kwdargs) ]
                    else :
                        try :
                            if stats != None : stats[2] += 1
                            next_actions = self.eval( obj, *args, **kwdargs )
                            cleanUp = False
                            obj = self.pointer
                        except _UnknownClause : 
                            call_origin = kwdargs.get('call_origin')
                            if call_origin == None :
                                sig = 'unknown'
                                raise UnknownClause(sig, location=None)
                            else :
                                loc = database.lineno(call_origin[1])
                                raise UnknownClause(call_origin[0], location=loc)
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
                    elif act == 'R' :
                        if stats != None : stats[3] += 1
                        cleanUp, next_actions = exec_node.newResultMulti(*args,**kwdargs)
                    elif act == 'c' :
                        if stats != None : stats[1] += 1
                        cleanUp, next_actions = exec_node.complete(*args,**kwdargs)
                    else :
                        raise InvalidEngineState('Unknown message')
                if not actions and not next_actions and self.cycle_root != None :
                    next_actions = self.cycle_root.closeCycle(True)
                actions += list(reversed(next_actions))
                if self.debug :
                    # if type(exec_node).__name__ in ('EvalDefine',) :
                    self.printStack(obj)
                    if act in 'rco' : print (obj, act, args)
                    print ( [ (a,o,x) for a,o,x,t in actions[-10:] ])
                    if len(self.stack) > max_stack : max_stack = len(self.stack)
                    if self.trace : 
                        a = sys.stdin.readline()
                        if a.strip() == 'gp' :
                            print (target)
                        elif a.strip() == 'l' :
                            self.trace = False
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
            if (pointer == None or pointer - 20 < i < pointer + 20) and x != None :
                if i == pointer :
                    print ('>>> %s: %s' % (i,x) )
                elif self.cycle_root != None and i == self.cycle_root.pointer  :
                    print ('ccc %s: %s' % (i,x) )
                else :
                    print ('    %s: %s' % (i,x) )        
        
    def eval_fact(engine, parent, node_id, node, context, target, identifier, **kwdargs) :
        actions = []
        try :
            # Verify that fact arguments unify with call arguments.
            for a,b in zip(node.args, context) :
                unify(a, b)
            # Successful unification: notify parent callback.
            target_node = target.addAtom(node_id, node.probability)
            if target_node != None :
                return [ newResult( parent, node.args, target_node, identifier, True ) ]
            else :
                return [ complete( parent, identifier) ]
        except UnifyError :
            # Failed unification: don't send result.
            # Send complete message.
            return [ complete( parent, identifier) ]
        return actions        # Pointer actions
        
            
    def eval_define(engine, node, context, target, parent, identifier=None, transform=None, **kwdargs) :
        goal = (node.functor, tuple(context))

        engine.stats[2] += 1
        results = target._cache.get(goal)
        if results != None :
            engine.stats[3] += 1
            actions = []
            n = len(results)
            if n > 0 :
                for result, target_node in results :
                    n -= 1
                    if target_node != NODE_FALSE :
                        if transform : result = transform(result)
                        if result == None :
                            if n == 0 :
                                actions += [ complete(parent, identifier) ]
                            else :
                                pass
                        else :
                            actions += [ newResult( parent, result, target_node, identifier, n==0 ) ]
                    elif n == 0 :
                        actions += [ complete(parent, identifier) ]
            else :
                actions += [ complete(parent, identifier) ]
            return actions
        else :
            active_node = target._cache.getEvalNode(goal)
            engine.stats[0] += 1
            if active_node != None :
                engine.stats[1] +=1
                # If current node is ground and active node has results already, then we can simple send that result.
                if active_node.is_ground and active_node.results :
                    active_node.flushBuffer(True)
                    active_node.is_cycle_parent = True  # Notify it that it's buffer was flushed
                    queue = []
                    for result, node in active_node.results :
                        if transform : result = transform(result)
                        if result == None :
                            queue += [ complete(parent, identifier) ]
                        else :
                            queue += [ newResult( parent, result, node, identifier, True ) ]
                    assert(len(queue) == 1)
                    return queue
                else :
                    evalnode = EvalDefine( pointer=engine.pointer, engine=engine, node=node, context=context, target=target, identifier=identifier, parent=parent, transform=transform, **kwdargs )
                    engine.add_record(evalnode)
                    return evalnode.cycleDetected(active_node)
            else :
                children = node.children.find( context )
                to_complete = len(children)
            
                if to_complete == 0 :
                    # No children, so complete immediately.
                    return [ complete(parent, identifier) ]
                else :
                    evalnode = EvalDefine( to_complete=to_complete, pointer=engine.pointer, engine=engine, node=node, context=context, target=target, identifier=identifier, transform=transform, parent=parent, **kwdargs )
                    engine.add_record(evalnode)
                    target._cache.activate(goal, evalnode)
                    actions = [ evalnode.createCall( child) for child in children ]
                    return actions
    
    def eval_conj(engine, **kwdargs) :
        return engine.eval_default(EvalAnd, **kwdargs)
    
    def eval_disj(engine, parent, node, transform=None, **kwdargs) :
        if len(node.children) == 0 :
            # No children, so complete immediately.
            return [ complete( parent, None) ]
        else :
            evalnode = EvalOr( pointer=engine.pointer, engine=engine, parent=parent, node=node, **kwdargs)
            engine.add_record( evalnode )
            return [ evalnode.createCall( child ) for child in node.children ]

    def eval_neg(engine, **kwdargs) :
        return engine.eval_default(EvalNot, **kwdargs)
    
    def eval_call(engine, node_id, node, context, **kwdargs) :
        mask = [ is_ground(s) for s in node.args ]
        def result_transform(result) :
            output = list(context)
            actions = []
            try :
                assert(len(result) == len(node.args))
                for call_arg, res_arg in zip(node.args,result) :
                    unify( res_arg, call_arg, output )
                return tuple(output)
            except UnifyError :
                pass
            
        call_args = [ instantiate(arg, context) for arg in node.args ]
        origin = '%s/%s' % (node.functor,len(node.args))
    
        kwdargs['call_origin'] = (origin,node.location)
        kwdargs['context'] = call_args
        kwdargs['transform'] = result_transform
    #    actions = [ call( node.defnode, (), kwdargs ) ]
        actions = engine.eval( node.defnode, **kwdargs )   
        #print ( [(a,b,c) for a,b,c,d in actions ])
        return actions
        
    def eval_clause(engine, **kwdargs) :
        return engine.eval_default(EvalClause, **kwdargs)
    
    def eval_choice(engine, parent, node_id, node, context, target, database, identifier, **kwdargs) :
        actions = []
        result = tuple(context)
        if not is_ground(*result) : raise NonGroundProbabilisticClause(location=database.lineno(node.location))
        probability = instantiate( node.probability, result )
        # Create a new atom in ground program.
        origin = (node.group, result)
        ground_node = target.addAtom( origin + (node.choice,) , probability, group=origin ) 
        # Notify parent.
        
        if ground_node != None :
            return [ newResult( parent, result, ground_node, identifier, True ) ]
        else :
            return [ complete( parent, identifier ) ]
    
    def eval_builtin(engine, **kwdargs) :
        return engine.eval_default(EvalBuiltIn, **kwdargs)
    
    def eval_default(engine, eval_type, **kwdargs) :
        node = eval_type(pointer=engine.pointer, engine=engine, **kwdargs)
        cleanUp, actions = node()    # Evaluate the node
        if not cleanUp : engine.add_record( node )
        return actions



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
            
    def notifyResultMulti(self, results, complete, parent=None) :
        if parent == None : parent = self.parent
        if self.transform : results = [ ( self.transform(res), node ) for res,node in results ]
        results = [ (res,node) for res, node in results if res != None ]
        if results :
            return [ newResultMulti(parent,results,self.identifier,complete) ]
        elif complete :
            return self.notifyComplete()
        else :
            return []
        
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
        
    def newResultMulti(self, results, source, complete ) :
        n = len(results)
        actions = []
        cleanUp = False
        for res, node in results :
            n -= 1
            cleanUpN, actionsN = self.newResult(res,node,source, (complete and n==0) )
            cleanUp |= cleanUpN
            actions += actionsN
        return cleanUp, actions
        
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
        self.to_complete = len(self.node.children)
            
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
                else :
                    a = False
            return a, actions
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
    
    def __str__(self) :
        return str(self.__base)

class DefineCache(object) : 
    
    def __init__(self) :
        self.__non_ground = {} #NestedDict()
        self.__ground = {} # NestedDict()
        self.__active = {} #NestedDict()
    
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
            return  [ (args,self.__ground[goal]) ]
        else :
            res_keys = self.__non_ground[goal]
            result = [ (res_key, self.__ground[(functor,res_key)]) for res_key in res_keys ]
#            result = { res_key : self.__ground[(functor,res_key)] for res_key in res_keys }
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
        
    def __str__(self) :
        return str(self.results)
        
        
class EvalDefine(EvalNode) :
    
    # A buffered Define node.
    def __init__(self, call=None, to_complete=None, **parent_args ) : 
        EvalNode.__init__(self,  **parent_args)
        # self.__buffer = defaultdict(list)
        # self.results = None
        
        self.results = ResultSet()
        
        self.cycle_children = []
        self.cycle_close = set()
        self.is_cycle_root = False
        self.is_cycle_child = False
        self.is_cycle_parent = False
        
        self.call = ( self.node.functor, tuple(self.context) )
        self.to_complete = to_complete
        self.is_ground = is_ground(*self.context)
        
    def notifyResultMe(self, arguments, node=0, is_last=False ) :
        parent = self.pointer
        return [ newResult( parent, arguments, node, self.identifier, is_last ) ]
        
    def notifyResultChildren(self, arguments, node=0, is_last=False ) :
        parents = self.cycle_children
        return [ newResult( parent, arguments, node, self.identifier, is_last ) for parent in parents ]
    
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
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
                        a = None
                    return a, actions
                else :
                    cache_key = (self.node.functor, res)
                    if cache_key in self.target._cache :
                        # Get direct
                        stored_result = self.target._cache[ cache_key ]
                        assert(len(stored_result) == 1)
                        result_node = stored_result[ 0 ][1]
                    else :
                        result_node = self.target.addOr( (node,), readonly=False )
                    name = str(Term(self.node.functor, *res))
                    self.target.addName(name, result_node, LABEL_NAMED)
                    self.results[res] = result_node
                    actions = []
                    # Send results to cycle children
                    #actions += self.notifyResultChildren(res, result_node)
                    # if self.pointer == 49336 :
                    #     self.engine.debug = True
                    #     self.engine.trace = True
                    if self.isOnCycle() : actions += self.notifyResult(res, result_node)
                    if self.is_ground and self.engine.debug : print ('Notify children with is_last:', self.pointer, self.cycle_children)
                    actions += self.notifyResultChildren(res, result_node, is_last=self.is_ground)
                    if self.is_ground :
                        self.engine.cycle_root.cycle_close -= set(self.cycle_children)
                    #    actions += [ complete( p, self.identifier) for p in self.cycle_children ]
                    if is_last : 
                        a, act = self.complete(source)
                        actions += act
                    else :
                        a = False
                    return a, actions
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
                stored_result = self.target._cache[ cache_key ]
                assert(len(stored_result)==1) 
                node = stored_result[0][1]
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
        return bool(self.cycle_children) or self.is_cycle_parent
            
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
            cycle_root.cycle_close = set()
            self.engine.cycle_root = cycle_parent
            queue += cycle_root.createCycle()
            queue += self.engine.notifyCycle(cycle_root) # Notify old cycle root up to new cycle root
            cycle_root = None
        if cycle_root == None : # No active cycle
            # Register cycle root with engine
            self.engine.cycle_root = cycle_parent
            # Mark parent as cycle root
            cycle_parent.is_cycle_root = True
            #
            cycle_parent.cycle_close.add(self.pointer)
            # Queue a create cycle message that will be passed up the call stack.
            queue += self.engine.notifyCycle(self)
            # Send a close cycle message to the cycle root.
        else :  # A cycle is already active
            # The new one is a subcycle
            #if not self.is_ground :
            cycle_root.cycle_close.add(self.pointer)
            # Queue a create cycle message that will be passed up the call stack.
            queue += self.engine.notifyCycle(self)
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
        
        # base_args = {}
        # base_args['database'] = self.database
        # base_args['target'] = self.target
        # base_args['context'] = self.context
        # base_args['parent'] = self.pointer
        # base_args['identifier'] = None
        # base_args['transform'] = None
        # base_args['call'] = self.call
        # #print ('AND')
        # and_actions = self.engine.eval( self.node.children[0], **base_args )
        # #print ("AND", and_actions)
        # return False, and_actions
        
        # a,b,c,d = self.createCall(  self.node.children[0], identifier='fc' )
        # # # del d['target']
        # # # del d['database']
        # # #print ('And: ', a,b,c,d)
        # #
        # next_actions = self.engine.eval( b, *c, **d )
        # self.engine.debug = True
        # self.engine.trace = True
        # actions = []
        # cleanUp = False
        # for a,b,c,d in next_actions :
        #     if a == 'r' and b == self.pointer :
        #         cU, act = self.newResult( *c, **d )
        #         actions += act
        #         cleanUp |= cU
        #     elif a == 'c' and b == self.pointer :
        #         cU, act = self.complete( *c, **d )
        #         actions += act
        #         cleanUp |= cU
        #     else :
        #         actions.append( (a,b,c,d) )
        # print ('AND', self.pointer, cleanUp, [ (a,b,c,d.get('parent'),d.get('identifier')) for a,b,c,d in actions] )
        #
        # return cleanUp, actions
        #
        # # return False, next_actions
        
        
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
    
    def __str__(self) :
        return EvalNode.__str__(self) + ' tc: %s' % self.to_complete
    
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
        
    def newResultMulti(self, results, source, complete ) :
        transform = self.getResultTransform()
        results = [ (transform(res), node) for res, node in results ]
        results = [ (res, node) for res, node in results if res != None ]
        if results :
            return complete, [ newResultMulti(self.parent, results, source, complete) ]
        else :        
            return self.complete(source)
        
        
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

    

def builtin_call( term, args=(), engine=None, callback=None, **kwdargs ) :
    # TODO does not support cycle through call!
    check_mode( (term,), 'c', functor='call' )
    # Find the define node for the given query term.
    term_call = term.withArgs( *(term.args + args ))
    results = engine.call( term_call, subcall=True, **kwdargs )
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


