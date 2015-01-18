from __future__ import print_function
from collections import defaultdict 

from problog.engine import unify, UnifyError, instantiate, extract_vars, is_ground

# New engine: notable differences
#  - keeps its own call stack -> no problem with Python's maximal recursion depth 
#  - supports skipping calls / tail-recursion optimization

# STATUS: works for non-cyclic programs?

# TODO:
#  - add choice node (annotated disjunctions)
#  - add builtins
#  - add caching of define nodes
#  - add cycle handling
#  - make clause and call skippable (only transform results) -> tail recursion optimization


def call(obj, args, kwdargs) :
    return ( 'e', obj, args, kwdargs ) 

def newResult(obj, result, ground_node, source, is_last) :
    return ( 'r', obj, (result, ground_node, source, is_last), {} )

def complete(obj, source) :
    return ( 'c', obj, (source,), {} )

class Engine(object) :
    
    def __init__(self) :
        self.stack = []
        
        self.node_types = {}
        self.node_types['fact'] = EvalFact
        self.node_types['conj'] = EvalAnd
        self.node_types['disj'] = EvalOr
        self.node_types['neg'] = EvalNot
        self.node_types['define'] = EvalDefine
        self.node_types['call'] = EvalCall
        self.node_types['clause'] = EvalClause
        self.node_types['choice'] = EvalChoice
        
    def create_node_type(self, node_type) :
        return self.node_types[node_type]
    
    def create(self, node_id, database, **kwdargs ) :
        # TODO built-ins -> negative node_id
        
        node = database.getNode(node_id)
        node_type = type(node).__name__
        
        exec_node = self.create_node_type( node_type )
        
        pointer = len(self.stack) 
        self.stack.append(exec_node(pointer=pointer, engine=self, database=database, node_id=node_id, node=node, **kwdargs))
        return pointer
        
    def call_node( self, pointer ) :
        return self.stack[pointer]()
        
        
    def execute(self, node_id, **kwdargs ) :
        pointer = self.create( node_id, parent=None, **kwdargs )
        cleanUp, actions = self.call_node(pointer)
        
        max_stack = len(self.stack)
        solutions = []
        while actions :
            act, obj, args, kwdargs = actions.pop(-1)
            if obj == None :
                if act == 'r' :
                    solutions.append( (args[0], args[1]) )
                elif act == 'c' :
                    print ('Maximal stack size:', max_stack)
                    return solutions
            else:
                if act == 'e' :
                    obj = self.create( obj, *args, **kwdargs )
                    exec_node = self.stack[obj]
                    cleanUp, next_actions = exec_node()
                else:
                    exec_node = self.stack[obj]
                    if exec_node == None :
                        print (act, obj)
                        raise Exception()
                    if act == 'r' :
                        cleanUp, next_actions = exec_node.newResult(*args,**kwdargs)
                    elif act == 'c' :
                        cleanUp, next_actions = exec_node.complete(*args,**kwdargs)
                actions += list(reversed(next_actions))
                # self.printStack(obj)
                # if act in 'rc' : print (obj, act, args)
                if len(self.stack) > max_stack : max_stack = len(self.stack)
                if cleanUp :
                    #print ('cleanUp', obj, self.stack[obj])
                    self.stack[obj] = None
                    while self.stack and self.stack[-1] == None :
                        self.stack.pop(-1)
        return solutions
        
    def printStack(self, pointer=None) :
        print ('===========================')
        for i,x in enumerate(self.stack) :
            if i == pointer :
                print ('>>> %s: %s' % (i,x) )
            else :
                print ('    %s: %s' % (i,x) )
        
        
        

NODE_TRUE = 0
NODE_FALSE = None


class EvalNode(object):

    def __init__(self, engine, database, target, node_id, node, context, parent, pointer, identifier=None, transform=None ) :
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
        
        
    def notifyResult(self, arguments, node=0, is_last=False ) :
        if self.transform == None :
            return [ newResult( self.parent, arguments, node, self.identifier, is_last )  ]
        else :
            print ('Transform:', arguments, self)
            return [ newResult( self.parent, self.transform(arguments), node, self.identifier, is_last )  ]
        
    def notifyComplete(self) :
        return [ complete( self.parent, self.identifier )  ]
        
    def createCall(self, node_id, *args, **kwdargs) :
        base_args = {}
        base_args['database'] = self.database
        base_args['target'] = self.target 
        base_args['context'] = self.context
        base_args['parent'] = self.pointer
        base_args['identifier'] = self.identifier
        base_args['transform'] = self.transform
        base_args.update(kwdargs)
        return call( node_id, args, base_args )
        
        
    def __repr__(self) :
        return '%s: %s %s' % (self.__class__.__name__, self.node, self.context)
        

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
        
        self.__buffer = defaultdict(list)
        
    def __call__(self) :
        children = self.node.children
        self.to_complete = len(children)
        if self.to_complete == 0 :
            # No children, so complete immediately.
            return True, self.notifyComplete()
        else :
            return False, [ self.createCall( child ) for child in children ]
            
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
        res = (tuple(result))
        self.__buffer[res].append( node )
        if is_last :
            return self.complete(source)
        else :
            return False, []
        
    def complete(self, source=None) :
        self.to_complete -= 1
        if self.to_complete == 0 :
            actions = []
            if len(self.__buffer) == 1 :
                for result, nodes in self.__buffer.items() :
                    target_node = self.target.addOr( nodes )
                    actions += self.notifyResult(result, target_node, True )
            else :
                for result, nodes in self.__buffer.items() :
                    target_node = self.target.addOr( nodes )
                    actions += self.notifyResult(result, target_node )
                actions += self.notifyComplete()
            return True, actions
        else :
            assert( self.to_complete > 0 )
            return False, []
            
class EvalDefine(EvalNode) :
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
        self.__buffer = defaultdict(list)
        
    def __call__(self) :
        children = self.node.children.find( self.context )
        self.to_complete = len(children)
        
        if self.to_complete == 0 :
            # No children, so complete immediately.
            return True, self.notifyComplete()
        elif len(children) == 1 :
            return True, [ self.createCall( child, parent=self.parent ) for child in children ]
        else :
            return False, [ self.createCall( child ) for child in children ]
            
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
        res = (tuple(result))
        self.__buffer[res].append( node )
        if is_last :
            return self.complete(source)
        else :
            return False, []
        
    def complete(self, source=None) :
        self.to_complete -= 1
        if self.to_complete == 0 :
            actions = []
            for result, nodes in self.__buffer.items() :
                target_node = self.target.addOr( nodes )
                actions += self.notifyResult(result, target_node )
            actions += self.notifyComplete()
            return True, actions
        else :
            assert( self.to_complete > 0 )
            return False, []



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
                actions += self.notifyResult(self.context, or_node)
        else :
            actions += self.notifyResult(self.context, NODE_TRUE)
        actions += self.notifyComplete()
        return True, actions
        
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
                    return True, [ self.createCall( self.node.children[1], context=result, parent=self.parent ) ]
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
                return output
            except UnifyError :
                pass
        return result_transform
            
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
        result = self.getResultTransform()( result )
        if result == None :
            return self.complete(source)
        else :
            return is_last, self.notifyResult(result, node, is_last)
        
    def complete(self, source=None) :
        return True, self.notifyComplete()    

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
            return output
            
        return result_transform
        
    def newResult(self, result, node=NODE_TRUE, source=None, is_last=False ) :
        result = self.getResultTransform()( result )
        if result == None :
            return self.complete(source)
        else :
            return is_last, self.notifyResult(result, node, is_last)
        
    def complete(self, source=None) :
        return True, self.notifyComplete()



def test(filename) :
    import problog
    
    pl = problog.program.PrologFile(filename)
    db = problog.program.ClauseDB.createFrom( pl )
    target = problog.formula.LogicFormula()
    
    eng = Engine()
    
    context = [None]
    parent = None
    
    print (db)
    print ('===============')
        
    query_node = db.find(problog.logic.Term('query',None) )
    print (query_node)
    queries = eng.execute( query_node, database=db, target=target, context=[None] )
        
    query = queries[0][0][0]
    node_id = db.find(query)
    results = eng.execute( node_id, database=db, target=target, context=([None]*query.arity) )
    
    print (target)
    print (results)
    
    
    
    
if __name__ == '__main__' :
    import sys
    
    test(sys.argv[1])