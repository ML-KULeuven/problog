from __future__ import print_function

from .program import ClauseDB, PrologString, PrologFile
from .logic import Term, Constant, InstantiationError
from .formula import LogicFormula

from collections import defaultdict
import os

"""
Assumptions
-----------
Assumption 1: no unification of non-ground terms (e.g. unbound variables)
Assumption 3: conjunction nodes have exactly two children   (internal representation only)

Assumption 8: no prolog builtins 
    - added some builtins (needs better framework)

-- REMOVED: Assumption 4: no OR
-- REMOVED: Assumption 5: no NOT
-- REMOVED: Assumption 7: no probabilistic grounding
-- REMOVED: Assumption 6: no CHOICE
-- REMOVED: Assumption 2: functor-free


Properties
----------

    1. Each source sends each message at least once to each listener, irrespective of when the listener was added. 	(complete information)
    2. Each source sends each message at most once to each listener. (no duplicates)
    3. Each source sends a ``complete`` message to each listener.	(termination)
    4. The ``complete`` message is the last message a listener receives from a source.  (correct order)


Variable representation
-----------------------

Variables can be represented as:

    * ``Var`` objects (these are removed during compilation)
    * Integer referring to its index in the current context
    * ``None`` unset value passed from other context

To make ``==`` and ``\==`` work, variable identifiers should be passed from higher context (this would complicate call-key computation).
To make ``=`` work variables should be redirectable (but this would complicate tabling).

"""

class UnifyError(Exception) : pass

def is_variable( v ) :
    """Test whether a Term represents a variable.
    
    :return: True if the expression is a variable
    """
    return v == None or type(v) == int
    
def is_ground( *terms ) :
    """Test whether a any of given terms contains a variable (recursively).
    
    :return: True if none of the arguments contains any variables.
    """
    for term in terms :
        if is_variable(term) :
            return False
        elif not is_ground(*term.args) : 
            return False
    return True
    
def instantiate( term, context ) :
    """Replace variables in Term by values based on context lookup table."""
    if term == None :
        return None
    elif type(term) == int :
        return context[term]
    else :
        return term.apply(context)
        

def unify_value( v1, v2 ) :
    """Test unification of two values and return most specific unifier."""
    
    if is_variable(v1) :
        return v2
    elif is_variable(v2) :
        return v1
    elif v1.signature == v2.signature : # Assume Term
        return v1.withArgs(*[ unify_value(a1,a2) for a1, a2 in zip(v1.args, v2.args) ])
    else :
        raise UnifyError()
        
def unify( source_value, target_value, target_context=None ) :
    """Unify two terms.
        If a target context is given, the context will be updated using the variable identifiers from the first term, and the values from the second term.
        
        :raise UnifyError: unification failed
        
    """    
    if type(target_value) == int :
        if target_context != None :
            current_value = target_context[target_value]
            if current_value == None :
                target_context[target_value] = source_value
            else :
                new_value = unify_value( source_value, current_value )
                target_context[target_value] = new_value
    elif target_value == None :
        pass
    else :
        assert( isinstance(target_value, Term) )
        if source_value == None :  # a variable
            pass
        else :
            assert( isinstance( source_value, Term ) )
            if target_value.signature == source_value.signature :
                for s_arg, t_arg in zip(source_value.args, target_value.args) :
                    unify( s_arg, t_arg, target_context )
            else :
                raise UnifyError()
    

class VariableUnification(Exception) : 
    """The engine does not support unification of two unbound variables."""
    
    def __init__(self) :
        Exception.__init__(self, 'Unification of unbound variables not supported!')
    

class _UnknownClause(Exception) :
    """Undefined clause in call used internally."""
    pass

class UnknownClause(Exception) :
    """Undefined clause in call."""
    
    def __init__(self, signature) :
        Exception.__init__(self, "No clauses found for '%s'!" % signature)


class Context(object) :
    """Variable context."""
    
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
        """Add a builtin."""
        sig = '%s/%s' % (pred, arity)
        self.__builtin_index[sig] = -(len(self.__builtins) + 1)
        self.__builtins.append( func )
        
    def getBuiltIns(self) :
        """Get the list of builtins."""
        return self.__builtin_index
    
    def _enter_call(self, node, context) :
        if self.debugger :
            self.debugger.enter(0, node, context)
        
    def _exit_call(self, node, context) :
        if self.debugger :
            self.debugger.exit(0, node, context, None)
    
    def _create_context(self, lst=[], size=None, define=None) :
        """Create a new context."""
        if size != None :
            assert( not lst )
            lst = [None] * size
        return Context(lst, define=define)
    
    def query(self, db, term, level=0) :
        """Perform a non-probabilistic query."""
        gp = LogicFormula()
        gp, result = self._ground(db, term, gp, level)
        return [ y for x,y in result ]
            
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
        res = ResultCollector()
        directives = db.getNode(directive_node).children
        while directives :
            current = directives.pop(0)
            self._eval( db, gp, current, self._create_context((),define=None), res )
            
        # # TODO warning if len(res.results) != number of directives
        # return

        
    
    def ground(self, db, term, gp=None, label=None) :
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
        gp, results = self._ground(db, term, gp)
        
        for node_id, args in results :
            gp.addName( term.withArgs(*args), node_id, label )
        if not results :
            gp.addName( term, None, label )
        
        return gp
    
    def _ground(self, db, term, gp=None, level=0) :
        # Convert logic program if needed.
        db = self.prepare(db)
        # Create a new target datastructure if none was given.
        if gp == None : gp = LogicFormula()
        # Find the define node for the given query term.
        clause_node = db.find(term)
        # If term not defined: fail query (no error)    # TODO add error to make it consistent?
        if clause_node == None : return gp, []
        # Create a new call.
        call_node = ClauseDB._call( term.functor, range(0,len(term.args)), clause_node )
        # Initialize a result collector callback.
        res = ResultCollector()
        try :
            # Evaluate call.
            self._eval_call(db, gp, None, call_node, self._create_context(term.args,define=None), res )
        except RuntimeError as err :
            if str(err).startswith('maximum recursion depth exceeded') :
                raise CallStackError()
            else :
                raise
        # Return ground program and results.
        return gp, res.results
    
    def _eval(self, db, gp, node_id, context, parent) :
        # Find the node and determine its type.
        node = db.getNode( node_id )
        ntype = type(node).__name__
        # Notify debugger of enter event.
        self._enter_call( node, context )
        # Select appropriate method for handling this node type.
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
        # Evaluate the node.
        f(db, gp, node_id, node, context, parent)
        # Notify debugger of exit event.
        self._exit_call( node, context )
        
    def _eval_fact( self, db, gp, node_id, node, call_args, parent ) :
        try :
            # Verify that fact arguments unify with call arguments.
            for a,b in zip(node.args, call_args) :
                unify(a, b)
            # Successful unification: notify parent callback.
            parent.newResult( node.args, ground_node=gp.addAtom(node_id, node.probability) )
        except UnifyError :
            # Failed unification: don't send result.
            pass
        # Send complete message.
        parent.complete()    

    def _eval_choice( self, db, gp, node_id, node, call_args, parent ) :
        # This never fails.
        # Choice is ground so result is the same as call arguments.
        result = tuple(call_args)
        # Ground probability.
        probability = instantiate( node.probability, call_args )
        # Create a new atom in ground program.
        origin = (node.group, result)
        ground_node = gp.addAtom( (node.group, result, node.choice) , probability, group=(node.group, result) ) 
        # Notify parent.
        parent.newResult( result, ground_node )
        parent.complete()
    
    def _eval_call( self, db, gp, node_id, node, context, parent ) :
        # Ground the call arguments based on the current context.
        call_args = [ instantiate(arg, context) for arg in node.args ]
        # Create a context switching node that unifies the results of the call with the call arguments. Results are passed to the parent callback.
        context_switch = ProcessCallReturn( node.args, context, parent )
        # Evaluate the define node.
        if node.defnode < 0 :
            # Negative node indicates a builtin.
            builtin = self._getBuiltIn( node.defnode )
            builtin( *call_args, context=context, callback=context_switch, database=db, engine=self, ground_program=gp )
        else :
            # Positive node indicates a non-builtin.
            try :
                # Evaluate the define node.
                self._eval( db, gp, node.defnode, self._create_context(call_args, define=context.define), context_switch )
            except _UnknownClause :
                # The given define node is empty: no definition found for this clause.
                sig = '%s/%s' % (node.functor, len(node.args))
                raise UnknownClause(sig)
            
    def _eval_clause( self, db, gp, node_id, node, call_args, parent ) :
        try :
            # Create a new context (i.e. variable values).
            context = self._create_context(size=node.varcount,define=call_args.define)
            # Fill in the context by unifying clause head arguments with call arguments.
            for head_arg, call_arg in zip(node.args, call_args) :
                # Remove variable identifiers from calling context.
                if type(call_arg) == int : call_arg = None
                # Unify argument and update context (raises UnifyError if not possible)
                unify( call_arg, head_arg, context)                
            # Create a context switching node that extracts the head arguments from the results obtained by evaluating the body. These results are send by the parent.
            context_switch = ProcessBodyReturn( node.args, node, node_id, parent )
            # Evaluate the body. Use context-switch as callback.
            self._eval( db, gp, node.child, context, context_switch )
        except UnifyError :
            # Call and clause head are not unifiable, just fail (complete without results).
            parent.complete()
            
    def _eval_conj( self, db, gp, node_id, node, context, parent ) :
        # Extract children (always exactly two).
        child1, child2 = node.children
        # Create a link between child1 and child2.
        # The link receives results of the first child and evaluates the second child based on the result.
        # The link receives the complete event from the first child and passes it to the parent.
        process = ProcessLink( self, db, gp, child2, parent, context.define )
        # Start evaluation of first child.
        self._eval( db, gp, child1, context, process )

    def _eval_disj( self, db, gp, node_id, node, context, parent ) :
        # Create a disjunction processor node, and register parent as listener.
        process = ProcessOr( len(node.children), parent )
        # Process all children.
        for child in node.children :
            self._eval( db, gp, child, context, process )

    def _eval_neg(self, db, gp, node_id, node, context, parent) :
        # Create a negation processing node, and register parent as listener.
        process = ProcessNot( gp, context, parent)
        # Evaluate the child node. Use processor as callback.
        self._eval( db, gp, node.child, context, process )
    
    def _eval_define( self, db, gp, node_id, node, call_args, parent ) :
        # Create lookup key. We will reuse results for identical calls.
        # EXTEND support call subsumption?
        key = (node_id, tuple(call_args))
        
        # Store cache in ground program
        if not hasattr(gp, '_def_nodes') : gp._def_nodes = {}
        def_nodes = gp._def_nodes
        
        # Find pre-existing node.
        pnode = def_nodes.get(key)
        if pnode == None :
            # Node does not exist: create it and add it to the list.
            pnode = ProcessDefine( self, db, gp, node_id, node, call_args, call_args.define )
            def_nodes[key] = pnode
            # Add parent as listener.
            pnode.addListener(parent)
            # Execute node. Note that for a given call (key), this is only done once!
            pnode.execute()
        else :
            # Node exists already.
            if call_args.define and call_args.define.hasAncestor(pnode) :
                # Cycle detected!
                # EXTEND Mark this information in the ground program?
                cnode = ProcessDefineCycle(pnode, call_args.define, parent)
            else :
                # Add ancestor here.
                pnode.addAncestor(call_args.define)
                # Not a cycle, just reusing. Register parent as listener (will retrigger past events.)
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
    
    def notifyListeners(self, result, ground_node=0) :
        """Send the ``newResult`` event to all the listeners of this node.
            The arguments are used as the arguments of the event.
        """
        EngineLogger.get().sendResult(self, result, ground_node)
        for listener, evttype in self.listeners :
            if evttype & self.EVT_RESULT :
                listener.newResult(result, ground_node)
    
    def notifyComplete(self) :
        """Send the ``complete`` event to all listeners of this node."""
        
        EngineLogger.get().sendComplete(self)
        if not self.isComplete :
            self.isComplete = True
            for listener, evttype in self.listeners :
                if evttype & self.EVT_COMPLETE :
                    listener.complete()
        
    def addListener(self, listener, eventtype=EVT_ALL) :
        """Add the given listener."""
        # Add the listener such that it receives future events.
        EngineLogger.get().connect(self,listener,eventtype)
        self.listeners.append((listener,eventtype))
        
    def complete(self) :
        """Process a ``complete`` event.
        
        By default forwards this events to its listeners.
        """
        EngineLogger.get().receiveComplete(self)
        self.notifyComplete()
        
    def newResult(self, result, ground_node=0) :
        """Process a new result.
        
        :param result: context or list of arguments
        :param ground_node: node is ground program
        
        By default forwards this events to its listeners.
        """
        EngineLogger.get().receiveResult(self, result, ground_node)
        self.notifyListeners(result, ground_node)

class ProcessOr(ProcessNode) :
    """Process a disjunction of nodes.
    
    :param count: number of disjuncts
    :type count: int
    :param parent: listener
    :type parent: ProcessNode
    
    Behaviour:
    
        * This node forwards all results to its listeners.
        * This node sends a complete message after receiving ``count`` ``complete`` messages from its children.
        * If the node is initialized with ``count`` equal to 0, it sends out a ``complete`` signal immediately.
    
    """
    
    def __init__(self, count, parent) :
        ProcessNode.__init__(self)
        self._count = count
        self.addListener(parent)
        if self._count == 0 :
            self.notifyComplete()
    
    def complete(self) :
        EngineLogger.get().receiveComplete(self)
        self._count -= 1
        if self._count <= 0 :
            self.notifyComplete()
            
class ProcessNot(ProcessNode) :
    """Process a negation node.
    
    Behaviour:
    
        * This node buffers all ground nodes.
        * Upon receiving a ``complete`` event, sends a result and a ``complete`` signal.
        * The result is negation of an ``or`` node of its children. No result is send if the children are deterministically true.
        
    """
    
    def __init__(self, gp, context, parent) :
        ProcessNode.__init__(self)
        self.context = context
        self.ground_nodes = []
        self.gp = gp
        self.addListener(parent)
        
    def newResult(self, result, ground_node=0) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        if ground_node != None :
            self.ground_nodes.append(ground_node)
        
    def complete(self) :
        EngineLogger.get().receiveComplete(self)
        if self.ground_nodes :
            or_node = self.gp.addNot(self.gp.addOr( self.ground_nodes ))
            if or_node != None :
                self.notifyListeners(self.context, ground_node=or_node)
        else :
            self.notifyListeners(self.context, ground_node=0)
        self.notifyComplete()

class ProcessLink(ProcessNode) :
    """Links two calls in a conjunction."""
    
    
    def __init__(self, engine, db, gp, node_id, parent, define) :
        ProcessNode.__init__(self)
        self.engine = engine
        self.db = db
        self.gp = gp
        self.node_id = node_id
        self.parent = parent
        self.addListener(self.parent, ProcessNode.EVT_COMPLETE)
        self.define = define
        self.required_complete = 1
        
    def newResult(self, result, ground_node=0) :
        self.required_complete += 1     # For each result of first conjuct, we call the second conjuct which should produce a complete.
        EngineLogger.get().receiveResult(self, result, ground_node, 'required: %s' % self.required_complete)
        self.engine._exit_call( self.node_id, result )    
        process = ProcessAnd(self.gp, ground_node)
        process.addListener(self.parent, ProcessNode.EVT_RESULT)
        process.addListener(self, ProcessNode.EVT_COMPLETE) # Register self as listener for complete events.
        self.engine._eval( self.db, self.gp, self.node_id, self.engine._create_context(result,define=self.define), process)
        
    def complete(self) :
        # Receive complete
        EngineLogger.get().receiveComplete(self, 'required: %s' % (self.required_complete-1))
        self.required_complete -= 1
        if self.required_complete == 0 :
            self.notifyComplete()
        
                
class ProcessAnd(ProcessNode) :
    """Process a conjunction."""
    
    def __init__(self, gp, first_node ) :
        ProcessNode.__init__(self)
        self.gp = gp
        self.first_node = first_node
        
    def newResult(self, result, ground_node=0) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        and_node = self.gp.addAnd( (self.first_node, ground_node) )
        self.notifyListeners(result, and_node)
        
class ProcessDefineCycle(ProcessNode) :
    """Process a cyclic define (child)."""
    
    def __init__(self, parent, context, listener) :
        self.parent = parent
        
        context.propagateCyclic(self.parent)
        
        # while context != self.parent :
        #     context.cyclic = True
        #     context = context.parent
        ProcessNode.__init__(self)
        self.addListener(listener)
        self.parent.addListener(self)
        self.parent.cyclic = True
        self.parent.addCycleChild(self)
        
    
    def __repr__(self) :
        return 'cycle child of %s [%s]' % (self.parent, id(self))
        
     
class ProcessDefine(ProcessNode) :
    """Process a standard define (or cycle parent)."""
    
    def __init__(self, engine, db, gp, node_id, node, args, parent) :
        self.node = node
        self.results = {}
        self.engine = engine
        self.db = db
        self.gp = gp
        self.node_id = node_id
        self.args = args
        self.parents = set()
        if parent : self.parents.add(parent)
        self.__is_cyclic = False
        self.__buffer = defaultdict(list)
        self.children = []
        self.execute_completed = False
        ProcessNode.__init__(self)
        
    @property
    def cyclic(self) :
        return self.__is_cyclic
        
    @cyclic.setter
    def cyclic(self, value) :
        if self.__is_cyclic != value :
            self.__is_cyclic = value
            self._cycle_detected()
        
    def propagateCyclic(self, root) :
        if root != self :
            self.cyclic = True
            for p in self.parents :
                p.propagateCyclic(root)
            
    def addCycleChild(self, cnode ) :
        self.children.append(cnode)
        if self.execute_completed :
            cnode.complete()
        
    def addAncestor(self, parent) :
        self.parents.add(parent)
        
    def getAncestors(self) :
        current = {self}
        just_added = current
        while just_added :
            latest = set()
            for a in just_added :
                latest |= a.parents
            latest -= current
            current |= latest
            just_added = latest
        return current
        
    def hasAncestor(self, anc) :
        return anc in self.getAncestors()
                        
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
        
        process = ProcessOr( len(children), self)
        # Evaluate the children
        for child in children :
            self.engine._eval( self.db, self.gp, child, self.engine._create_context(self.args,define=self), parent=process )
        
        self.execute_completed = True
        for c in self.children :
            c.complete()

    
    def newResult(self, result, ground_node=0) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        if self.cyclic :
            self.newResultUnbuffered(result, ground_node)
        else :
            self.newResultBuffered(result, ground_node)
    
    def newResultBuffered(self, result, ground_node=0) :
        res = (tuple(result))
        self.__buffer[res].append( ground_node )
                        
    def newResultUnbuffered(self, result, ground_node=0) :
        res = (tuple(result))
        if res in self.results :
            res_node = self.results[res]
            self.gp.addDisjunct( res_node, ground_node )
        else :
            self.engine._exit_call( self.node, result )
            result_node = self.gp.addOr( (ground_node,), readonly=False )
            self.results[ res ] = result_node
            
            self.notifyListeners(result, result_node )
            
    def complete(self) :
        EngineLogger.get().receiveComplete(self)
        self._flush_buffer()
        self.notifyComplete()
    
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
            self.notifyListeners(result, node)
        self.__buffer.clear()
    
        
    def __repr__(self) :
        return '%s %s(%s)' % (id(self), self.node.functor, ', '.join(map(str,self.args)))
        
def extract_vars(*args, **kwd) :
    counter = kwd.get('counter', defaultdict(int))
    for arg in args :
        if type(arg) == int :
            counter[arg] += 1
        elif isinstance(arg,Term) :
            extract_vars(*arg.args, counter=counter)
    return counter
            
class ProcessBodyReturn(ProcessNode) :
    """Process the results of a clause body."""
    
    def __init__(self, head_args, node, node_id, parent) :
        ProcessNode.__init__(self)
        self.head_args = head_args
        self.head_vars = extract_vars(*self.head_args)
        self.node_id = node_id
        self.node = node
        self.addListener(parent)
                    
    def newResult(self, result, ground_node=0) :
        for i, res in enumerate(result) :
            if not is_ground(res) and self.head_vars[i] > 1 :
                raise VariableUnification()
        
        EngineLogger.get().receiveResult(self, result, ground_node)
        output = [ instantiate(arg, result) for arg in self.head_args ]
        self.notifyListeners(output, ground_node)
                
class ProcessCallReturn(ProcessNode) :
    """Process the results of a call."""
    
    def __init__(self, call_args, context, parent) :
        ProcessNode.__init__(self)
        self.call_args = call_args
        self.context = context
        self.addListener(parent)
                    
    def newResult(self, result, ground_node=0) :
        EngineLogger.get().receiveResult(self, result, ground_node)
        output = list(self.context)
        #try :
        for call_arg, res_arg in zip(self.call_args,result) :
            unify( res_arg, call_arg, output )
        self.notifyListeners(output, ground_node)

        # except UnifyError :
        #     pass
    

class ResultCollector(ProcessNode) :
    """Collect results."""
    
    def __init__(self) :
        ProcessNode.__init__(self)
        self.results = []
    
    def newResult( self, result, ground_result) :
        self.results.append( (ground_result, result  ))
        
    def complete(self) :
        pass

class PrologInstantiationError(Exception) : pass

class PrologTypeError(Exception) : pass


def builtin_true( context, callback, **kwdargs ) :
    """``true``"""
    callback.newResult(context)
    callback.complete()    

def builtin_fail( callback, **kwdargs ) :
    """``fail``"""
    callback.complete()

def builtin_eq( A, B, callback, **kwdargs ) :
    """``A = B``
        A and B not both variables
    """
    if A == None and B == None :
        raise VariableUnification()
    else :
        try :
            R = unify_value(A,B)
            callback.newResult( ( R, R ) )
        except UnifyError :
            pass
        callback.complete()

def builtin_neq( A, B, callback, **kwdargs ) :
    """``A \= B``
        A and B not both variables
    """
    if A == None and B == None :
        callback.complete() # FAIL
    else :
        try :
            R = unify_value(A,B)
        except UnifyError :
            callback.newResult( ( A, B ) )
        callback.complete()
            
def builtin_notsame( A, B, callback, **kwdargs ) :
    """``A \== B``"""
    if A == None and B == None :
        raise RuntimeError('Operation not supported!')  # TODO make this work
    else :
        # In Python A != B is not always the same as not A == B.
        if not A == B :
            callback.newResult( (A,B) )
        callback.complete()    

def builtin_same( A, B, callback, **kwdargs ) :
    """``A == B``"""
    if A == None and B == None :
        raise RuntimeError('Operation not supported!')  # TODO make this work
    else :
        if A == B :
            callback.newResult( (A,B) )
        callback.complete()    

def builtin_gt( A, B, callback, **kwdargs ) :
    """``A > B`` 
        A and B are ground
    """
    vA = A.value
    vB = B.value
    
    if (vA > vB) :
        callback.newResult( (A,B) )
    callback.complete()

def builtin_lt( A, B, callback, **kwdargs ) :
    """``A > B`` 
        A and B are ground
    """
    vA = A.value
    vB = B.value
    
    if (vA < vB) :
        callback.newResult( (A,B) )
    callback.complete()

def builtin_le( A, B, callback, **kwdargs ) :
    """``A =< B``
        A and B are ground
    """
    vA = A.value
    vB = B.value
    
    if (vA <= vB) :
        callback.newResult( (A,B) )
    callback.complete()

def builtin_ge( A, B, callback, **kwdargs ) :
    """``A >= B`` 
        A and B are ground
    """
    vA = A.value
    vB = B.value
    
    if (vA >= vB) :
        callback.newResult( (A,B) )
    callback.complete()

def builtin_val_neq( A, B, callback, **kwdargs ) :
    """``A =\= B`` 
        A and B are ground
    """
    vA = A.value
    vB = B.value
    
    if (vA != vB) :
        callback.newResult( (A,B) )
    callback.complete()

def builtin_val_eq( A, B, callback, **kwdargs ) :
    """``A =:= B`` 
        A and B are ground
    """
    vA = A.value
    vB = B.value
    
    if (vA == vB) :
        callback.newResult( (A,B) )
    callback.complete()

def builtin_is( A, B, callback, **kwdargs ) :
    """``A is B``
        B is ground
    """
    vB = B.value
    try :
        R = Constant(vB)
        unify_value(A,R)
        callback.newResult( (R,B) )
    except UnifyError :
        pass
    callback.complete()
    
def atom_to_filename(atom) :
    atom = str(atom)
    if atom[0] == atom[-1] == "'" :
        atom = atom[1:-1]
    return atom
    
    
def builtin_consult( filename, callback=None, database=None, engine=None, context=None, **kwdargs ) :
    
    filename = os.path.join(database.source_root, atom_to_filename( filename ))
    if not os.path.exists( filename ) :
        filename += '.pl'
    if not os.path.exists( filename ) :
        # TODO better exception
        raise Exception('File not found!')
    
    # Prevent loading the same file twice
    if not filename in database.source_files : 
        database.source_files.append(filename)
        pl = PrologFile( filename )
        for clause in pl :
            database += clause
    callback.newResult(context)
    callback.complete()

def is_var(term) :
    return is_variable(term) or term.isVar()

def is_term(term) :
    return not is_var(term) and not is_constant(term)

def is_float_pos(term) :
    return is_constant(term) and term.isFloat()

def is_float_neg(term) :
    return is_term(term) and term.arity == 1 and term.functor == "'-'" and is_float_pos(term.args[0])

def is_float(term) :
    return is_float_pos(term) or is_float_neg(term)

def is_integer_pos(term) :
    return is_constant(term) and term.isInteger()

def is_integer_neg(term) :
    return is_term(term) and term.arity == 1 and term.functor == "'-'" and is_integer_pos(term.args[0])

def is_integer(term) :
    return is_integer_pos(term) or is_integer_neg(term)

def is_string(term) :
    return is_constant(term) and term.isString()

def is_number(term) :
    return is_float(term) and is_integer(term)

def is_constant(term) :
    return not is_var(term) and term.isConstant()

def is_atom(term) :
    return is_term(term) and term.arity == 0

def is_rational(term) :
    return False

def is_dbref(term) :
    return False

def is_compound(term) :
    return is_term(term) and term.arity > 0
    
def is_list_maybe(term) :
    """Check whether the term looks like a list (i.e. of the form '.'(_,_))."""
    return is_compound(term) and term.functor == '.' and term.arity == 2
    
def is_list_nonempty(term) :
    if is_list_maybe(term) :
        tail = list_tail(term)
        return is_list_empty(tail) or is_var(tail)
    return False

def is_fixed_list(term) :
    return is_list_empty(term) or is_fixed_list_nonempty(term)

def is_fixed_list_nonempty(term) :
    if is_list_maybe(term) :
        tail = list_tail(term)
        return is_list_empty(tail)
    return False
    
def is_list_empty(term) :
    return is_atom(term) and term.functor == '[]'
    
def is_list(term) :
    return is_list_empty(term) or is_list_nonempty(term)
    
def is_compare(term) :
    return is_atom(term) and term.functor in ("'<'", "'='", "'>'")
    
def list_elements(term) :
    elements = []
    tail = term
    while is_list_maybe(tail) :
        elements.append(tail.args[0])
        tail = tail.args[1]
    return elements, tail
    
def list_tail(term) :
    tail = term
    while is_list_maybe(tail) :
        tail = tail.args[1]
    return tail
               

def builtin_split_call( term, parts, callback, **kwdargs) :
    """T =.. L"""
    # f(A,b,c) =.. [X,x,B,C] => f(x,b,c) =.. [f,x,b,c]
    try :
        if not is_var(term) :
            part_list = ( term.withArgs(), ) + term.args
            current = Term('[]')
            for t in reversed(part_list) :
                current = Term('.', t, current)
            L = unify_value(current, parts)
        else :
            L = parts

        # Now: 
        #   - if term == f(A,B,C) -> L = f(A,B,C)
        #   - if term is var -> L = parts (is_var of List)
        if is_var(L) :
            # Shouldn't be => ERROR
            raise InstantiationError("'=../2' expects ground functor.")
        elif not is_list(L) :
            # Not a list:
            raise UnifyError()
        else :
            elements, tail = list_elements(L)
            if not is_list_empty(tail) :
                raise InstantiationError("'=../2' expects fixed length list.")
            elif not elements :
                raise InstantiationError("'=../2' expects non-empty list.")
            elif not is_atom(elements[0]) :
                raise InstantiationError("'=../2' expects atom as functor.")
            else :
                term_new = elements[0](*elements[1:])
            T = unify_value(term, term_new)
        callback.newResult((T,L))
    except UnifyError :
        # Can't unify something -> FAIL.
        pass
    callback.complete()

def builtin_arg(index,term,argument,callback,**kwdargs) :
    if is_var(term) or is_var(index) :
        raise InstantiationError("'arg/3' expects ground arguments at position 1 and 2.")
    elif not is_integer(index) or int(index) < 0 :
        raise InstantiationError("'arg/3' expects a positive integer at position 1.")
    elif int(index) > 0 and int(index) <= len(term.args) :
        try :
            arg = term.args[int(index)-1]
            res = unify_value(arg,argument)
            callback.newResult(index,term,res)
        except UnifyError :
            pass
    else :
        # Index out of bounds -> fail silently
        pass
    callback.complete()

def builtin_functor(term,functor,arity,callback,**kwdargs) :
    if is_var(term) :
        if is_atom(functor) and is_integer(arity) and int(arity) >= 0 :
            callback.newResult( Term(functor, *((None,)*int(arity)) ), functor, arity )
        else :
            raise InstantiationError("'functor/3' received unexpected arguments")
    else :
        try :
            func_out = unify_value(functor, Term(term.functor))
            arity_out = unify_value(arity, Constant(term.arity))
            callback.newResult(term, func_out, arity_out)
        except UnifyError :
            pass
    callback.complete()

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


def builtin_call( term, args=(), callback=None, database=None, engine=None, context=None, ground_program=None, **kwdargs ) :
    if not is_term(term) :
        raise InstantiationError("'call/1' expects a callable.")
    else :
        # Find the define node for the given query term.
        clause_node = database.find(term.withArgs( *(term.args+args)))
        # If term not defined: raise error
        if clause_node == None : raise UnknownClause('%s/%s' % (term.functor, len(term.args)))
        # Create a new call.
        call_node = ClauseDB._call( term.functor, range(0, len(term.args) + len(args)), clause_node )
        # Create a callback node that wraps the results in the functor.
        cb = CallProcessNode(term, args, callback)
        # Evaluate call.
        engine._eval_call(database, ground_program, None, call_node, engine._create_context(term.args+args,define=context.define), cb )        

def builtin_callN( term, *args, **kwdargs ) :
    return builtin_call(term, args, **kwdargs)

class StructSort(object) :
    
    def __init__(self, obj, *args):
        self.obj = obj
    def __lt__(self, other):
        return struct_cmp(self.obj, other.obj) < 0
    def __gt__(self, other):
        return struct_cmp(self.obj, other.obj) > 0
    def __eq__(self, other):
        return struct_cmp(self.obj, other.obj) == 0
    def __le__(self, other):
        return struct_cmp(self.obj, other.obj) <= 0  
    def __ge__(self, other):
        return struct_cmp(self.obj, other.obj) >= 0
    def __ne__(self, other):
        return struct_cmp(self.obj, other.obj) != 0


mode_types = {
    'i' : ('integer', is_integer),
    'I' : ('positive_integer', is_integer_pos),
    'v' : ('var', is_var),
    'l' : ('list', is_list),
    'L' : ('fixed_list', is_fixed_list),    # List of fixed length (i.e. tail is [])
    '*' : ('any', lambda x : True ),
    '<' : ('compare', is_compare )
}


class CallModeError(Exception) :
    
    def __init__(self, functor, args, accepted) :
        self.scope = '%s/%s'  % ( functor, len(args) )
        self.received = ', '.join(map(self.show_arg,args))
        self.expected = [  ', '.join(map(self.show_mode,mode)) for mode in accepted  ]
        message = 'Invalid argument types for call'
        if self.scope : message += " to '%s'" % self.scope
        message += ': arguments: (%s)' % self.received
        message += ', expected: (%s)' % ') or ('.join(self.expected) 
        Exception.__init__(self, message)
        
    def show_arg(self, x) :
        if x == None :
            return '_'
        else :
            return str(x)
    
    def show_mode(self, t) :
        return mode_types[t][0]
        
class BooleanBuiltIn(object) :
    """Simple builtin that consist of a check without unification. (e.g. var(X), integer(X), ... )."""
    
    def __init__(self, base_function) :
        self.base_function = base_function
    
    def __call__( self, *args, **kwdargs ) :
        callback = kwdargs.get('callback')
        if self.base_function(*args) :
            callback.newResult(args)
        callback.complete()
        
class SimpleBuiltIn(object) :
    """Simple builtin that does cannot be involved in a cycle or require engine information and has 0 or more results."""

    def __init__(self, base_function) :
        self.base_function = base_function
    
    def __call__(self, *args, **kwdargs ) :
        callback = kwdargs.get('callback')
        results = self.base_function(*args)
        if results :
            for result in results :
                callback.newResult(result)
        callback.complete()
    
def check_mode( args, accepted, functor=None ) :
    for i, mode in enumerate(accepted) :
        correct = True
        for a,t in zip(args,mode) :
            name, test = mode_types[t]
            if not test(a) : 
                correct = False
                break
        if correct : return i
    raise CallModeError(functor, args, accepted)


@BooleanBuiltIn
def builtin_var( term ) :
    return is_var(term)

@BooleanBuiltIn
def builtin_atom( term ) :
    return is_atom(term)

@BooleanBuiltIn
def builtin_atomic( term ) :
    return is_atom(term) or is_number(term)

@BooleanBuiltIn
def builtin_compound( term ) :
    return is_compound(term)

@BooleanBuiltIn
def builtin_float( term ) :
    return is_float(term)

@BooleanBuiltIn
def builtin_integer( term ) :
    return is_integer(term)

@BooleanBuiltIn
def builtin_nonvar( term ) :
    return not is_var(term)

@BooleanBuiltIn
def builtin_number( term ) :
    return is_number(term) 

@BooleanBuiltIn
def builtin_simple( term ) :
    return is_var(term) or is_atomic(term)
    
@BooleanBuiltIn
def builtin_callable( term ) :
    return is_term(term)

@BooleanBuiltIn
def builtin_rational( term ) :
    return is_rational(term)

@BooleanBuiltIn
def builtin_dbreference( term ) :
    return is_dbref(term)  
    
@BooleanBuiltIn
def builtin_primitive( term ) :
    return is_atomic(term) or is_dbref(term)

@BooleanBuiltIn
def builtin_ground( term ) :
    return is_ground(term)

@BooleanBuiltIn
def builtin_is_list( term ) :
    return is_list(term)

def compare(a,b) :
    if a < b :
        return -1
    elif a > b :
        return 1
    else :
        return 0
    
def struct_cmp( A, B ) :
    # Note: structural comparison
    # 1) Var < Num < Str < Atom < Compound
    # 2) Var by address
    # 3) Number by value, if == between int and float => float is smaller (iso prolog: Float always < Integer )
    # 4) String alphabetical
    # 5) Atoms alphabetical
    # 6) Compound: arity / functor / arguments
        
    # 1) Variables are smallest
    if is_var(A) :
        if is_var(B) :
            # 2) Variable by address
            return compare(A,B)
        else :
            return -1
    elif is_var(B) :
        return 1
    # assert( not is_var(A) and not is_var(B) )
    
    # 2) Numbers are second smallest
    if is_number(A) :
        if is_number(B) :
            # Just compare numbers on float value
            res = compare(float(A),float(B))
            if res == 0 :
                # If the same, float is smaller.
                if is_float(A) and is_integer(B) : 
                    return -1
                elif is_float(B) and is_integer(A) : 
                    return 1
                else :
                    return 0
        else :
            return -1
    elif is_number(B) :
        return 1
        
    # 3) Strings are third
    if is_string(A) :
        if is_string(B) :
            return compare(str(A),str(B))
        else :
            return -1
    elif is_string(B) :
        return 1
    
    # 4) Atoms / terms come next
    # 4.1) By arity
    res = compare(A.arity,B.arity)
    if res != 0 : return res
    
    # 4.2) By functor
    res = compare(A.functor,B.functor)
    if res != 0 : return res
    
    # 4.3) By arguments (recursively)
    for a,b in zip(A.args,B.args) :
        res = struct_cmp(a,b)
        if res != 0 : return res
        
    return 0

@BooleanBuiltIn    
def builtin_struct_lt(A, B) :
    return struct_cmp(A,B) < 0    

@BooleanBuiltIn    
def builtin_struct_le(A, B) :
    return struct_cmp(A,B) <= 0

@BooleanBuiltIn    
def builtin_struct_gt(A, B) :
    return struct_cmp(A,B) > 0

@BooleanBuiltIn    
def builtin_struct_ge(A, B) :
    return struct_cmp(A,B) >= 0

@SimpleBuiltIn
def builtin_compare(C, A, B) :
    mode = check_mode( (C,A,B), [ '<**', 'v**' ], functor='compare')
    compares = "'>'","'='","'<'" 
    c = struct_cmp(A,B)
    c_token = compares[1-c]
    
    if mode == 0 : # Given compare
        if c_token == C.functor : return [ (C,A,B) ]
    else :  # Unknown compare
        return [ (Term(c_token), A, B ) ]
    
# numbervars(T,+N1,-Nn)    number the variables TBD?

def build_list(elements, tail) :
    current = tail
    for el in reversed(elements) :
        current = Term('.', el, current)
    return current

@SimpleBuiltIn
def builtin_length(L, N) :
    mode = check_mode( (L,N), [ 'LI', 'Lv', 'lI', 'vI' ], functor='length')
    # Note that Prolog also accepts 'vv' and 'lv', but these are unbounded.
    # Note that lI is a subset of LI, but only first matching mode is returned.
    if mode == 0 or mode == 1 :  # Given fixed list and maybe length
        elements, tail = list_elements(L)
        list_size = len(elements)
        try :
            N = unify_value(N, Constant(list_size))
            return [ ( L, N ) ]
        except UnifyError :
            return []    
    else :    # Unbounded list or variable list and fixed length.
        if mode == 2 :
            elements, tail = list_elements(L)
        else :
            elements, tail = [], L
        remain = int(N) - len(elements)
        if remain < 0 :
            raise UnifyError()
        else :
            extra = [None] * remain
        newL = build_list( elements + extra, Term('[]'))
        return [ (newL, N)]


@SimpleBuiltIn
def builtin_sort( L, S ) :
    # TODO doesn't work properly with variables e.g. gives sort([X,Y,Y],[_]) should be sort([X,Y,Y],[X,Y])
    mode = check_mode( (L,S), [ 'L*' ], functor='sort' )
    elements, tail = list_elements(L)  
    # assert( is_list_empty(tail) )
    try :
        sorted_list = build_list(sorted(set(elements), key=StructSort), Term('[]'))
        S_out = unify_value(S,sorted_list)
        return [(L,S_out)]
    except UnifyError :
        return []

@SimpleBuiltIn
def builtin_between( low, high, value ) :
    mode = check_mode((low,high,value), [ 'iii', 'iiv' ], functor='between')
    low_v = int(low)
    high_v = int(high)
    if mode == 0 : # Check    
        value_v = int(value)
        if low_v <= value_v <= high_v :
            return [(low,high,value)]
    else : # Enumerate
        results = []
        for value_v in range(low_v, high_v+1) :
            results.append( (low,high,Constant(value_v)) ) 
        return results

@SimpleBuiltIn
def builtin_succ( a, b ) :
    mode = check_mode((a,b), [ 'vI', 'Iv', 'II' ], functor='succ')
    if mode == 0 :
        b_v = int(b)
        return [(Constant(b_v-1), b)]
    elif mode == 1 :
        a_v = int(a)
        return [(a, Constant(a_v+1))]
    else :
        a_v = int(a)
        b_v = int(b)
        if b_v == a_v + 1 :
            return [(a, b)]
    return []

@SimpleBuiltIn
def builtin_plus( a, b, c ) :
    mode = check_mode((a,b,c), [ 'iii', 'iiv', 'ivi', 'vii' ], functor='plus')
    if mode == 0 :
        a_v = int(a)
        b_v = int(b)
        c_v = int(c)
        if a_v + b_v == c_v :
            return [(a,b,c)]
    elif mode == 1 :
        a_v = int(a)
        b_v = int(b)
        return [(a, b, Constant(a_v+b_v))]
    elif mode == 2 :
        a_v = int(a)
        c_v = int(c)
        return [(a, Constant(c_v-a_v), c)]
    else :
        b_v = int(b)
        c_v = int(c)
        return [(Constant(c_v-b_v), b, c)]
    return []


def addBuiltins(engine) :
    """Add Prolog builtins to the given engine."""
    
    # Shortcut some wrappers
    
    engine.addBuiltIn('true', 0, builtin_true)
    engine.addBuiltIn('fail', 0, builtin_fail)
    engine.addBuiltIn('false', 0, builtin_fail)

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

    engine.addBuiltIn('var', 1, builtin_var)
    engine.addBuiltIn('atom', 1, builtin_atom)
    engine.addBuiltIn('atomic', 1, builtin_atomic)
    engine.addBuiltIn('compound', 1, builtin_compound)
    engine.addBuiltIn('float', 1, builtin_float)
    engine.addBuiltIn('rational', 1, builtin_rational)
    engine.addBuiltIn('integer', 1, builtin_integer)
    engine.addBuiltIn('nonvar', 1, builtin_nonvar)
    engine.addBuiltIn('number', 1, builtin_number)
    engine.addBuiltIn('simple', 1, builtin_simple)
    engine.addBuiltIn('callable', 1, builtin_callable)
    engine.addBuiltIn('dbreference', 1, builtin_dbreference)
    engine.addBuiltIn('primitive', 1, builtin_primitive)
    engine.addBuiltIn('ground', 1, builtin_ground)
    engine.addBuiltIn('is_list', 1, builtin_is_list)
    
    engine.addBuiltIn('=..', 2, builtin_split_call)
    engine.addBuiltIn('arg', 3, builtin_arg)
    engine.addBuiltIn('functor', 3, builtin_functor)
    
    engine.addBuiltIn('@>',2, builtin_struct_gt)
    engine.addBuiltIn('@<',2, builtin_struct_lt)
    engine.addBuiltIn('@>=',2, builtin_struct_ge)
    engine.addBuiltIn('@=<',2, builtin_struct_le)
    engine.addBuiltIn('compare',3, builtin_compare)

    engine.addBuiltIn('length',2, builtin_length)

    engine.addBuiltIn('sort',2, builtin_sort)
    engine.addBuiltIn('between', 3, builtin_between)
    engine.addBuiltIn('succ',2, builtin_succ)
    engine.addBuiltIn('plus',3, builtin_plus)

    # These are special builtins
    engine.addBuiltIn('call', 1, builtin_call)
    for i in range(2,10) :
        engine.addBuiltIn('call', i, builtin_callN)
    engine.addBuiltIn('consult', 1, builtin_consult)



DefaultEngine = EventBasedEngine


class UserAbort(Exception) : pass

class UserFail(Exception) : pass

class NonGroundQuery(Exception) : pass

class UnboundProgramError(Exception) : pass

class CallStackError(Exception) : pass


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

class EngineLogger(object) :
    """Logger for engine messaging."""

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
        print (type(source).__name__, id(source), 'receive', result, node, source, *extra)
        
    def receiveComplete(self, source, *extra) :
        print (type(source).__name__, id(source), 'receive complete', source, *extra)
        
    def sendResult(self, source, result, node, *extra) :
        print (type(source).__name__, id(source), 'send', result, node, source, *extra)

    def sendComplete(self, source, *extra) :
        print (type(source).__name__, id(source), 'send complete', source, *extra)
        
    def create(self, source) :
        print (type(source).__name__, id(source), 'create', source)
        
    def connect(self, source, listener, evt_type) :
        print (type(source).__name__, id(source), 'connect', type(listener).__name__, id(listener))
