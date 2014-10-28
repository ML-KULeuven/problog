from __future__ import print_function

# Input python 2 and 3 compatible input
try:
    input = raw_input
except NameError:
    pass

from .program import ClauseDB
from .unification import TermDB, UnifyError

from collections import defaultdict, namedtuple
     
class CycleDetected(Exception) : pass
        
class BaseEngine(object) :
    
    def __init__(self) :
        pass
                
    def addBuiltIn(self, sig, func) :
        pass
        
    def getBuiltIns(self) :
        pass
    
    def query(self, db, term, level=0) :
        pass
    
    def ground(self, db, term, gp=None, level=0) :
        pass
        
class DefaultEngine(BaseEngine) :
    """Standard grounding engine."""
    
    def __init__(self, debugger = None) :
        self.__debugger = debugger
        
        self.__builtin_index = {}
        self.__builtins = []
        
    def _getBuiltIn(self, index) :
        real_index = -(index + 1)
        return self.__builtins[real_index]
        
    def addBuiltIn(self, sig, func) :
        self.__builtin_index[sig] = -(len(self.__builtin_index) + 1)
        self.__builtins.append( func )
        
    def getBuiltIns(self) :
        return self.__builtin_index
        
    def _enter_call(self, *args) :
        if self.__debugger != None :
            self.__debugger.enter(*args)
        
    def _exit_call(self, *args) :
        if self.__debugger != None :
            self.__debugger.exit(*args)

    def query(self, db, term, level=0) :
        gp = DummyGroundProgram()
        gp, result = self.ground(db, term, gp, level)
        
        return [ y for x,y in result ]        
    
    def ground(self, db, term, gp=None, level=0) :
        db = ClauseDB.createFrom(db, builtins=self.getBuiltIns())
        
        if gp == None :
            gp = GroundProgram()
        
        tdb = TermDB()
        args = [ tdb.add(x) for x in term.args ]
        clause_node = db.find(term)
        
        if clause_node == None : return gp, []
        
        call_node = ClauseDB._call( term.functor, args, clause_node )
        
        query = self._ground_call(db, gp, None, call_node, args, tdb, anc=[], call_key=None, level=level)
        return gp, query
    
        
    def _ground( self, db, gp, node_id, input_vars, tdb, level=0, **extra) :
        node = db.getNode(node_id)
          
        call_terms = [ tdb[arg] for arg in input_vars ] 
                
        try :
            try :
                functor = node.functor
            except AttributeError :
                functor = str(node)
            self._enter_call(level, '%s: %s' % (node_id,functor), call_terms)
        except UserFail :
            self._exit_call(level, '%s: %s' % (node_id,functor), call_terms, 'USER')
            return ()
            
        
        nodetype = type(node).__name__
        rV = None
        if not node :
            raise _UnknownClause()
        elif nodetype == 'fact' :
            func = self._ground_fact     
        elif nodetype == 'call' :
            func = self._ground_call
        elif nodetype == 'clause' :
            func = self._ground_clause
        elif nodetype == 'neg' :
            func = self._ground_not
        elif nodetype == 'conj' :
            func = self._ground_and
        elif nodetype == 'disj' :
            func = self._ground_or
        elif nodetype == 'define':
            func = self._ground_define
        elif nodetype == 'choice' :
            func = self._ground_choice
        else :
            raise NotImplementedError("Unknown node type: '%s'" % nodetype )
        
        try :
            rV = func(db, gp, node_id, node, input_vars, tdb, level=level+1, **extra )
            self._exit_call(level, '%s: %s' % (node_id,functor), call_terms, rV)
        except CycleDetected :
            self._exit_call(level, '%s: %s' % (node_id,functor), call_terms, 'CYCLE')
            rV = ()

        return rV

        
    def _ground_fact(self, db, gp, node_id, node, local_vars, tdb, **extra) :
        with tdb :
            atoms = list(map( tdb.add, node.args))
            
            # Match call arguments with fact arguments
            for a,b in zip( local_vars, atoms ) :
                tdb.unify(a,b)
                
            args = list(map(tdb.reduce, local_vars))
            node = gp.addFact( node.functor, args, node.probability )
            return [ (node, args) ]
        return []

    def _ground_choice(self, db, gp, node_id, node, local_vars, tdb, **extra) :
        with tdb :
            atoms = list(map( tdb.add, node.args))
            
            # Match call arguments with fact arguments
            for a,b in zip( local_vars, atoms ) :
                tdb.unify(a,b)
                
            args = tuple(map(tdb.reduce, local_vars))
            origin = (node.group, args)
            node = gp.addChoice( origin, node.choice  ,node.probability )
            return [ (node, args) ]
        return []

                    
    def _ground_and(self, db, gp, node_id, node, local_vars, tdb, **extra) :
        with tdb :
            result1 = self._ground(db, gp, node.children[0], local_vars, tdb, **extra)
            
            result = []
            for node1, res1 in result1 :
                with tdb :
                    for a,b in zip(res1, local_vars) :
                        tdb.unify(a,b)
                    result2 = self._ground(db, gp, node.children[1], local_vars, tdb, **extra)     
                    for node2, res2 in result2 :
                        result.append( ( gp.addAndNode( (node1, node2) ), res2 ) )
                    
            return result
    
    def _ground_or(self, db, gp, node_id, node, local_vars, tdb, **extra) :
        result = []
        for n in node.children :
            result += self._ground(db, gp, n, local_vars, tdb, **extra) 
        return result
            
    def _ground_call(self, db, gp, node_id, node, local_vars, tdb, level=0,**extra) :
        with tdb :
            call_args = [ tdb.add(arg) for arg in node.args ]
            
            if node.defnode < 0 :
                builtin = self._getBuiltIn( node.defnode )
                try :
                    self._enter_call(level, node.functor, node.args)
                except UserFail :
                    self._exit_call(level, node.functor, node.args, 'USER')
                    return ()
                
                sub = builtin( engine=self, clausedb=db, args=call_args, tdb=tdb, functor=node.functor, arity=len(node.args), level=level, **extra)
                
                sub = [ (0, s) for s in sub]
                self._exit_call(level, node.functor, node.args, sub)
            else :
                try :
                    sub = self._ground(db, gp, node.defnode, call_args, tdb, level=level, **extra)
                except _UnknownClause :
                    raise UnknownClause(node)
            # result at 
            result = []
            for node, res in sub :
                with tdb :
                    for a,b in zip(res,call_args) :
                        tdb.unify(a,b)
                    result.append( ( node, [ tdb.getTerm(arg) for arg in local_vars ] ) )
            return result
    
            
    def _ground_not(self, db, gp, node_id, node, local_vars, tdb, **extra) :
        subnode = node.child
        subresult = self._ground(db, gp, subnode, local_vars, tdb, **extra)
        
        if subresult :
            return [ (gp.negate(n),r) for n,r in subresult if n != gp.TRUE ]
        else :
            return [ (gp.TRUE, [ tdb[v] for v in local_vars ]) ]
                        
    def _ground_clause(self, db, gp, node_id, node, args, tdb, **extra) :
        new_context = self._create_context( tdb, node.varcount, args, node.args )
        if new_context == None : return []
        
        new_tdb, new_vars, new_args = new_context
        
        sub = self._ground(db, gp, node.child, new_vars, new_tdb, **extra)
            
        # sub contains values of local variables in the new context 
        #  => we need to translate them back to the 'args' from the old context
        result = []
        for node_id, res in sub :
            res_new = self._integrate_context(tdb, res, *new_context)
            if res_new != None :
                result.append( (node_id, res_new ) )
        return result
            
    def _ground_define(self, db, gp, node_id, node, args, tdb, **extra) :
        if type(node.children) == list :
            children = node.children            
        else :
            children = node.children.find( [ tdb[arg] for arg in args ] )
        
        results = []
        for n in children :
            results += self._ground(db, gp, n, args, tdb, **extra) 
                
        # - All entries should be ground.
        # - All entries should be grouped by same facts => create or-nodes.
        
        groups = defaultdict(list)
        for node, result in results :
            # for res in result :
            #     if (not res.isGround()) :
            #         raise NonGroundQuery()
            groups[ tuple(result) ].append(node)
        
        results = []
        for result, nodes in groups.items() :
            if len(nodes) > 1 :
                node = gp.addOrNode( nodes )
            else :
                node = nodes[0]
            results.append( (node,result) )
        return results
        
    def _create_context( self, tdb, varcount, call_args, def_args ) :
        new_tdb = TermDB()
        
        # Reserve spaces for clause variables
        new_vars = range(0,varcount)
        for i in new_vars :
            new_tdb.newVar()
        
        # Add call arguments to local tdb
        new_args = tdb.copyTo( new_tdb, *call_args )
                
        try :
            # Unify call arguments with head arguments
            for call_arg, def_arg in zip(new_args, def_args) :
                new_tdb.unify(call_arg, def_arg)
            return new_tdb, new_vars, new_args
        except UnifyError :
            return None
        
    def _integrate_context( self, tdb, result, new_tdb, new_vars, new_args ) :
        with tdb :
            # Unify local_vars with res
            with new_tdb :
                for a,b in zip(result, new_vars) :
                    new_tdb.unify(a,b)
                    
                return [ tdb.getTerm(x) for x in new_tdb.copyTo(tdb, *new_args) ]
        return None

class CycleFreeEngine(DefaultEngine):
    """Grounding engine with cycle detection."""
    
    def __init__(self, **kwd) :
        DefaultEngine.__init__(self, **kwd)
    
    def _ground_define(self, db, gp, node_id, node, args, tdb, anc=[], call_key=None, level=0) :
        call_key_1 = []
        call_terms = []
        for arg in args :
            call_terms.append(tdb[arg])
            if not tdb.isGround(arg) :
                call_key_1.append(None)
            else :
                call_key_1.append( str(tdb.getTerm(arg)) )
        call_key = (node_id, tuple(call_key_1) )
        
        if call_key != None and call_key in anc :
            raise CycleDetected()

        return DefaultEngine._ground_define(self, db, gp, node_id, node, args, tdb, anc=anc+[call_key], call_key=call_key, level=level)


class TabledEngine(DefaultEngine) :
    """Grounding engine with tabling.
    
    .. todo::
        
        Reset table.
    
    """
    
    def __init__(self, **kwd) :
        DefaultEngine.__init__(self, **kwd)
        # The table of stored results.
        self.__table = {}
        # Set of calls that should not be grounded (because they are cyclic, and depend on call stack).
        self.__do_not_ground = set()
    
    def _ground_define(self, db, gp, node_id, node, args, tdb, anc=[], **extra) :
        # Compute the key for this definition call (defnode + arguments).
        call_key_1 = []
        for arg in args :
            if not tdb.isGround(arg) :
                call_key_1.append(None)
            else :
                call_key_1.append( tdb.getTerm(arg) )
        call_key = (node_id, tuple(call_key_1) )

        # Get the record from the table.
        record = self.__table.get( call_key )

        if record == '#GROUNDING#' :
            # This call is currently being grounded. This means we have detected a cycle.
            # Mark all ancestors up to the cyclic one as 'do not ground'.
            for key in reversed(anc) :
                self.__do_not_ground.add(key)
                if key == call_key : break
            # Signal cycle detection (same as failing).
            raise CycleDetected()
        elif record == None or call_key in self.__do_not_ground :
            # The call has not been grounded yet or was marked as 'do not ground'.
            # Initialize ground record (for cycle detection)
            self.__table[call_key] = '#GROUNDING#'     
            # Compute the result using the default engine.
            result = DefaultEngine._ground_define(self, db, gp, node_id, node, args, tdb, anc=anc+[call_key], **extra)
            # Store the result in the table.
            self.__table[call_key] = result
            # Return the result.
            return result
        else :
            # The call was grounded before: return the stored record.
            return record

class DummyGroundProgram(object) :
    
    def __init__(self) :
        pass
        
    def getFact(self, name) :
        return 0
        
    def addChoice(self, origin, choice, probability) :
        return 0
        
    def negate(self, t) :
        return None

    def addFact(self, functor, args, probability) :
        return 0
    
    def reserveNode(self) :
        return 0
        
    def addNode(self, nodetype, content) :
        return 0
        
    def addOrNode(self, content, index=None) :
        return 0
        
    def addAndNode(self, content, index=None) :
        if (None in content) :
            return None
        else :
            return 0
    
# Taken and modified from from ProbFOIL
class GroundProgram(object) :
    """Represents and AND-OR graph."""
    
    _fact = namedtuple('fact', ('functor', 'args', 'probability') )
    _conj = namedtuple('conj', ('children') )
    _disj = namedtuple('disj', ('children') )
    _choice = namedtuple('choice', ('origin', 'choice', 'probability'))
    _alias = namedtuple('alias', ('alias') )
    
    # Invariant: stored nodes do not have TRUE or FALSE in their content.
    
    TRUE = 0
    FALSE = None
    
    def __init__(self, parent=None, compress=True) :
        self.__nodes = []
        self.__fact_names = {}
        self.__nodes_by_content = {}
        
        self.__compress = compress
        self.__choice_nodes = {}
        if parent != None :
            self.__offset = len(parent)
        else :
            self.__offset = 0
        self.__parent = parent
        
        self.__cyclefree = True
        
    def isAcyclic(self) :
        return self.__cyclefree
        
    def setCyclic(self) :
        self.__cyclefree = False
        
    def getFact(self, name) :
        return self.__fact_names.get(name, None)
        
    def addChoice(self, origin, choice, probability) :
        node_id = self.__choice_nodes.get((origin,choice))
        if node_id == None :
            node_id = self._addNode( self._choice( origin, choice, probability ))
            self.__choice_nodes[(origin,choice)] = node_id
        return node_id
                        
    def negate(self, t) :
        if t == self.TRUE :
            return self.FALSE
        elif t == self.FALSE :
            return self.TRUE
        else :
            return -t
                    
    def addFact(self, functor, args, probability) :
        """Add a named fact to the grounding."""
        
        name = (functor, tuple(args))
        
        node_id = self.getFact(name)
        if node_id == None : # Fact doesn't exist yet
            if probability == None and self.__compress :
                node_id = 0
            else :
                node_id = self._addNode( self._fact( functor, tuple(args), probability  ) )
            self.__fact_names[name] = node_id
        return node_id
        
    def addNode(self, nodetype, content) :
        if nodetype == 'or' :
            return self.addOrNode(content)
        elif nodetype == 'and' :
            return self.addAndNode(content)
        else :
            raise Exception("Unknown node type '%s'" % nodetype)
        
    def addOrNode(self, content, index=None) :
        """Add an OR node."""
        node_id = self._addCompoundNode('or', content, self.TRUE, self.FALSE, index=None)
        return self._addAlias(node_id, index)
        
    def addAndNode(self, content, index=None) :
        """Add an AND node."""
        node_id =  self._addCompoundNode('and', content, self.FALSE, self.TRUE, index=None)
        return self._addAlias(node_id, index)
        
    def _addAlias( self, node_id, index ) :
        if index == None or index == node_id :
            return node_id
        else :
            self._setNode( index, self._alias(node_id) )
            return node_id
        
    def _addCompoundNode(self, nodetype, content, t, f, index=None) :
        assert( content )   # Content should not be empty
        
        # If there is a t node, (true for OR, false for AND)
        if t in content : return t
        
        # Eliminate unneeded node nodes (false for OR, true for AND)
        content = filter( lambda x : x != f, content )

        # Put into fixed order and eliminate duplicate nodes
        content = tuple(sorted(set(content)))
        
        # Empty OR node fails, AND node is true
        if not content : return f
                
        # Contains opposites: return 'TRUE' for or, 'FALSE' for and
        if len(set(content)) > len(set(map(abs,content))) : return t
            
        # If node has only one child, just return the child.
        if len(content) == 1 : return content[0]
        
        # Lookup node for reuse
        key = (nodetype, content)
        node_id = self.__nodes_by_content.get(key, None)
        
        if node_id == None :    
            # Node doesn't exist yet
            if nodetype == 'or' :
                node = self._disj(content)
            else :
                node = self._conj(content)
            
            if index != None :
                self._setNode( index, node )
                node_id = index
            else :
                node_id = self._addNode( node )
            self.__nodes_by_content[ key ] = node_id
        return node_id
        
    def _addNode(self, node) :
        node_id = len(self) + 1
        self.__nodes.append( node )
        return node_id

    def _setNode(self, index, node) :
        self.__nodes[index-1] = node
        
    def reserveNode(self) :
        return self._addNode(None)
        
    def getNode(self, index) :
        assert (index != None and index > 0)
        if index <= self.__offset :
            return self.__parent.getNode(index)
        else :
            return self.__nodes[index-self.__offset-1]
                
        
    def _selectNodes(self, queries, node_selection) :
        for q in queries :
            node_id = q
            if node_id :
                self._selectNode(abs(node_id), node_selection)
        
    def _selectNode(self, node_id, node_selection) :
        assert(node_id != 0)
        if not node_selection[node_id-1] :
            node_selection[node_id-1] = True
            nodetype, content = self.getNode(node_id)
            
            if nodetype in ('and','or') :
                for subnode in content :
                    if subnode :
                        self._selectNode(abs(subnode), node_selection)
        
    def __len__(self) :
        return len(self.__nodes) + self.__offset
                
    def toCNF(self, queries=None) :
        # if self.hasCycle :
        #     raise NotImplementedError('The dependency graph contains a cycle!')
        
        if queries != None :
            node_selection = [False] * len(self)    # selection table
            self._selectNodes(queries, node_selection)
        else :
            node_selection = [True] * len(self)    # selection table
        
        choices = defaultdict(list)
        sums = defaultdict(float)
        lines = []
        facts = {}
        for index, sel in enumerate( node_selection ) :
          if sel :
            index += 1
            node = self.getNode(index)
            nodetype = type(node).__name__
            if nodetype == 'choice' :
                choices[node.origin].append(index)
                sums[node.origin] += node.probability.value
                facts[index] = (node.probability.value, 1.0)
            elif nodetype == 'fact' :                
                facts[index] = (node.probability.value, 1.0-node.probability.value)
            elif nodetype == 'conj' :
                line = str(index) + ' ' + ' '.join( map( lambda x : str(-(x)), node.children ) ) + ' 0'
                lines.append(line)
                for x in node.children  :
                    lines.append( "%s %s 0" % (-index, x) )
            elif nodetype == 'disj' :
                line = str(-index) + ' ' + ' '.join( map( lambda x : str(x), node.children ) ) + ' 0'
                lines.append(line)
                for x in node.children  :
                    lines.append( "%s %s 0" % (index, -x) )
                # lines.append('')
            else :
                raise ValueError("Unknown node type!")
        
        #choices = [ v for v in choices.values() if len(v) > 1 ]
        
        atom_count = len(node_selection)
        for k, s in choices.items() :
            if sums[k] < 1.0-1e-6 :
                facts[atom_count+1] = (1.0 - sums[k], 1.0)
                s.append(atom_count+1)
                atom_count += 1
            for i, a in enumerate(s) :
                for b in s[i+1:] :
                     lines.append('-%s -%s 0' % (a,b))
            lines.append(' '.join(map(str,s + [0]))) 
                        
        clause_count = len(lines)
        
        return [ 'p cnf %s %s' % (atom_count, clause_count) ] + lines, facts, choices
                
    def __str__(self) :
        s =  '\n'.join('%s: %s' % (i+1,n) for i, n in enumerate(self.__nodes))   
        s += '\n' + str(self.__fact_names)
        return s
    
    def toDot(self, queries=[], with_facts=False) :
        
        clusters = defaultdict(list)
        
        negative = set([])
        
        
        s = 'digraph GP {'
        for index, node in enumerate(self.__nodes) :
            index += 1
            nodetype = type(node).__name__
                        
            if nodetype == 'conj' :
                s += '%s [label="AND", shape="box"];\n' % (index)
                for c in node.children :
                    if c < 0 and not c in negative :
                        s += '%s [label="NOT"];' % (c)
                        s += '%s -> %s;' % (c,-c)
                        negative.add(c)
                    if c != 0 :
                        s += '%s -> %s;\n' % (index,c)
            elif nodetype == 'disj' :
                s += '%s [label="OR", shape="diamond"];\n' % (index)
                for c in node.children :
                    if c < 0 and not c in negative :
                        s += '%s [label="NOT"];\n' % (c)
                        s += '%s -> %s;\n' % (c,-c)
                        negative.add(c)
                    if c != 0 :
                        s += '%s -> %s;\n' % (index,c)
            elif nodetype == 'fact' :
                s += '%s [label="%s", shape="circle"];\n' % (index, node.probability)
                        #, node.functor, ', '.join(map(str,node.args)))
            elif nodetype == 'choice' :
                clusters[node.origin].append('%s [ shape="circle", label="%s" ];' % (index, node.probability))
            else :
                raise ValueError()
        
        c = 0
        for cluster, text in clusters.items() :
            s += 'subgraph cluster_%s { style="dotted"; color="red"; %s }\n\n' % (c,'\n'.join(text))
            c += 1 
            
        for index, name in queries :
            s += 'q_%s [ label="%s", shape="plaintext" ];\n'   % (index, name)
            s += 'q_%s -> %s [style="dotted"];\n'  % (index, index)

        return s + '}'

class _UnknownClause(Exception) : pass

class UnknownClause(Exception) : pass

class UserAbort(Exception) : pass

class UserFail(Exception) : pass

class NonGroundQuery(Exception) : pass
        
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
