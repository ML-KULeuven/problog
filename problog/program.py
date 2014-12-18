from __future__ import print_function

from .logic import *

from .parser import DefaultPrologParser, Factory

from collections import namedtuple, defaultdict

class SimpleProgram(LogicProgram) :
    """LogicProgram implementation as a list of clauses."""
    
    def __init__(self) :
        self.__clauses = []
        
    def _addAnnotatedDisjunction(self, clause) :
        self.__clauses.append( clause )
        
    def _addClause(self, clause) :
        self.__clauses.append( clause )
        
    def _addFact(self, fact) :
        self.__clauses.append( fact )
    
    def __iter__(self) :
        return iter(self.__clauses)

class PrologString(LogicProgram) :
    
    def __init__(self, string, parser=None) :
        self.__string = string
        if parser == None :
            self.parser = DefaultPrologParser(PrologFactory())
        else :
            self.parser = parser
        
    def __iter__(self) :
        """Iterator for the clauses in the program."""
        program = self.parser.parseString(self.__string)
        return iter(program)


class PrologFile(LogicProgram) :
    """LogicProgram implementation as a pointer to a Prolog file.
    
    :param filename: filename of the Prolog file (optional)
    :type filename: string
    :param allow_update: allow modifications to given file, otherwise forces copy
    :type allow_update: bool
    """
    
    def __init__(self, filename=None, allow_update=False, parser=None) :
        if parser == None :
            self.parser = DefaultPrologParser(PrologFactory())
        else :
            self.parser = parser
        
        
        if filename == None :
            filename = self._new_filename()
            allow_update = True
        self.__filename = filename
        self.__allow_update = allow_update
        
        self.__buffer = []  # for new clauses
        
    def _new_filename(self) :
        import tempfile
        (handle, filename) = tempfile.mkstemp(suffix='.pl')
        return filename
        
    def _write_buffer(self) :
        if self.__buffer :
            if not self.__allow_update :
                # Copy file
                new_filename = self._new_filename()
                shutil.copyfile(self.__filename, new_filename)
                self.__filename = new_filename
        
            filename = self.__filename
            with open(filename, 'a') as f :
                for line in self.__buffer :
                    f.write(line)
                self.__buffer = []
        
    def _get_filename(self) :
        self._write_buffer()
        return self.__filename
    filename = property( _get_filename )
        
    def __iter__(self) :
        """Iterator for the clauses in the program."""
        program = self.parser.parseFile(self.filename)
        return iter(program)
        
    def _addAnnotatedDisjunction(self, clause) :
        self.__buffer.append( clause )
        
    def _addClause(self, clause) :
        self.__buffer.append( clause )
        
    def _addFact(self, fact) :
        self.__buffer.append( fact )
        
        
class PrologFactory(Factory) :
    """Factory object for creating suitable objects from the parse tree."""
        
    def build_program(self, clauses) :
        # LogicProgram
        result = SimpleProgram()
        for clause in clauses :
            result += clause
        return result
    
    def build_function(self, functor, arguments) :
        return Term( functor, *arguments )
        
    def build_variable(self, name) :
        return Var(name)
        
    def build_constant(self, value) :
        return Constant(value)
        
    def build_binop(self, functor, operand1, operand2, function=None, **extra) :
        return self.build_function("'" + functor + "'", (operand1, operand2))
        
    def build_unop(self, functor, operand, **extra) :
        return self.build_function("'" + functor + "'", (operand,) )
        
    def build_list(self, values, tail=None, **extra) :
        if tail == None :
            current = Term('[]')
        else :
            current = tail
        for value in reversed(values) :
            current = self.build_function('.', (value, current) )
        return current
        
    def build_string(self, value) :
        return self.build_constant('"' + value + '"');
    
    def build_cut(self) :
        raise NotImplementedError('Not supported!')
        
    def build_clause(self, functor, operand1, operand2, **extra) :
        heads = operand1
        #heads = self._uncurry( operand1, ';' )
        if len(heads) > 1 :
            return AnnotatedDisjunction(heads, operand2)
        else :
            return Clause(operand1[0], operand2)
        
    def build_disjunction(self, functor, operand1, operand2, **extra) :
        return Or(operand1, operand2)
    
    def build_conjunction(self, functor, operand1, operand2, **extra) :
        return And(operand1, operand2)
    
    def build_not(self, functor, operand, **extra) :
        return Not(operand)
        
    def build_probabilistic(self, operand1, operand2, **extra) :
        operand2.probability = operand1
        return operand2
        
    def _uncurry(self, term, func=None) :
        if func == None : func = term.functor
        
        body = []
        current = term
        while isinstance(current, Term) and current.functor == func :
            body.append(current.args[0])
            current = current.args[1]
        body.append(current)
        return body    

class ClauseIndex(list) :
    
    def __init__(self, parent, arity) :
        self.__parent = parent
        self.__index = [ defaultdict(set) for i in range(0,arity) ]
        
    def find(self, arguments) :
        results = set(self)
        for i, arg in enumerate(arguments) :
            if not isinstance(arg,Term) or not arg.isGround() :    # No restrict
                pass
            else :
                curr = self.__index[i][None] | self.__index[i][arg]
                results &= curr
        return results
        
    def _add(self, key, item) :
        for i, k in enumerate(key) :
            self.__index[i][k].add(item)
        
    def append(self, item) :
        list.append(self, item)
        key = []
        args = self.__parent.getNode(item).args
        for arg in args :
            if isinstance(arg,Term) and arg.isGround() :
                key.append(arg)
            else :
                key.append(None)
        self._add(key, item)

class ClauseDB(LogicProgram) :
    """Compiled logic program.
    
    A logic program is compiled into a table of instructions.
    The types of instructions are:
    
    define( functor, arity, defs )
        Pointer to all definitions of functor/arity.
        Definitions can be: ``fact``, ``clause`` or ``adc``.
    
    clause( functor, arguments, bodynode, varcount )
        Single clause. Functor is the head functor, Arguments are the head arguments. Body node is a pointer to the node representing the body. Var count is the number of variables in head and body.
        
    fact( functor, arguments, probability )
        Single fact. 
        
    adc( functor, arguments, bodynode, varcount, parent )
        Single annotated disjunction choice. Fields have same meaning as with ``clause``, parent_node points to the parent ``ad`` node.
        
    ad( childnodes )
        Annotated disjunction group. Child nodes point to the ``adc`` nodes of the clause.

    call( functor, arguments, defnode )
        Body literal with call to clause or builtin. Arguments contains the call arguments, definition node is the pointer to the definition node of the given functor/arity.
    
    conj( childnodes )
        Logical and. Currently, only 2 children are supported.
    
    disj( childnodes )
        Logical or. Currently, only 2 children are supported.
    
    neg( childnode )
        Logical not.
                
    .. todo:: 
        
        * add annotated disjunctions (*ad*)
        * add probability field
        * remove empty nodes -> replace by None pointer in call => requires prior knowledge of builtins
    
    """
    
    _define = namedtuple('define', ('functor', 'arity', 'children') )
    _clause = namedtuple('clause', ('functor', 'args', 'probability', 'child', 'varcount', 'group') )
    _fact   = namedtuple('fact'  , ('functor', 'args', 'probability') )
    _call   = namedtuple('call'  , ('functor', 'args', 'defnode' )) 
    _disj   = namedtuple('disj'  , ('children' ) )
    _conj   = namedtuple('conj'  , ('children' ) )
    _neg    = namedtuple('neg'   , ('child' ) )
    _choice = namedtuple('choice', ('functor', 'args', 'probability', 'group', 'choice') )
    
    def __init__(self, builtins=None) :
        LogicProgram.__init__(self)
        self.__nodes = []   # list of nodes
        self.__heads = {}   # head.sig => node index
        
        self.__builtins = builtins
    
    def __len__(self) :
        return len(self.__nodes)
        
    def _getBuiltIn(self, signature) :
        if self.__builtins == None :
            return None
        else :
            return self.__builtins.get(signature)
    
    def _create_index(self, arity) :
        # return []
        return ClauseIndex(self, arity)
            
    def _addAndNode( self, op1, op2 ) :
        """Add an *and* node."""
        return self._appendNode( self._conj((op1,op2)))
        
    def _addNotNode( self, op1 ) :
        """Add a *not* node."""
        return self._appendNode( self._neg(op1) )
        
    def _addOrNode( self, op1, op2 ) :
        """Add an *or* node."""
        return self._appendNode( self._disj((op1,op2)))
    
    def _addDefineNode( self, head, childnode ) :
        define_index = self._addHead( head )
        define_node = self.getNode(define_index)
        if not define_node :
            clauses = self._create_index(head.arity)
            self._setNode( define_index, self._define( head.functor, head.arity, clauses ) )
        else :
            clauses = define_node.children
        clauses.append( childnode )
        return childnode
    
    def _addChoiceNode(self, choice, args, probability, group) :
        functor = 'ad_%s_%s' % (group, choice)
        choice_node = self._appendNode( self._choice(functor, args, probability, group, choice) )
        return choice_node
        
    def _addClauseNode( self, head, body, varcount, group=None ) :
        clause_node = self._appendNode( self._clause( head.functor, head.args, head.probability, body, varcount, group ) )
        return self._addDefineNode( head, clause_node )

        
    def _addCallNode( self, term ) :
        """Add a *call* node."""
        defnode = self._addHead(term, create=False)
        return self._appendNode( self._call( term.functor, term.args, defnode ) )
    
    def getNode(self, index) :
        """Get the instruction node at the given index.
        
        :param index: index of the node to retrieve
        :type index: :class:`int`
        :returns: requested node
        :rtype: :class:`tuple`
        :raises IndexError: the given index does not point to a node
        
        """
        return self.__nodes[index]
        
    def _setNode(self, index, node) :
        self.__nodes[index] = node
        
    def _appendNode(self, node=()) :
        index = len(self.__nodes)
        self.__nodes.append( node )
        return index
    
    def _getHead(self, head) :
        return self.__heads.get( head.signature )
        
    def _setHead(self, head, index) :
        self.__heads[ head.signature ] = index
    
    def _addHead( self, head, create=True ) :
        node = self._getBuiltIn( head.signature )
        if node != None :
            if create :
                raise AccessError("Can not overwrite built-in '%s'." % head.signature )
            else :
                return node
        
        node = self._getHead( head )
        if node == None :
            if create :
                node = self._appendNode( self._define( head.functor, head.arity, self._create_index(head.arity)) )
            else :
                node = self._appendNode()
            self._setHead( head, node )
        return node

    def find(self, head ) :
        """Find the ``define`` node corresponding to the given head.
        
        :param head: clause head to match
        :type head: :class:`.basic.Term`
        :returns: location of the clause node in the database, returns ``None`` if no such node exists
        :rtype: :class:`int` or ``None``
        """
        return self._getHead( head )
       
    def __repr__(self) :
        s = ''
        for i,n in enumerate(self.__nodes) :
            s += '%s: %s\n' % (i,n)
        s += str(self.__heads)
        return s
        
    def _addClause(self, clause) :
        """Add a clause to the database.
        
        :param clause: Clause to add
        :type clause: :class:`.Clause`
        :returns: location of the definition node in the database
        :rtype: :class:`int`
        """
        return self._compile( clause )
    
    def _addAnnotatedDisjunction(self, clause) :
        return self._compile( clause )
    
    def _addFact( self, term) :
        variables = _AutoDict()
        new_head = term.apply(variables)
        if len(variables) == 0 :
            fact_node = self._appendNode( self._fact(term.functor, term.args, term.probability))
            return self._addDefineNode( term, fact_node )
        else :
            return self._addClause( Clause(term, Term('true')) )
    
    def _compile(self, struct, variables=None) :
        if variables == None : variables = _AutoDict()
        
        if isinstance(struct, And) :
            op1 = self._compile(struct.op1, variables)
            op2 = self._compile(struct.op2, variables)
            return self._addAndNode( op1, op2)
        elif isinstance(struct, Or) :
            op1 = self._compile(struct.op1, variables)
            op2 = self._compile(struct.op2, variables)
            return self._addOrNode( op1, op2)
        elif isinstance(struct, Not) :
            child = self._compile(struct.child, variables)
            return self._addNotNode( child)
        elif isinstance(struct, AnnotatedDisjunction) :
            # Determine number of variables in the head
            new_heads = [ head.apply(variables) for head in struct.heads ]            
            head_count = len(variables)
            
            # Body arguments
            body_args = tuple(range(0,head_count))
            
            # Group id
            group = len(self.__nodes)
            
            # Create the body clause
            body_head = Term('ad_%s_body' % group, *body_args)
            body_node = self._compile(struct.body, variables)
            clause_body = self._addClauseNode( body_head, body_node, len(variables) )
            #clause_body = self._appendNode( self._clause( body_head.functor, body_head.args, None, body_node, len(variables), group=None ) )
            clause_body = self._addHead( body_head )
            
            for choice, head in enumerate(new_heads) :
                # For each head: add choice node
                choice_node = self._addChoiceNode(choice, body_args, head.probability, group )
                choice_call = self._appendNode( self._call( 'ad_%s_%s' % (group, choice), body_args, choice_node ) )
                body_call = self._appendNode( self._call( 'ad_%s_body' % group, body_args , clause_body ) )
                choice_body = self._addAndNode( body_call, choice_call )
                head_clause = self._addClauseNode( head, choice_body, head_count, group=group )
            return None
        elif isinstance(struct, Clause) :
            if struct.head.probability != None :
                return self._compile( AnnotatedDisjunction( [struct.head], struct.body))
            else :
                new_head = struct.head.apply(variables)
                head_count = len(variables)
                body_node = self._compile(struct.body, variables)
                return self._addClauseNode(new_head, body_node, len(variables))
        elif isinstance(struct, Term) :
            return self._addCallNode( struct.apply(variables) )
        else :
            raise ValueError("Unknown structure type: '%s'" % struct )
    
    def _create_vars(self, term) :
        if type(term) == int :
            return Var('V_' + str(term))
        else :
            args = [ self._create_vars(arg) for arg in term.args ]
            return term.withArgs(*args)
        
    def _extract(self, node_id) :
        node = self.getNode(node_id)
        if not node :
            raise ValueError("Unexpected empty node.")    
        
        groups = defaultdict(list)
        nodetype = type(node).__name__
        if nodetype == 'fact' :
            return Term(node.functor, *node.args, p=node.probability)
        elif nodetype == 'clause' :
            if clause.group != None :   # part of annotated disjunction
                groups
            else :
                head = self._create_vars( Term(node.functor,*node.args, p=node.probability) )
                return Clause( head, self._extract(node.child))
        elif nodetype == 'call' :
            func = node.functor
            args = node.args
            return self._create_vars( Term(func, *args) )
        elif nodetype == 'conj' :
            a,b = node.children
            return And( self._extract(a), self._extract(b) )
        elif nodetype == 'disj' :
            a,b = node.children
            return Or( self._extract(a), self._extract(b) )
        elif nodetype == 'neg' :
            return Not( self._extract(node.child))
            
        else :
            raise ValueError("Unknown node type: '%s'" % nodetype)    
        
    def __iter__(self) :
        clause_groups = defaultdict(list)
        for index, node in enumerate(self.__nodes) :
            if not node : continue
            nodetype = type(node).__name__
            if nodetype == 'fact' :
                yield Term(node.functor, *node.args, p=node.probability)
            elif nodetype == 'clause' :
                if node.group == None :
                    head = self._create_vars( Term(node.functor,*node.args, p=node.probability) )
                    yield Clause( head, self._extract(node.child))
                else :
                    clause_groups[node.group].append(index)
            
            
            #if node and type(node).__name__ in ('fact', 'clause') :
               # yield self._extract( index )
        for group in clause_groups.values() :
            heads = []
            body = None
            for index in group :
                node = self.getNode(index)
                heads.append( self._create_vars( Term( node.functor, *node.args, p=node.probability)))
                if body == None :
                    body_node = self.getNode(node.child)
                    body_node = self.getNode(body_node.children[0])
                    body = self._create_vars( Term(body_node.functor, *body_node.args) )
            yield AnnotatedDisjunction(heads, body)

class AccessError(Exception) : pass            
        
class _AutoDict(dict) :
    
    def __init__(self) :
        dict.__init__(self)
        self.__record = set()
        self.__anon = 0
    
    def __getitem__(self, key) :
        if key == '_' :
            value = len(self)
            self.__anon += 1
            return value
        else :        
            value = self.get(key)
            if value == None :
                value = len(self)
                self[key] = value
            self.__record.add(value)
            return value
            
    def __len__(self) :
        return dict.__len__(self) + self.__anon
        
    def usedVars(self) :
        result = set(self.__record)
        self.__record.clear()
        return result
        
    def define(self, key) :
        if not key in self :
            value = len(self)
            self[key] = value
            
