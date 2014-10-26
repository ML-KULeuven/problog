from __future__ import print_function

from .basic import Term, Var, And, Or, Not, Clause, LogicProgram, Constant, AnnotatedDisjunction

from .basic import LogicProgram

from ..parser import PrologParser, Factory

from collections import namedtuple

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


class PrologFile(LogicProgram) :
    """LogicProgram implementation as a pointer to a Prolog file.
    
    :param filename: filename of the Prolog file (optional)
    :type filename: string
    :param allow_update: allow modifications to given file, otherwise forces copy
    :type allow_update: bool
    """
    
    def __init__(self, filename=None, allow_update=False) :
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
        parser = PrologParser(PrologFactory())
        program = parser.parseFile(self.filename)
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
            current = '[]'
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
        if functor == '<-' :
            heads = self._uncurry( operand1, ';' )
            return AnnotatedDisjunction(heads, operand2)
        else :
            return Clause(operand1, operand2)
        
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
    _clause = namedtuple('clause', ('functor', 'args', 'probability', 'child', 'varcount') )
    _fact   = namedtuple('fact'  , ('functor', 'args', 'probability') )
    _adc    = namedtuple('adc'   , ('functor', 'args', 'probability', 'ad' ) )
    _ad     = namedtuple('ad'    , ('functor', 'args', 'child', 'varcount', 'choices') )
    _call   = namedtuple('call'  , ('functor', 'args', 'defnode' )) 
    _disj   = namedtuple('disj'  , ('children' ) )
    _conj   = namedtuple('conj'  , ('children' ) )
    _neg    = namedtuple('neg'   , ('child' ) )
        
    def __init__(self) :
        LogicProgram.__init__(self)
        self.__nodes = []   # list of nodes
        self.__heads = {}   # head.sig => node index
    
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
            clauses = []
            self._setNode( define_index, self._define( head.functor, head.arity, clauses ) )
        else :
            clauses = define_node.children
        clauses.append( childnode )
        return childnode
    
    def _addFact( self, term) :
        fact_node = self._appendNode( self._fact(term.functor, term.args, term.probability))
        return self._addDefineNode( term, fact_node )
        
    def _addClauseNode( self, head, body, varcount ) :
        clause_node = self._appendNode( self._clause( head.functor, head.args, head.probability, body, varcount ) )
        return self._addDefineNode( head, clause_node )

    def _addADChoiceNode( self, head, ad_node ) :
        return self._appendNode( self._adc( head.functor, head.args, head.probability, ad_node ) )
        
    def _addCallNode( self, term ) :
        """Add a *call* node."""
        defnode = self._addHead(term, create=False)
        return self._appendNode( self._call( term.functor, term.args, defnode ) )
    
    def _addADNode( self, heads, head_count, body_node, body_vars ) :
        """Add an annotated disjunction.

        :param heads: list of heads
        :type heads: seq of Term
        :param head_count: number of variables in head
        
        """
        ad_index = self._appendNode(None)
        head_functor = '#ad_%s' % ad_index
        head_args = tuple(range(0,head_count))
        
        adc_nodes = [ self._addADChoiceNode( head, ad_index ) for head in heads ]
        ad_node = self._ad( head_functor, head_args, body_node, body_vars, adc_nodes )
        self._setNode( ad_index, ad_node )
        for head, adc_node in zip(heads, adc_nodes) :
            self._addDefineNode(head, adc_node )
        return ad_index
        
    
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
        node = self._getHead( head )
        if node == None :
            if create :
                node = self._appendNode( self._define( head.functor, head.arity, []) )
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
            # variables is empty
            new_heads = [ head.apply(variables) for head in struct.heads ]
            head_count = len(variables)
            body_node = self._compile(struct.body, variables)
            return self._addADNode( new_heads, head_count, body_node, len(variables) )
        elif isinstance(struct, Clause) :
            new_head = struct.head.apply(variables)
            head_count = len(variables)
            body_node = self._compile(struct.body, variables)
            if new_head.probability != None :
                return self._addADNode( [new_head], head_count, body_node, len(variables) )
            else :
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
            
        nodetype = type(node).__name__
        if nodetype == 'fact' :
            return Term(node.functor, *node.args, p=node.probability)
        elif nodetype == 'clause' :
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
        elif nodetype == 'ad' :
            heads = [ self._extract( c ) for c in node.choices ]
            return AnnotatedDisjunction( heads, self._extract( node.child ) )
        elif nodetype == 'adc' :
            return self._create_vars( Term(node.functor, *node.args, p=node.probability) )
            
        else :
            raise ValueError("Unknown node type: '%s'" % nodetype)    
        
    def __iter__(self) :
        for index, node in enumerate(self.__nodes) :
            if node and type(node).__name__ in ('fact', 'clause', 'ad') :
                yield self._extract( index )
        
class _AutoDict(dict) :
    
    def __init__(self) :
        dict.__init__(self)
        self.__record = set()
    
    def __getitem__(self, key) :
        value = self.get(key)
        if value == None :
            value = len(self)
            self[key] = value
        self.__record.add(value)
        return value
        
    def usedVars(self) :
        result = set(self.__record)
        self.__record.clear()
        return result
        
    def define(self, key) :
        if not key in self :
            value = len(self)
            self[key] = value
            