from __future__ import print_function

from .basic import Term, Var, And, Or, Not, Clause, LogicProgram, Constant, AnnotatedDisjunction

from .basic import LogicProgram

from ..parser import PrologParser, Factory


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
        # Term
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
            heads = self._uncurry( operand1 )
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
    
    The following instruction types are available:
    
    * ('clause', [ <definitions> ], <head.functor>, <head.arity> )    => uniquely identified by functor
    * ('call', <clause / builtin>, <term.arguments>, <term.functor> )
    * ('def', < body node >, < head.arguments >, < #vars in head+body >, <head.functor> )
    * ('fact', < arguments >, <probability> )
    * ('ad', [ <adc_nodes> ])
    * ('adc', <body node>, [<arguments>], <#vars in head+body>, <head.functor>, <parent ad node>  )
    
    * ('and', [child1,child2] )
    * ('or', [child1,child2] )
    * ('not', child)
        
    .. todo:: 
        
        * add annotated disjunctions (*ad*)
        * add probability field
        * remove empty nodes -> replace by None pointer in call => requires prior knowledge of builtins
    
    """
    
    def __init__(self) :
        LogicProgram.__init__(self)
        self.__nodes = []   # list of nodes
        self.__heads = {}   # head.sig => node index
    
    def getNode(self, index) :
        """Get the instruction node at the given index.
        
        :param index: index of the node to retrieve
        :type index: :class:`int`
        :returns: requested node
        :rtype: :class:`tuple`
        :raises IndexError: the given index does not point to a node
        
        """
        return self.__nodes[index]
        
    def _setNode(self, index, node_type, node_content, *node_extra) :
        self.__nodes[index] = (node_type, node_content) + node_extra
        
    def _appendNode(self, *node_extra) :
        index = len(self.__nodes)
        self.__nodes.append( node_extra )
        return index
    
    def _getHead(self, head) :
        return self.__heads.get( head.signature )
        
    def _setHead(self, head, index) :
        self.__heads[ head.signature ] = index
    
    def _addHead( self, head, create=True ) :
        node = self._getHead( head )
        if node == None :
            if create :
                node = self._appendNode( 'clause', set([]), head.functor, head.arity )
            else :
                node = self._appendNode()
            self._setHead( head, node )
        return node

    def _addAD( self, heads, body_node, body_vars ) :
        ad_node = self._appendNode(None)
        adc_nodes = [ self._addADChoice( head, body_node, body_vars, ad_node ) for head in heads ]
        self._setNode( ad_node, 'ad', adc_nodes )

        for head, adc_node in zip(heads, adc_nodes) :
            self._addClauseBody(head, adc_node )
        return ad_node
        
        # 'ad', body node, head terms
        subnode = self._addDef( Term('', *range(0,body_vars)), body_node, body_vars )
        node = self._appendNode('ad', subnode, heads)
        for head in heads :
            self._addClauseBody( head, node )
        return None

    def _addADChoice( self, term, subnode, body_vars, parent ) :
        """Add a *definition* node."""
        return self._appendNode('adc', subnode, term.args, body_vars, term.functor, parent)
    
    def _addClauseNode( self, head, body_node, body_vars ) :
        """Add a clause node."""
        subnode = self._addDef( head, body_node, body_vars )        
        return self._addClauseBody( head, subnode )
        
    def _addFact( self, term) :
        """Add a fact to the database.
        
        :param term: term to add
        :type term: :class:`.basic.Term`
        :returns: location of the fact in the database
        :rtype: :class:`int`
        """
        subnode = self._appendNode('fact', term.args, term.probability, term.functor )
        return self._addClauseBody( term, subnode )
                
    def _addClauseBody( self, head, subnode ) :
        index = self._addHead( head )
        def_node = self.getNode(index)
        if not def_node :
            clauses = set([])
            self._setNode( index, 'clause', clauses, head.functor, head.arity )
        else :
            clauses = def_node[1]
        clauses.add( subnode )
        return index
            
    def _addDef( self, term, subnode, body_vars ) :
        """Add a *definition* node."""
        return self._appendNode('def', subnode, term.args, body_vars, term.functor)
                
    def _addCall( self, term ) :
        """Add a *call* node."""
        node = self._addHead(term, create=False)
        #print (type(term), type(term.functor), term)
        return self._appendNode( 'call', node, term.args, term.functor )
        
    def _addAnd( self, op1, op2 ) :
        """Add an *and* node."""
        return self._appendNode( 'and', (op1,op2) )
        
    def _addNot( self, op1 ) :
        """Add a *not* node."""
        return self._appendNode( 'not', op1 )        
        
    def _addOr( self, op1, op2 ) :
        """Add an *or* node."""
        return self._appendNode( 'or', (op1,op2) )
    
    def find(self, head ) :
        """Find the clause node corresponding to the given head.
        
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
            return self._addAnd( op1, op2)
        elif isinstance(struct, Or) :
            op1 = self._compile(struct.op1, variables)
            op2 = self._compile(struct.op2, variables)
            return self._addOr( op1, op2)
        elif isinstance(struct, Not) :
            child = self._compile(struct.child, variables)
            return self._addNot( child)
        elif isinstance(struct, AnnotatedDisjunction) :
            new_heads = [ head.apply(variables) for head in struct.heads ]
            body_node = self._compile(struct.body, variables)
            return self._addAD( new_heads, body_node, len(variables) )
        elif isinstance(struct, Clause) :
            new_head = struct.head.apply(variables)
            body_node = self._compile(struct.body, variables)
            return self._addClauseNode(new_head, body_node, len(variables))
        elif isinstance(struct, Term) :
            return self._addCall( struct.apply(variables) )
        else :
            raise ValueError("Unknown structure type: '%s'" % struct )
    
    def _create_vars(self, term) :
        if type(term) == int :
            return Var('V_' + str(term))
        else :
            args = [ self._create_vars(arg) for arg in term.args ]
            return term.withArgs(*args)
        
    def _extract(self, node_id, func=None) :
        node = self.getNode(node_id)
        if not node :
            raise ValueError("Unexpected empty node.")    
        if node[0] == 'fact' :
            return Term(func, *node[1])
        elif node[0] == 'def' :
            head = self._create_vars( Term(func,*node[2]) )
            return Clause( head, self._extract(node[1]))
        elif node[0] == 'call' :
            func = node[3]
            args = node[2]
            return self._create_vars( Term(func, *args) )
        elif node[0] == 'and' :
            a,b = node[1]
            return And( self._extract(a), self._extract(b) )
        elif node[0] == 'or' :
            a,b = node[1]
            return Or( self._extract(a), self._extract(b) )
        else :
            raise ValueError("Unknown node type: '%s'" % node[0])    
        
        
    def __iter__(self) :
        for node in self.__nodes :
            if node and node[0] == 'clause' :
                for defnode in node[1] :
                    yield self._extract( defnode, node[2] )
        
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
            