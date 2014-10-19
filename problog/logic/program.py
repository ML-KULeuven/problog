from __future__ import print_function

from .basic import Term, Var

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

class Clause(object) :
    
    def __init__(self, head, body) :
        self.head = head
        self.body = body
        
    def _compile(self, db) :
        var = _AutoDict()        
        new_head = self.head.apply(var)
        body_node = self.body._compile(db, var)[0]
        res = db._addClauseNode(new_head, body_node, len(var))
        return res

    def __repr__(self) :
        return "%s :- %s" % (self.head, self.body)
        
class Lit(Term) :
    """Body literal. This corresponds to a ``call``.
    
    This is a subclass of :class:`problog.logic.basic.Term`.
    
    :param functor: functor of the call
    :type functor: :class:`str`
    :param args: arguments of the call
    :type args: :class:`.Term`
    
    """
    
    def __init__(self, functor, *args) :
        Term.__init__(self, functor, *args)
        
    def _compile(self, db, variables) :
        """Compile the literal into a ``call`` and add it to the database.
        
        :param db: clause database 
        :type db: ClauseDB
        :param variables: variable translation dictionary 
        :type variables: AutoDict
        """
        return db._addCall( self.apply(variables) ), variables.usedVars()
        
    @classmethod
    def create(cls, func) :
        """Create a factory function for a Lit with a given functor.
        
        :param func: functor 
        :type func: :class:`str`
        :returns: callable -- function for creating Lits
        
        Example::
        
            p = Lit.create('p')
            t1 = p(X,Y)
            t2 = p(p(X,X))
        
        """
        
        return lambda *args : Lit(func,*args)
        
    def __lshift__(self, body) :
        return Clause(self, body)
        
    def __and__(self, rhs) :
        return And(self, rhs)
        
    def __or__(self, rhs) :
        return Or(self, rhs)
                
    def __invert__(self) :
        return Not(self)
        
    def withArgs(self,*args) :
        """Creates a new Lit with the same functor and the given arguments.
        
        :param args: new arguments for the term
        :type args: any
        :returns: a new term with the given arguments
        :rtype: :class:`Lit`
        
        """
        return Lit(self.functor, *args)

        
class Or(object) :
    """Or"""
    
    def __init__(self, op1, op2) :
        self.op1 = op1
        self.op2 = op2
        
    def _compile(self, db, variables) :
        op1, op1Vars = self.op1._compile(db, variables)
        op2, op2Vars = self.op2._compile(db, variables)
        opVars = (op1Vars | op2Vars)
        return db._addOr( op1, op2, usedVars = opVars), opVars
        
    def __or__(self, rhs) :
        self.op2 = self.op2 | rhs
        return self
        
    def __and__(self, rhs) :
        return And(self, rhs)
            
    def __repr__(self) :
        lhs = str(self.op1)
        rhs = str(self.op2)        
        return "%s; %s" % (lhs, rhs)
        
    
class And(object) :
    """And"""
    
    def __init__(self, op1, op2) :
        self.op1 = op1
        self.op2 = op2
    
    def _compile(self, db, variables) :
        op1, op1Vars = self.op1._compile(db, variables)
        op2, op2Vars = self.op2._compile(db, variables)
        opVars = (op1Vars | op2Vars)
        return db._addAnd( op1, op2, usedVars = opVars), opVars
        
    def __and__(self, rhs) :
        self.op2 = self.op2 & rhs
        return self
        
    def __or__(self, rhs) :
        return Or(self, rhs)
    
    def __repr__(self) :
        lhs = str(self.op1)
        rhs = str(self.op2)
        if isinstance(self.op2, Or) :
            rhs = '(%s)' % rhs
        if isinstance(self.op1, Or) :
            lhs = '(%s)' % lhs
        
        return "%s, %s" % (lhs, rhs)
        
class Not(object) :
    """Not"""
    
    def __init__(self, child) :
        self.child = child
        
    def _compile(self, db, variables) :
        op, opVars = self.child._compile(db, variables)
        return db._addNot( op, usedVars = opVars), opVars
        
    def __repr__(self) :
        c = str(self.child)
        if isinstance(self.child, And) :
            c = '(%s)' % c
        return '\+(%s)' % c

class LogicProgram(object) :
    
    def __init__(self) :
        pass
        
    def __iter__(self) :
        """Iterator for the clauses in the program."""
        raise NotImplementedError("LogicProgram.__iter__ is an abstract method." )
        
    def addClause(self, clause) :
        raise NotImplementedError("LogicProgram.addClause is an abstract method." )
        
    def addFact(self, fact) :
        raise NotImplementedError("LogicProgram.addFact is an abstract method." )
        
    def __iadd__(self, clausefact) :
        if isinstance(clausefact, Clause) :
            self.addClause(clausefact)
        else :
            self.addFact(clausefact)
        return self


class ClauseDB(LogicProgram) :
    """Compiled logic program.
    
    A logic program is compiled into a table of instructions.
    
    The following instruction types are available:
    
    * and ( child1, child2 )
    * or  ( children )
    * not ( child )
    * call ( arguments )
    * clause ( definitions )
    * def ( head arguments, body node, variables in body )
    * fact ( argument )
    * *empty* ( undefined (e.g. builtin) )
    
    .. todo:: 
        
        annotated disjunction are not supported yet
    
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
                node = self._appendNode( 'clause', [], head.functor )
            else :
                node = self._appendNode()
            self._setHead( head, node )
        return node
    
    def _addClauseNode( self, head, body_node, body_vars ) :
        """Add a clause node."""
        subnode = self._addDef( head, body_node, body_vars )        
        return self._addClauseBody( head, subnode )
        
    def addFact( self, term) :
        """Add a fact to the database.
        
        :param term: term to add
        :type term: :class:`.basic.Term`
        :returns: location of the fact in the database
        :rtype: :class:`int`
        """
        subnode = self._appendNode('fact', term.args)
        return self._addClauseBody( term, subnode )
                
    def _addClauseBody( self, head, subnode ) :
        index = self._addHead( head )
        def_node = self.getNode(index)
        if not def_node :
            clauses = []
            self._setNode( index, 'clause', clauses, head.functor )
        else :
            clauses = def_node[1]
        clauses.append( subnode )
        return index
            
    def _addDef( self, term, subnode, body_vars ) :
        """Add a *definition* node."""
        return self._appendNode('def', subnode, term.args, body_vars)
                
    def _addCall( self, term ) :
        """Add a *call* node."""
        node = self._addHead(term, create=False)
        #print (type(term), type(term.functor), term)
        return self._appendNode( 'call', node, term.args, term.functor )
        
    def _addAnd( self, op1, op2, usedVars=[] ) :
        """Add an *and* node."""
        return self._appendNode( 'and', (op1,op2) )
        
    def _addNot( self, op1, usedVars=[] ) :
        """Add a *not* node."""
        return self._appendNode( 'not', op1 )        
        
    def _addOr( self, op1, op2, usedVars=[] ) :
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
        
    def addClause(self, clause) :
        """Add a clause to the database.
        
        :param clause: Clause to add
        :type clause: :class:`.Clause`
        :returns: location of the definition node in the database
        :rtype: :class:`int`
        """
        return clause._compile(self)
    
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
            return Lit(func, *node[1])
        elif node[0] == 'def' :
            head = self._create_vars( Term(func,*node[2]) )
            return Clause( head, self._extract(node[1]))
        elif node[0] == 'call' :
            func = node[3]
            args = node[2]
            return self._create_vars( Lit(func, *args) )
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
        
        
            