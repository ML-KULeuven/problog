"""
This module contains basic logic constructs.
    
    A Term can be:
        * a function (see :class:`Term`)
        * a variable (see :class:`Var`)
        * a constant (see :class:`Constant`)
        
    Four functions are handled separately:
        * conjunction (see :class:`And`)
        * disjunction (see :class:`Or`)
        * negation (see :class:`Not`)
        * clause (see :class:`Clause`)
    
    **Syntactic sugar**
    
    Clauses can be constructed by virtue of overloading of Python operators:
    
      =========== =========== ============
       Prolog      Python      English
      =========== =========== ============
       ``:-``          ``<<``      clause
       ``,``           ``&``       and
       ``;``           ``|``       or
       ``\+``          ``~``       not
      =========== =========== ============
      
    .. warning::
    
        Due to Python's operator priorities, the body of the clause has to be between parentheses.
    
    
    **Example**::
    
        from problog.logic import Var, Term
        
        # Functors (arguments will be added later)
        ancestor = Term('anc')
        parent = Term('par')
        
        # Literals
        leo3 = Term('leo3')
        al2 = Term('al2')
        phil = Term('phil')  
        
        # Variables
        X = Var('X')
        Y = Var('Y')
        Z = Var('Z')
        
        # Clauses
        c1 = ( ancestor(X,Y) << parent(X,Y) )
        c2 = ( ancestor(X,Y) << ( parent(X,Z) & ancestor(Z,Y) ) )
        c3 = ( parent( leo3, al2 ) )
        c4 = ( parent( al2, phil ) )
        
        
.. moduleauthor:: Anton Dries <anton.dries@cs.kuleuven.be>

"""

from __future__ import print_function

class Term(object) :
    """Represent a first-order Term."""
    
    def __init__(self, functor, *args) :
        self.__functor = functor
        self.__args = args
    
    def _get_functor(self) : return self.__functor
    functor = property( _get_functor, doc="Term functor" )
        
    def _get_args(self) : return self.__args
    args = property( _get_args, doc="Term arguments" )
    
    def _get_arity(self) : return len(self.__args)
    arity = property( _get_arity, doc="Number of arguments.")
        
    def _get_signature(self) : return '%s/%s' % (self.functor, self.arity)
    signature = property( _get_signature, doc="Term's signature ``functor/arity``" )
        
    def apply(self, subst) :
        """Apply the given substitution to the variables in the term.
        
        :param subst: A mapping from variable names to something else
        :type subst: an object with a __getitem__ method
        :raises: whatever subst.__getitem__ raises
        :returns: a new Term with all variables replaced by their values from the given substitution
        :rtype: :class:`Term`
        
        """
        return Term( self.functor, *[ arg.apply(subst) for arg in self.args ])
            
    def __repr__(self) :
        if self.args :
            return '%s(%s)' % (self.functor, ','.join(map(str,self.args)))
        else :
            return '%s' % (self.functor,)
        
    def __call__(self, *args) :
        return self.withArgs(*args)
        
    def withArgs(self,*args) :
        """Creates a new Term with the same functor and the given arguments.
        
        :param args: new arguments for the term
        :type args: any
        :returns: a new term with the given arguments
        :rtype: :class:`Term`
        
        """
        return self.__class__(self.functor, *args)
        
    def isVar(self) :
        """Checks whether this Term represents a variable.
        
            :returns: ``False``
        """
        return False
        
    def isConstant(self) :
        """Checks whether this Term represents a constant.
        
            :returns: ``False``
        """
        return False
        
    def __eq__(self, other) :
        # TODO: this can be very slow?
        if other == None :
            return None
        else :
            return (self.functor, self.args) == (other.functor, other.args)
        
    def __hash__(self) :
        return hash((self.functor, self.args))
        
    def __lshift__(self, body) :
        return Clause(self, body)
    
    def __and__(self, rhs) :
        return And(self, rhs)
    
    def __or__(self, rhs) :
        return Or(self, rhs)
            
    def __invert__(self) :
        return Not(self)
    

class Var(Term) :
    """A Term representing a variable.
    
    :param name: name of the variable
    :type name: :class:`str`
    
    """
    
    def __init__(self, name) :
        Term.__init__(self,name)
    
    def _get_name(self) : return self.functor    
    name = property( _get_name , doc="Name of the variable.")    
        
    def apply(self, subst) :
        """Replace the variable with a value from the given substitution.
        
        :param subst: A mapping from variable names to something else
        :type subst: an object with a __getitem__ method
        :raises: whatever subst.__getitem__ raises
        :returns: the value from subst that corresponds to this variable's name
        """
        return subst[self.name]
    
    def isVar(self) :
        """Checks whether this Term represents a variable.
        
        :returns: ``True``
        """        
        return True
        
class Constant(Term) :
    """A constant. 
    
        :param value: the value of the constant
        :type value: :class:`str`, :class:`float` or :class:`int`.
        
    """
    
    def __init__(self, value) :
        Term.__init__(self,value)
    
    def _get_value(self) : return self.functor        
    value = property( _get_value , doc="Value of the constant.")
                
    def isConstant(self) :
        """Checks whether this Term represents a constant.
        
        :returns: True
        """
        return True
    
    def __str__(self) :
        if type(self.functor) == int or type(self.functor) == float :
            return str(self.functor)
        else :
            return '"%s"' % self.functor
        
    def isString(self) :
        """Check whether this constant is a string.
        
            :returns: true if the value represents a string
            :rtype: :class:`bool`
        """
        return type(self.value) == str
        
    def isFloat(self) :
        """Check whether this constant is a float.
        
            :returns: true if the value represents a float
            :rtype: :class:`bool`
        """
        return type(self.value) == float
        
    def isInteger(self) :
        """Check whether this constant is an integer.
        
            :returns: true if the value represents an integer
            :rtype: :class:`bool`
        """
        return type(self.value) == int
        
class Clause(Term) :
    """A clause."""
    
    def __init__(self, head, body) :
        Term.__init__(self,':-',head,body)
        self.head = head
        self.body = body
        
    def __repr__(self) :
        return "%s :- %s" % (self.head, self.body)
        
class Or(Term) :
    """Or"""
    
    def __init__(self, op1, op2) :
        Term.__init__(self, ';', op1, op2)
        self.op1 = op1
        self.op2 = op2
    
    def __or__(self, rhs) :
        self.op2 = self.op2 | rhs
        return self
        
    def __and__(self, rhs) :
        return And(self, rhs)
            
    def __repr__(self) :
        lhs = str(self.op1)
        rhs = str(self.op2)        
        return "%s; %s" % (lhs, rhs)
        
    
class And(Term) :
    """And"""
    
    def __init__(self, op1, op2) :
        Term.__init__(self, ',', op1, op2)
        self.op1 = op1
        self.op2 = op2
    
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
        
class Not(Term) :
    """Not"""
    
    def __init__(self, child) :
        Term.__init__(self, '\+', child)
        self.child = child
    
    def __repr__(self) :
        c = str(self.child)
        if isinstance(self.child, And) :
            c = '(%s)' % c
        return '\+(%s)' % c

class LogicProgram(object) :
    """LogicProgram"""
    
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