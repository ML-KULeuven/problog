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
from __future__ import division

from collections import defaultdict

class Term(object) :
    """Represent a first-order Term."""
    
    def __init__(self, functor, *args, **kwdargs) :
        self.__functor = functor
        self.__args = args
        self.__probability = kwdargs.get('p')
    
    @property
    def functor(self) : 
        """Term functor"""
        return self.__functor

    @property
    def args(self) : 
        """Term arguments"""
        return self.__args
    
    @property
    def arity(self) : 
        """Number of arguments."""
        return len(self.__args)
    
    @property
    def value(self) : 
        """Value of the Term obtained by computing the function is represents."""
        return computeFunction(self.functor, self.args)        

    @property
    def signature(self) : 
        """Term's signature ``functor/arity``"""
        if type(self.functor) == str :
            return '%s/%s' % (self.functor.strip("'"), self.arity)
        else :
            return '%s/%s' % (self.functor, self.arity)
    
    @property
    def probability(self) : 
        """Term's probability"""
        return self.__probability
    
    @probability.setter
    def probability(self, p) : 
        self.__probability = p
        
    def apply(self, subst) :
        """Apply the given substitution to the variables in the term.
        
        :param subst: A mapping from variable names to something else
        :type subst: an object with a __getitem__ method
        :raises: whatever subst.__getitem__ raises
        :returns: a new Term with all variables replaced by their values from the given substitution
        :rtype: :class:`Term`
        
        """
        
        args = []
        for arg in self.args :
            if not isinstance(arg, Term) :
                args.append( subst[arg] )
            else :
                args.append( arg.apply(subst) )
        
        if self.probability == None :
            return self.__class__( self.functor, *args)
        else :
            return self.__class__( self.functor, *args, p=self.probability.apply(subst))
            
    def __repr__(self) :
        if self.probability == None :
            prob = ''
        else :
            prob = '%s::' % self.probability
        
        if self.args :
            return '%s%s(%s)' % (prob, self.functor, ','.join(map(str,self.args)))
        else :
            return '%s%s' % (prob, self.functor,)
        
    def __call__(self, *args) :
        return self.withArgs(*args)
        
    def withArgs(self,*args) :
        """Creates a new Term with the same functor and the given arguments.
        
        :param args: new arguments for the term
        :type args: any
        :returns: a new term with the given arguments
        :rtype: :class:`Term`
        
        """
        if self.probability != None :
            return self.__class__(self.functor, *args, p=self.probability)
        else :
            return self.__class__(self.functor, *args)
            
    def withProbability(self, p=None) :
        return self.__class__(self.functor, *self.args, p=p)
        
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
        
    def isGround(self) :
        """Checks whether the term contains any variables."""
        for arg in self.args :
            if not isinstance(arg,Term) or not arg.isGround() :
                return False
        return True
        
    def __eq__(self, other) :
        # TODO: this can be very slow?
        if not isinstance(other,Term) :
            return False
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
    
    def __float__(self) :
        return float(self.value)
        
    def __int__(self) :
        return int(self.value)
    
class Var(Term) :
    """A Term representing a variable.
    
    :param name: name of the variable
    :type name: :class:`str`
    
    """
    
    def __init__(self, name) :
        Term.__init__(self,name)
    
    @property
    def name(self) : 
        """Name of the variable"""
        return self.functor    

    @property
    def value(self) : 
        """Value of the constant."""
        raise ValueError('Variables do not support evaluation.')
       
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
        
    def isGround(self) :
        return False
        
class Constant(Term) :
    """A constant. 
    
        :param value: the value of the constant
        :type value: :class:`str`, :class:`float` or :class:`int`.
        
    """
    
    def __init__(self, value) :
        Term.__init__(self,value)
    
    @property
    def value(self) : 
        """Value of the constant."""
        return self.functor
                
    def isConstant(self) :
        """Checks whether this Term represents a constant.
        
        :returns: True
        """
        return True
    
    def __hash__(self) :
        return hash(self.functor)
    
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
        
    def __eq__(self, other) :
        return str(self) == str(other)
                
class Clause(Term) :
    """A clause."""
    
    def __init__(self, head, body) :
        Term.__init__(self,':-',head,body)
        self.head = head
        self.body = body
        
    def __repr__(self) :
        return "%s :- %s" % (self.head, self.body)
        
class AnnotatedDisjunction(Term) :
    
    def __init__(self, heads, body) :
        Term.__init__(self, '<-', heads, body)
        self.heads = heads
        self.body = body
        
    def __repr__(self) :
        return "%s <- %s" % ('; '.join(map(str,self.heads)), self.body)
        
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

    def _addAnnotatedDisjunction(self, clause) :
        """Add a clause to the logic program."""
        raise NotImplementedError("LogicProgram.addClause is an abstract method." )
        
    def _addClause(self, clause) :
        """Add a clause to the logic program."""
        raise NotImplementedError("LogicProgram.addClause is an abstract method." )
        
    def _addFact(self, fact) :
        """Add a fact to the logic program."""
        raise NotImplementedError("LogicProgram.addFact is an abstract method." )
    
    def _uncurry(self, term, func=None) :
        if func == None : func = term.functor
        
        body = []
        current = term
        while isinstance(current, Term) and current.functor == func :
            body.append(current.args[0])
            current = current.args[1]
        body.append(current)
        return body    
        
    def __iadd__(self, clausefact) :
        """Add clause or fact using the ``+=`` operator."""
        if isinstance(clausefact, Or) :
            heads = self._uncurry( clausefact, ';')
            self._addAnnotatedDisjunction( AnnotatedDisjunction(heads, Term('true') ))
        elif isinstance(clausefact, AnnotatedDisjunction) :
            self._addAnnotatedDisjunction(clausefact)
        elif isinstance(clausefact, Clause) :
            self._addClause(clausefact)
        else :
            self._addFact(clausefact)
        return self
    
    @classmethod    
    def createFrom(cls, src, force_copy=False, **extra) :
        """Create a LogicProgram of the current class from another LogicProgram.
        
        :param lp: logic program to convert
        :type lp: :class:`.LogicProgram`
        :param force_copy: default False, If true, always create a copy of the original logic program.
        :type force_copy: bool
        :returns: LogicProgram that is (externally) identical to given one
        :rtype: object of the class on which this method is invoked
        
        If the original LogicProgram already has the right class and force_copy is False, then the original program is returned.
        """
        if not force_copy and src.__class__ == cls :
            return src
        else :
            obj = cls(**extra)
            for clause in src :
                obj += clause
            return obj
    
                    
def computeFunction(func, args) :
    """Compute the result of an arithmetic function given by a functor and a list of arguments.
    
        :raises: ValueError if the function is unknown or the arguments can not be computed
        :returns: the result of the function
        :rtype: Constant
        
    Currently the following functions are supported: ``+/2``, ``-/2``, ``/\/2``, ``\//2``, ``xor/2``, ``*/2``, ``//2``, ``///2``, ``<</2``, ``>>/2``, ``mod/2``, ``rem/2``, ``**/2``, ``^/2``, ``+/1``, ``-/1``, ``\\/1``.
    
    """
    
    functions = {
        ("'+'", 2) : (lambda a,b : a + b),
        ("'-'", 2) : (lambda a,b : a - b),
        ("'/\\'", 2) : (lambda a,b : a & b),
        ("'\\/'", 2) : (lambda a,b : a | b),
        ("'xor'", 2) : (lambda a,b : a ^ b),
        ("'*'", 2) : (lambda a,b : a * b),
        ("'/'", 2) : (lambda a,b : a / b),
        ("'//'", 2) : (lambda a,b : a // b),        
        ("'<<'", 2) : (lambda a,b : a << b),
        ("'>>'", 2) : (lambda a,b : a >> b),
        ("'mod'", 2) : (lambda a,b : a % b),
        ("'rem'", 2) : (lambda a,b : a % b),
        ("'**'", 2) : (lambda a,b : a ** b),
        ("'^'", 2) : (lambda a,b : a ** b),
        ("'+'", 1) : (lambda a : a),
        ("'-'", 1) : (lambda a : -a),
        ("'\\'", 1) : (lambda a : ~a),
    }
    
    function = functions.get( (func, len(args) ) )
    if function == None :
        raise ValueError("Unknown function: '%s'/%s" % (func, len(args)) )
    else :
        values = [ arg.value for arg in args ]
        return function(*values)
