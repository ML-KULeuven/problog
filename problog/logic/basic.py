"""
This module contains basic logic constructs.

    A Term can be:
        * a function (see :class:`Term`)
        * a variable (see :class:`Var`)
        * a constant (see :class:`Constant`)

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
        
    @classmethod
    def create(self, functor) :
        """Create factory for a given functor.
        
        :param functor: functor of the Terms to be created with the factory
        :type functor: :class:`str`
        :returns: factory function that accepts a list of arguments
        :rtype: callable
        
        Example usage:
        
        >>> f = Term.create('f')
        >>> f(1,2)
        f(1,2)
        >>> f(1,2,3)
        f(1,2,3)
        
        """
        return lambda *args : Term(functor, *args)
            
    def __repr__(self) :
        if self.args :
            return '%s(%s)' % (self.functor, ','.join(map(str,self.args)))
        else :
            return '%s' % (self.functor,)
                    
    def withArgs(self,*args) :
        """Creates a new Term with the same functor and the given arguments.
        
        :param args: new arguments for the term
        :type args: any
        :returns: a new term with the given arguments
        :rtype: :class:`Term`
        
        """
        return Term(self.functor, *args)
        
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
    
    def withArgs(self,*args) :
        """Return this variable.
        
        :returns: the variable itself
        """
        return self
                
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
    
    def withArgs(self,*args) :
        """Return this constant.
        
        :returns: the constant itself
        """
        return self
        
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
