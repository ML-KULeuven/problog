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

Part of the ProbLog distribution.

Copyright 2015 KU Leuven, DTAI Research Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function
from __future__ import division  # consistent behaviour of / and // in python 2 and 3

import math
import sys

from .util import OrderedSet
from .core import GroundingError

from collections import deque


class InstantiationError(GroundingError):
    pass


class ArithmeticError(GroundingError):
    pass


def term2str(term):
    """
    Convert a Term argument to string.
    This also works for variables represented as None or an integer.

    :param term: the term to convert
    :type term: Term | None | int
    :return: string representation of the given term where None is converted to '_'.
    :rtype: str
    """
    if term is None:
        return '_'
    elif type(term) is int:
        if term >= 0:
            return 'A%s' % (term+1)
        else:
            return 'X%s' % (-term)
    elif isinstance(term, And):
        return '(%s)' % term
    else:
        return str(term)


def list2term(lst):
    """
    Transform a Python list of terms in to a Prolog Term.
    :param lst: list of Terms
    :type lst: list of Term
    :return: Term representing a Prolog list
    :rtype: Term
    """
    tail = Term('[]')
    for e in reversed(lst):
        tail = Term('.', e, tail)
    return tail


def term2list(term):
    """
    Transform a Prolog list to a Python list of terms.
    :param term: term representing a fixed length Prolog list
    :type term: Term
    :raise ValueError: given term is not a valid fixed length Prolog list
    :return: Python list containing the elements from the Prolog list
    :rtype: list of Term
    """
    result = []
    while term.functor == '.' and term.arity == 2:
        result.append(term.args[0])
        term = term.args[1]
    if not term == Term('[]'):
        raise ValueError('Expected fixed list.')
    return result


def is_ground(*terms):
    """Test whether a any of given terms contains a variable.
    :param terms: list of terms to test for the presence of variables
    :param terms: tuple of (Term | int | None)
    :return: True if none of the arguments contains any variables.
    """
    for term in terms:
        if is_variable(term):
            return False
        elif not term.is_ground():
            return False
    return True


def is_variable(term):
    """Test whether a Term represents a variable.

    :param term: term to check
    :return: True if the expression is a variable
    """
    return term is None or type(term) == int


def is_list(term):
    return not is_variable(term) and term.functor == '.' and term.arity == 2


class Term(object):

    def __init__(self, functor, *args, **kwdargs):
        """
        A first order term, for example 'p(X,Y)'.
        :param functor: the functor of the term ('p' in the example)
        :type functor: str
        :param args: the arguments of the Term ('X' and 'Y' in the example)
        :type args: tuple of (Term | None | int)
        :param kwdargs: additional arguments; currently 'p' (probability) and 'location' (character position in input)
        """
        self.__functor = functor
        self.__args = args
        self.__arity = len(self.__args)
        self.probability = kwdargs.get('p')
        self.location = kwdargs.get('location')
        self.__signature = None
        self.__hash = None
        self._cache_is_ground = None
        self._cache_list_length = None
        self._cache_variables = None

    @property
    def functor(self):
        """Term functor"""
        return self.__functor

    @functor.setter
    def functor(self, value):
        self.__functor = value
        self.__signature = None
        self.__hash = None

    @property
    def args(self):
        """Term arguments"""
        return self.__args

    @property
    def arity(self):
        """Number of arguments"""
        return self.__arity

    @property
    def value(self):
        """Value of the Term obtained by computing the function is represents"""
        return computeFunction(self.functor, self.args)

    def compute_value(self, functions=None):
        return computeFunction(self.functor, self.args, functions)
    
    @property
    def signature(self):
        """Term's signature ``functor/arity``"""
        if self.__signature is None:
            functor = str(self.functor)
            self.__signature = '%s/%s' % (functor.strip("'"), self.arity)
        return self.__signature

    def apply(self, subst):
        """Apply the given substitution to the variables in the term.

        :param subst: A mapping from variable names to something else
        :type subst: an object with a __getitem__ method
        :raises: whatever subst.__getitem__ raises
        :returns: a new Term with all variables replaced by their values from the given substitution
        :rtype: :class:`Term`

        """
        if self.is_ground() and self.probability is None:
            # No variables to substitute.
            return self

        old_stack = [deque([self])]
        new_stack = []
        term_stack = []
        while old_stack:
            current = old_stack[-1].popleft()
            if current is None or type(current) == int:
                if new_stack:
                    new_stack[-1].append(subst[current])
                else:
                    return subst[current]
            elif current.isVar():
                if new_stack:
                    new_stack[-1].append(subst[current.name])
                else:
                    return subst[current.name]
            else:
                # Add arguments to stack
                term_stack.append(current)
                q = deque(current.args)
                if current.probability is not None:
                    q.append(current.probability)
                old_stack.append(q)
                new_stack.append([])
            while old_stack and not old_stack[-1]:
                old_stack.pop(-1)
                new_args = new_stack.pop(-1)
                term = term_stack.pop(-1)
                if term.probability is not None:
                    new_term = term.withArgs(*new_args[:-1], p=new_args[-1])
                else:
                    new_term = term.withArgs(*new_args)
                if new_stack:
                    new_stack[-1].append(new_term)
                else:
                    return new_term

    def __repr__(self):
        # Non-recursive version of __repr__
        stack = [deque([self])]
        # current: popleft from stack[-1]
        # arguments: new level on stack
        parts = []
        put = parts.append
        while stack:
            current = stack[-1].popleft()
            if current is None:
                put('_')
            elif type(current) == str:
                put(current)
            elif type(current) == int:
                if current < 0:
                    put('X%s' % -current)
                else:
                    put('A%s' % (current+1))
            elif type(current) == And:  # Depends on level
                q = deque()
                q.append('(')
                if type(current.args[0]) == Or:
                    q.append('(')
                    q.append(current.args[0])
                    q.append(')')
                else:
                    q.append(current.args[0])
                tail = current.args[1]
                while isinstance(tail, Term) and tail.functor == ',' and tail.arity == 2:
                    q.append(', ')
                    if type(tail.args[0]) == Or:
                        q.append('(')
                        q.append(tail.args[0])
                        q.append(')')
                    else:
                        q.append(tail.args[0])
                    tail = tail.args[1]
                q.append(', ')
                if type(tail) == Or:
                    q.append('(')
                    q.append(tail)
                    q.append(')')
                else:
                    q.append(tail)
                q.append(')')
                stack.append(q)
            elif type(current) == Or:
                q = deque()
                q.append(current.args[0])
                tail = current.args[1]
                while isinstance(tail, Term) and tail.functor == ';' and tail.arity == 2:
                    q.append('; ')
                    q.append(tail.args[0])
                    tail = tail.args[1]
                q.append('; ')
                q.append(tail)
                stack.append(q)
            elif isinstance(current, Term) and current.functor == '.' and current.arity == 2:  # List
                q = deque()
                q.append('[')
                q.append(current.args[0])
                tail = current.args[1]
                while isinstance(tail, Term) and tail.functor == '.' and tail.arity == 2:
                    q.append(', ')
                    q.append(tail.args[0])
                    tail = tail.args[1]
                if not tail == Term('[]'):
                    q.append(' | ')
                    q.append(tail)
                q.append(']')
                stack.append(q)
            elif isinstance(current, Term):
                if current.probability is not None:
                    put(str(current.probability))       # This is a recursive call.
                    put('::')
                put(str(current.functor))
                if current.args:
                    q = deque()
                    q.append('(')
                    q.append(current.args[0])
                    for a in current.args[1:]:
                        q.append(',')
                        q.append(a)
                    q.append(')')
                    stack.append(q)
            else:
                put(str(current))
            while stack and not stack[-1]:
                stack.pop(-1)
        return ''.join(parts)

    def __call__(self, *args, **kwdargs):
        """
        Create a new Term with the same functor and the given arguments.
        :param args: new arguments
        :type args: tuple of (Term | None | int)
        :return:
        :rtype: Term
        """
        return self.withArgs(*args, **kwdargs)

    def withArgs(self, *args, **kwdargs):
        """Creates a new Term with the same functor and the given arguments.

        :param args: new arguments for the term
        :type args: tuple of (Term | int | None)
        :param kwdargs: keyword arguments for the term
        :type kwdargs: p=Constant | p=Var | p=float
        :returns: a new term with the given arguments
        :rtype: :class:`Term`

        """
        if not kwdargs and list(map(id, args)) == list(map(id, self.args)):
            return self

        if 'p' in kwdargs:
            p = kwdargs['p']
            if type(p) == float:
                p = Constant(p)
        else:
            p = self.probability
        if p is not None:
            return self.__class__(self.functor, *args, p=p, location=self.location)
        else:
            return self.__class__(self.functor, *args, location=self.location)

    def withProbability(self, p=None):
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

    def is_ground(self):
        """Checks whether the term contains any variables."""
        if self._cache_is_ground is None:
            queue = deque([self])
            while queue:
                term = queue.popleft()
                if term is None or type(term) == int or term.isVar():
                    self._cache_is_ground = False
                    return False
                elif not term._cache_is_ground:
                    queue.extend(term.args)
            self._cache_is_ground = True
            return True
        else:
            return self._cache_is_ground

    def variables(self):
        """
        Extract the variables present in the term.
        :return: set of variables
        :rtype: OrderedSet
        """
        if self._cache_variables is None:
            variables = OrderedSet()
            queue = deque([self])
            while queue:
                term = queue.popleft()
                if term is None or type(term) == int or term.isVar():
                    variables.add(term)
                else:
                    queue.extend(term.args)
                    if term.probability:
                        queue.append(term.probability)
            self._cache_variables = variables
        return self._cache_variables

    def _list_length(self):
        if self._cache_list_length is None:
            l = 0
            current = self
            while not is_variable(current) and current.functor == '.' and current.arity == 2:
                if current._cache_list_length is not None:
                    return l + current._cache_list_length
                l += 1
                current = current.args[1]
            self._cache_list_length = l
        return self._cache_list_length

    def _list_decompose(self):
        elements = []
        current = self
        while not is_variable(current) and current.functor == '.' and current.arity == 2:
            elements.append(current.args[0])
            current = current.args[1]
        return elements, current

    def __eq__(self, other):
        if not isinstance(other, Term):
            return False
        # Non-recursive version of equality check.
        l1 = deque([self])
        l2 = deque([other])
        while l1 and l2:
            t1 = l1.popleft()
            t2 = l2.popleft()
            if len(l1) != len(l2):
                return False
            elif id(t1) == id(t2):
                pass
            elif type(t1) != type(t2):
                return False
            elif type(t1) == int:
                if t1 != t2:
                    return False
            elif t1 is None:
                if t2 is not None:
                    return False
            elif isinstance(t1, Constant):  # t2 too
                if type(t1.functor) != type(t2.functor):
                    return False
                elif t1.functor != t2.functor:
                    return False
            else:  # t1 and t2 are Terms
                if t1.__functor != t2.__functor:
                    return False
                if t1.__arity != t2.__arity:
                    return False
                l1.extend(t1.__args)
                l2.extend(t2.__args)
        return l1 == l2  # Should both be empty.

    def __eq__list(self, other):

        if self._list_length() != other._list_length():
            return False

        elems1, tail1 = self._list_decompose()
        elems2, tail2 = other._list_decompose()

        if tail1 != tail2:
            return False
        else:
            for e1, e2 in zip(elems1, elems2):
                if e1 != e2:
                    return False
        return True



    def __eq__rec(self, other) :
        # TODO: this can be very slow?
        if other is None :
            return False
        try :
            return (self.functor, self.args) == (other.functor, other.args)
        except AttributeError :
            # Other is wrong type
            return False

    def __hash__(self):
        if self.__hash is None:
            self.__hash = hash((self.__functor, self.__arity, self._list_length()))
        return self.__hash

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

    def is_positive(self) :
        return not self.is_negative()

    def is_negative(self) :
        return False

    def __neg__(self) :
        return Not('\+', self)

    def __abs__(self) :
        return self

class Var(Term) :
    """A Term representing a variable.

    :param name: name of the variable
    :type name: :class:`str`

    """

    def __init__(self, name, location=None) :
        Term.__init__(self,name, location=location)

    @property
    def name(self) :
        """Name of the variable"""
        return self.functor

    @property
    def value(self) :
        """Value of the constant."""
        raise ValueError('Variables do not support evaluation: {}.'.format(self.name))

    def compute_value(self, functions=None):
        """Value of the constant."""
        raise ValueError('Variables do not support evaluation: {}.'.format(self.name))

    def isVar(self) :
        """Checks whether this Term represents a variable.

        :returns: ``True``
        """
        return True

    def isGround(self) :
        return False

    def __hash__(self) :
        return hash(self.name)

    def __eq__(self, other) :
        return str(other) == str(self)

class Constant(Term) :
    """A constant.

        :param value: the value of the constant
        :type value: :class:`str`, :class:`float` or :class:`int`.

    """

    def __init__(self, value, location=None) :
        Term.__init__(self,value,location=location)

    @property
    def value(self) :
        """Value of the constant."""
        return self.functor

    def compute_value(self, functions=None):
        return self.functor
                
    def isConstant(self) :
        """Checks whether this Term represents a constant.

        :returns: True
        """
        return True

    def __hash__(self) :
        return hash(self.functor)

    def __str__(self) :
        return str(self.functor)

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

    def __init__(self, head, body, location=None) :
        Term.__init__(self,':-',head,body,location=location)
        self.head = head
        self.body = body

    def __repr__(self) :
        return "%s :- %s" % (self.head, self.body)

class AnnotatedDisjunction(Term) :

    def __init__(self, heads, body, location=None) :
        Term.__init__(self, '<-', heads, body,location=location)
        self.heads = heads
        self.body = body

    def __repr__(self) :
        return "%s <- %s" % ('; '.join(map(str,self.heads)), self.body)

class Or(Term) :
    """Or"""

    def __init__(self, op1, op2, location=None) :
        Term.__init__(self, ';', op1, op2, location=location)
        self.op1 = op1
        self.op2 = op2

    @classmethod
    def fromList(cls, lst):
        if lst :
            n = len(lst) - 1
            tail = lst[n]
            while n > 0 :
                n -= 1
                tail = Or(lst[n],tail)
            return tail
        else :
            return Term('fail')

    def toList(self):
        body = []
        current = self
        while isinstance(current, Term) and current.functor == self.functor:
            body.append(current.args[0])
            current = current.args[1]
        body.append(current)
        return body

    def __or__(self, rhs) :
        return Or(self.op1, self.op2 | rhs)

    def __and__(self, rhs) :
        return And(self, rhs)

    def __repr__(self) :
        lhs = term2str(self.op1)
        rhs = term2str(self.op2)
        return "%s; %s" % (lhs, rhs)

    def withArgs(self, *args):
        """Creates a new Term with the same functor and the given arguments.

        :param args: new arguments for the term
        :type args: tuple of (Term | int | None)
        :returns: a new term with the given arguments
        :rtype: :class:`Term`

        """
        return self.__class__(*args, location=self.location)



class And(Term) :
    """And"""

    def __init__(self, op1, op2, location=None) :
        Term.__init__(self, ',', op1, op2, location=location)
        self.op1 = op1
        self.op2 = op2

    @classmethod
    def fromList(self, lst) :
        if lst :
            n = len(lst) - 1
            tail = lst[n]
            while n > 0 :
                n -= 1
                tail = And(lst[n],tail)
            return tail
        else :
            return Term('true')

    def __and__(self, rhs) :
        return And(self.op1, self.op2 & rhs)

    def __or__(self, rhs) :
        return Or(self, rhs)

    def __repr__(self) :
        lhs = term2str(self.op1)
        rhs = term2str(self.op2)
        if isinstance(self.op2, Or) :
            rhs = '(%s)' % rhs
        if isinstance(self.op1, Or) :
            lhs = '(%s)' % lhs

        return "%s, %s" % (lhs, rhs)

    def withArgs(self, *args):
        """Creates a new Term with the same functor and the given arguments.

        :param args: new arguments for the term
        :type args: tuple of (Term | int | None)
        :returns: a new term with the given arguments
        :rtype: :class:`Term`

        """
        return self.__class__(*args, location=self.location)

class Not(Term) :
    """Not"""

    def __init__(self, functor, child, location=None) :
        Term.__init__(self, functor, child, location=location)
        self.child = child

    def __repr__(self):
        c = str(self.child)
        if isinstance(self.child, And) or isinstance(self.child, Or):
            c = '(%s)' % c
        if self.functor == 'not':
            return 'not %s' % c
        else:
            return '%s%s' % (self.functor, c)

    def is_negative(self) :
        return True

    def __neg__(self) :
        return self.child

    def __abs__(self) :
        return -self


class LogicProgram(object):
    """LogicProgram"""

    def __init__(self, source_root='.', source_files=None, line_info=None):
        if source_files is None:
            source_files = [None]
        if line_info is None:
            line_info = [None]
        self.source_root = source_root
        self.source_files = source_files
        self.source_parent = [None]
        # line_info should be array, corresponding to 'source_files'.
        self.line_info = line_info

    def __iter__(self):
        """Iterator for the clauses in the program."""
        raise NotImplementedError("LogicProgram.__iter__ is an abstract method." )

    def add_clause(self, clause):
        """Add a clause to the logic program."""
        raise NotImplementedError("LogicProgram.addClause is an abstract method." )

    def add_fact(self, fact):
        """Add a fact to the logic program."""
        raise NotImplementedError("LogicProgram.addFact is an abstract method." )

    def __iadd__(self, clausefact):
        """Add clause or fact using the ``+=`` operator."""
        if isinstance(clausefact, Or):
            heads = clausefact.toList()
            # TODO move this to parser code
            for head in heads:
                if not type(head) == Term:
                    # TODO compute correct location
                    raise GroundingError("Unexpected fact '%s'" % head)
                elif len(heads) > 1 and head.probability is None:
                    raise GroundingError("Non-probabilistic head in multi-head clause '%s'" % head)
            self.add_clause(AnnotatedDisjunction(heads, Term('true')))
        elif isinstance(clausefact, AnnotatedDisjunction):
            self.add_clause(clausefact)
        elif isinstance(clausefact, Clause):
            self.add_clause(clausefact)
        elif type(clausefact) == Term:
            self.add_fact(clausefact)
        else:
            raise GroundingError("Unexpected fact '%s'" % clausefact, self.lineno(clausefact.location))
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
        if not force_copy and src.__class__ == cls:
            return src
        else :
            obj = cls(**extra)
            if hasattr(src, 'source_root') and hasattr(src, 'source_files'):
                obj.source_root = src.source_root
                obj.source_files = src.source_files[:]
                obj.source_parent = src.source_parent[:]
            if hasattr(src, 'line_info'):
                obj.line_info = src.line_info[:]
            for clause in src:
                obj += clause
            return obj

    def lineno(self, char, force_filename=False):
        """Transform character position to line:column format.

        :param char: character position
        :return: line, column (or None if information is not available)
        """
        # Input should be tuple (file, char)
        if isinstance(char, tuple):
            fn, char = char
        else:
            fn = 0

        if self.line_info[fn] is None or char is None:
            # No line info available
            return None
        else:
            import bisect
            i = bisect.bisect_right(self.line_info[fn], char)
            lineno = i
            charno = char - self.line_info[fn][i-1]
            if fn == 0 and not force_filename:
                filename = None
            else:
                filename = self.source_files[fn]
            return filename, lineno, charno

functions = {
    ("+", 2) : (lambda a,b : a + b),
    ("-", 2) : (lambda a,b : a - b),
    ("/\\", 2) : (lambda a,b : a & b),
    ("\\/", 2) : (lambda a,b : a | b),
    ("xor", 2) : (lambda a,b : a ^ b),
    ("xor", 2) : (lambda a,b : a ^ b),
    ("#", 2) : (lambda a,b : a ^ b),
    ("><", 2) : (lambda a,b : a ^ b),
    ("*", 2) : (lambda a,b : a * b),
    ("/", 2) : (lambda a,b : a / b),
    ("//", 2) : (lambda a,b : a // b),
    ("<<", 2) : (lambda a,b : a << b),
    (">>", 2) : (lambda a,b : a >> b),
    ("mod", 2) : (lambda a,b : a % b),
    ("mod", 2) : (lambda a,b : a % b),
    ("rem", 2) : (lambda a,b : a % b),
    ("rem", 2) : (lambda a,b : a % b),
    ("div", 2) : (lambda a, b : ( a - (a % b) ) // b),
    ("div", 2) : (lambda a, b : ( a - (a % b) ) // b),
    ("**", 2) : (lambda a,b : a ** b),
    ("^", 2) : (lambda a,b : a ** b),
    ("+", 1) : (lambda a : a),
    ("-", 1) : (lambda a : -a),
    ("\\", 1) : (lambda a : ~a),
    ("atan",2) : math.atan2,
    ("atan2",2) : math.atan2,
    ("integer",1) : int,
    ("float",1) : float,
    ("float_integer_part",1) : lambda f : int(f) ,
    ("float_fractional_part",1) : lambda f : f - int(f),
    ("abs",1) : abs,
    ("ceiling",1) : math.ceil,
    ("round",1) : round,
    ("truncate",1) : math.trunc,
    ("min",2) : min,
    ("max",2) : max,
    ("exp",2) : math.pow,
    ("epsilon",0) : lambda : sys.float_info.epsilon,
    ("inf",0) : lambda : float('inf'),
    ("nan",0) : lambda : float('nan'),
    ("sign",1): lambda x: x if x >= 0 else -x

}

from_math_1 = ['exp', 'log', 'log10', 'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh',
             'tanh', 'asinh', 'acosh', 'atanh', 'lgamma', 'gamma', 'erf', 'erfc', 'floor'  ]
for f in from_math_1 :
    functions[(f,1)] = getattr(math,f)

from_math_0 = [ 'pi', 'e' ]
for f in from_math_0 :
    functions[(f,0)] = lambda: getattr(math,f)

def unquote(s) :
    return s.strip("'")


def computeFunction(func, args, extra_functions=None):
    """Compute the result of an arithmetic function given by a functor and a list of arguments.

    :param func: functor
    :type: basestring
    :param args: arguments
    :type args: list of Term
    :raises: ArithmeticError if the function is unknown or if an error occurs while computing it
    :return: result of the function
    :rtype: Constant

    Currently the following functions are supported:
        ``+/2``, ``-/2``, ``/\/2``, ``\//2``, ``xor/2``, ``*/2``, ``//2``,
        ``///2``, ``<</2``, ``>>/2``, ``mod/2``, ``rem/2``, ``**/2``, ``^/2``,
        ``+/1``, ``-/1``, ``\\/1``.

    """

    if extra_functions is None:
        extra_functions = {}

    function = functions.get((unquote(func), len(args)))
    if function is None:
        function = extra_functions.get((unquote(func), len(args)))
        if function is None:
            raise ArithmeticError("Unknown function '%s'/%s" % (func, len(args)))
    try:
        values = [arg.compute_value(extra_functions) for arg in args]
        if None in values:
            return None
        else:
            return function(*values)
    except ZeroDivisionError:
        raise ArithmeticError("Division by zero.")
