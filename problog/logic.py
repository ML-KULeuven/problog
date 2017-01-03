"""

problog.logic - Basic logic
---------------------------

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


..
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
from .errors import GroundingError

from collections import deque


def term2str(term):
    """Convert a term argument to string.

    :param term: the term to convert
    :type term: Term | None | int
    :return: string representation of the given term where None is converted to '_'.
    :rtype: str
    """
    if term is None:
        return '_'
    elif type(term) is int:
        if term >= 0:
            return 'A%s' % (term + 1)
        else:
            return 'X%s' % (-term)
    else:
        return str(term)


def list2term(lst):
    """Transform a Python list of terms in to a Prolog Term.

    :param lst: list of Terms
    :type lst: list of Term
    :return: Term representing a Prolog list
    :rtype: Term
    """
    from .pypl import py2pl
    tail = Term('[]')
    for e in reversed(lst):
        tail = Term('.', py2pl(e), tail)
    return tail


def term2list(term):
    """Transform a Prolog list to a Python list of terms.

    :param term: term representing a fixed length Prolog list
    :type term: Term
    :raise ValueError: given term is not a valid fixed length Prolog list
    :return: Python list containing the elements from the Prolog list
    :rtype: list of Term
    """
    from .pypl import pl2py
    result = []
    while not is_variable(term) and term.functor == '.' and term.arity == 2:
        result.append(pl2py(term.args[0]))
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
    return term is None or type(term) == int or term.is_var()


def is_list(term):
    """Test whether a Term is a list.

    :param term: term to check
    :return: True if the term is a list.
    """
    return not is_variable(term) and term.functor == '.' and term.arity == 2


class Term(object):
    """
    A first order term, for example 'p(X,Y)'.
    :param functor: the functor of the term ('p' in the example)
    :type functor: str
    :param args: the arguments of the Term ('X' and 'Y' in the example)
    :type args: tuple of (Term | None | int)
    :param kwdargs: additional arguments; currently 'p' (probability) and 'location' \
    (character position in input)
    """

    def __init__(self, functor, *args, **kwdargs):
        self.__functor = functor
        self.__args = args
        self.__arity = len(self.__args)
        self.probability = kwdargs.get('p')
        self.location = kwdargs.get('location')
        self.op_priority = kwdargs.get('priority')
        self.op_spec = kwdargs.get('opspec')
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
        """Term functor

        :param value: new value
        """
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
        return self.compute_value()

    def compute_value(self, functions=None):
        """Compute value of the Term by computing the function it represents.

        :param functions: dictionary of user-defined functions
        :return: value of the Term
        """
        return compute_function(self.functor, self.args, functions)

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
            elif current.is_var():
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
                    new_term = term.with_args(*new_args[:-1], p=new_args[-1])
                else:
                    new_term = term.with_args(*new_args)
                if new_stack:
                    new_stack[-1].append(new_term)
                else:
                    return new_term

    def apply_term(self, subst):
        """Apply the given substitution to all (sub)terms in the term.

        :param subst: A mapping from variable names to something else
        :type subst: an object with a __getitem__ method
        :raises: whatever subst.__getitem__ raises
        :returns: a new Term with all variables replaced by their values from the given substitution
        :rtype: :class:`Term`

        """

        old_stack = [deque([self])]
        new_stack = []
        term_stack = []
        while old_stack:
            current = old_stack[-1].popleft()
            if current in subst:
                if new_stack:
                    new_stack[-1].append(subst[current])
                else:
                    return subst[current]
            elif current is None or type(current) == int:
                new_stack[-1].append(current)
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
                    new_term = term.with_args(*new_args[:-1], p=new_args[-1])
                else:
                    new_term = term.with_args(*new_args)
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
                    put('A%s' % (current + 1))
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
            elif isinstance(current, Term) and current.functor == '.' and current.arity == 2:
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
            elif isinstance(current, Term) and current.op_spec is not None:
                # Is a binary or unary operator.
                if len(current.op_spec) == 2:  # unary operator
                    cf = str(current.functor).strip("'")
                    if 'a' <= cf[0] <= 'z':
                        put(' ' + cf + ' ')
                    else:
                        put(cf)
                    q = deque()
                    q.append(current.args[0])
                    stack.append(q)
                else:
                    a = current.args[0]
                    b = current.args[1]
                    q = deque()
                    if not isinstance(a, Term) or a.op_priority is None or \
                            a.op_priority < current.op_priority or \
                            (a.op_priority == current.op_priority and current.op_spec == 'yfx'):
                        # no parenthesis around a
                        q.append(a)
                    else:
                        q.append('(')
                        q.append(a)
                        q.append(')')
                    op = str(current.functor).strip("'")
                    if 'a' <= op[0] <= 'z':
                        q.append(' %s ' % op)
                    else:
                        q.append('%s' % op)
                    if not isinstance(b, Term) or \
                            b.op_priority is None or b.op_priority < current.op_priority or \
                            (b.op_priority == current.op_priority and current.op_spec == 'xfy'):
                        # no parenthesis around b
                        q.append(b)
                    else:
                        q.append('(')
                        q.append(b)
                        q.append(')')
                    stack.append(q)
            elif isinstance(current, Term):
                if current.probability is not None:
                    put(str(current.probability))  # This is a recursive call.
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
        """Create a new Term with the same functor and the given arguments.

        :param args: new arguments
        :type args: tuple of (Term | None | int)
        :return:
        :rtype: Term
        """
        return self.with_args(*args, **kwdargs)

    def with_args(self, *args, **kwdargs):
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

        extra = {}
        if p is not None:
            return self.__class__(self.functor, *args, p=p, location=self.location, priority=self.op_priority, opspec=self.op_spec)
        else:
            if self.__class__ in (Clause, AnnotatedDisjunction, And, Or):
                return self.__class__(*args, location=self.location, priority=self.op_priority, opspec=self.op_spec)
            else:
                return self.__class__(self.functor, *args, location=self.location, priority=self.op_priority, opspec=self.op_spec)

    def with_probability(self, p=None):
        """Creates a new Term with the same functor and arguments but with a different probability.

        :param p: new probability (None clears the probability)
        :return: copy of the Term
        """
        return self.__class__(self.functor, *self.args, p=p, priority=self.op_priority, opspec=self.op_spec, location=self.location)

    def is_var(self):
        """Checks whether this Term represents a variable."""
        return False

    def is_constant(self):
        """Checks whether this Term represents a constant."""
        return False

    def is_ground(self):
        """Checks whether the term contains any variables."""
        if self._cache_is_ground is None:
            queue = deque([self])
            while queue:
                term = queue.popleft()
                if term is None or type(term) == int or term.is_var():
                    self._cache_is_ground = False
                    return False
                elif isinstance(term, Term):
                    if not term._cache_is_ground:
                        queue.extend(term.args)
            self._cache_is_ground = True
            return True
        else:
            return self._cache_is_ground

    def is_negated(self):
        """Checks whether the term represent a negated term."""
        return False

    def variables(self, exclude_local=False):
        """Extract the variables present in the term.

        :return: set of variables
        :rtype: :class:`problog.util.OrderedSet`
        """
        if exclude_local and self.__functor == 'findall' and self.__arity == 3:
            return self.args[2].variables()
        elif self._cache_variables is None:
            variables = OrderedSet()
            queue = deque([self])
            while queue:
                term = queue.popleft()
                if term is None or type(term) == int or term.is_var():
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

    def _eq__list(self, other):
        """Custom equivalence test for lists.

        :param other: other Term representing a list
        :type other: Term
        :return: True if lists contain the same elements, False otherwise
        """

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

    def __hash__(self):
        if self.__hash is None:
            firstarg = None
            if len(self.__args) > 0:
                firstarg = self.__args[0]

            self.__hash = hash((self.__functor, self.__arity, firstarg, self._list_length()))
        return self.__hash

    def __lshift__(self, body):
        return Clause(self, body)

    def __and__(self, rhs):
        return And(self, rhs)

    def __or__(self, rhs):
        return Or(self, rhs)

    def __invert__(self):
        return Not('\+', self)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __neg__(self):
        return Not('\+', self)

    def __abs__(self):
        return self

    @classmethod
    def from_string(cls, str, factory=None, parser=None):
        if factory is None:
            from .program import ExtendedPrologFactory
            factory = ExtendedPrologFactory()
        if parser is None:
            from .parser import PrologParser
            parser = PrologParser(factory)

        if not str.strip().endswith("."):
            str += "."

        parsed = parser.parseString(str)
        if len(parsed) != 1:
            raise ValueError("Invalid term: '" + str + "'")
        else:
            return parsed[0]


class Var(Term):
    """A Term representing a variable.

    :param name: name of the variable
    :type name: :class:`str`

    """

    def __init__(self, name, location=None, **kwdargs):
        Term.__init__(self, name, location=location, **kwdargs)

    @property
    def name(self):
        """Name of the variable"""
        return self.functor

    def compute_value(self, functions=None):
        raise InstantiationError('Variables do not support evaluation: {}.'.format(self.name))

    def is_var(self):
        return True

    def is_ground(self):
        return False

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return str(other) == str(self)


class Constant(Term):
    """A constant.

        :param value: the value of the constant
        :type value: :class:`str`, :class:`float` or :class:`int`.

    """

    FLOAT_PRECISION = 15

    def __init__(self, value, location=None, **kwdargs):
        if self.FLOAT_PRECISION is not None and type(value) == float:
            value = round(value, self.FLOAT_PRECISION)
        Term.__init__(self, value, location=location, **kwdargs)

    def compute_value(self, functions=None):
        return self.functor

    def is_constant(self):
        return True

    def __hash__(self):
        return hash(self.functor)

    def __str__(self):
        return str(self.functor)

    def is_string(self):
        """Check whether this constant is a string.

            :returns: true if the value represents a string
            :rtype: :class:`bool`
        """
        return type(self.value) == str

    def is_float(self):
        """Check whether this constant is a float.

            :returns: true if the value represents a float
            :rtype: :class:`bool`
        """
        return type(self.value) == float

    def is_integer(self):
        """Check whether this constant is an integer.

            :returns: true if the value represents an integer
            :rtype: :class:`bool`
        """
        return type(self.value) == int

    def __eq__(self, other):
        return str(self) == str(other)


class Clause(Term):
    """A clause."""

    def __init__(self, head, body, **kwdargs):
        Term.__init__(self, ':-', head, body, **kwdargs)
        self.head = head
        self.body = body

    def __repr__(self):
        return "%s :- %s" % (self.head, self.body)


class AnnotatedDisjunction(Term):
    """An annotated disjunction."""

    def __init__(self, heads, body, **kwdargs):
        Term.__init__(self, ':-', heads, body, **kwdargs)
        self.heads = heads
        self.body = body

    def __repr__(self):
        if self.body is None:
            return "%s" % ('; '.join(map(str, self.heads)))
        else:
            return "%s :- %s" % ('; '.join(map(str, self.heads)), self.body)


class Or(Term):
    """Or"""

    def __init__(self, op1, op2, **kwdargs):
        Term.__init__(self, ';', op1, op2, **kwdargs)
        self.op1 = op1
        self.op2 = op2

    @classmethod
    def from_list(cls, lst):
        """Create a disjunction based on the terms in the list.

        :param lst: list of terms
        :return: disjunction over the given terms
        """
        if lst:
            n = len(lst) - 1
            tail = lst[n]
            while n > 0:
                n -= 1
                tail = Or(lst[n], tail)
            return tail
        else:
            return Term('fail')

    def to_list(self):
        """Extract the terms of the disjunction into the list.

        :return: list of disjuncts
        """
        body = []
        current = self
        while isinstance(current, Term) and current.functor == self.functor:
            body.append(current.args[0])
            current = current.args[1]
        body.append(current)
        return body

    def __or__(self, rhs):
        return Or(self.op1, self.op2 | rhs)

    def __and__(self, rhs):
        return And(self, rhs)

    def __repr__(self):
        lhs = term2str(self.op1)
        rhs = term2str(self.op2)
        return "%s; %s" % (lhs, rhs)

    def with_args(self, *args):
        return self.__class__(*args, location=self.location)


class And(Term):
    """And"""

    def __init__(self, op1, op2, location=None, **kwdargs):
        Term.__init__(self, ',', op1, op2, location=location, **kwdargs)
        self.op1 = op1
        self.op2 = op2

    @classmethod
    def from_list(cls, lst):
        """Create a conjunction based on the terms in the list.

        :param lst: list of terms
        :return: conjunction over the given terms
        """
        if lst:
            n = len(lst) - 1
            tail = lst[n]
            while n > 0:
                n -= 1
                tail = And(lst[n], tail)
            return tail
        else:
            return Term('true')

    def to_list(self):
        """Extract the terms of the conjunction into the list.

        :return: list of disjuncts
        """
        body = []
        current = self
        while isinstance(current, Term) and current.functor == self.functor:
            body.append(current.args[0])
            current = current.args[1]
        body.append(current)
        return body

    def __and__(self, rhs):
        return And(self.op1, self.op2 & rhs)

    def __or__(self, rhs):
        return Or(self, rhs)

    def __repr__(self):
        lhs = term2str(self.op1)
        rhs = term2str(self.op2)
        if isinstance(self.op2, Or):
            rhs = '(%s)' % rhs
        if isinstance(self.op1, Or):
            lhs = '(%s)' % lhs

        return "%s, %s" % (lhs, rhs)

    def with_args(self, *args):
        return self.__class__(*args, location=self.location)


class Not(Term):
    """Not"""

    def __init__(self, functor, child, location=None, **kwdargs):
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

    def is_negated(self):
        return True

    def __neg__(self):
        return self.child

    def __abs__(self):
        return -self


_arithmetic_functions = {
    ("+", 2): (lambda a, b: a + b),
    ("-", 2): (lambda a, b: a - b),
    ("/\\", 2): (lambda a, b: a & b),
    ("\\/", 2): (lambda a, b: a | b),
    ("xor", 2): (lambda a, b: a ^ b),
    ("xor", 2): (lambda a, b: a ^ b),
    ("#", 2): (lambda a, b: a ^ b),
    ("><", 2): (lambda a, b: a ^ b),
    ("*", 2): (lambda a, b: a * b),
    ("/", 2): (lambda a, b: a / b),
    ("//", 2): (lambda a, b: a // b),
    ("<<", 2): (lambda a, b: a << b),
    (">>", 2): (lambda a, b: a >> b),
    ("mod", 2): (lambda a, b: a % b),
    ("mod", 2): (lambda a, b: a % b),
    ("rem", 2): (lambda a, b: a % b),
    ("rem", 2): (lambda a, b: a % b),
    ("div", 2): (lambda a, b: (a - (a % b)) // b),
    ("div", 2): (lambda a, b: (a - (a % b)) // b),
    ("**", 2): (lambda a, b: a ** b),
    ("^", 2): (lambda a, b: a ** b),
    ("+", 1): (lambda a: a),
    ("-", 1): (lambda a: -a),
    ("\\", 1): (lambda a: ~a),
    ("atan", 2): math.atan2,
    ("atan2", 2): math.atan2,
    ("integer", 1): int,
    ("float", 1): float,
    ("float_integer_part", 1): lambda f: int(f),
    ("float_fractional_part", 1): lambda f: f - int(f),
    ("abs", 1): abs,
    ("ceiling", 1): lambda x: int(math.ceil(x)),
    ("round", 1): lambda x: int(round(x)),
    ("floor", 1): lambda x: int(math.floor(x)),
    ("truncate", 1): lambda x: int(math.trunc),
    ("min", 2): min,
    ("max", 2): max,
    ("exp", 2): math.pow,
    ("epsilon", 0): lambda: sys.float_info.epsilon,
    ("inf", 0): lambda: float('inf'),
    ("nan", 0): lambda: float('nan'),
    ("sign", 1): lambda x: 1 if x > 0 else -1 if x < 0 else 0

}

_from_math_1 = ['exp', 'log', 'log10', 'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'lgamma', 'gamma', 'erf',
                'erfc']
for _f in _from_math_1:
    _arithmetic_functions[(_f, 1)] = getattr(math, _f)

# _from_math_0 = ['pi', 'e']
# for _f in _from_math_0:
#     _x = getattr(math, _f)
_arithmetic_functions[('pi', 0)] = lambda: math.pi
_arithmetic_functions[('e', 0)] = lambda: math.e


def unquote(s):
    """Strip single quotes from the string.

    :param s: string to remove quotes from
    :return: string with quotes removed
    """
    return s.strip("'")


def compute_function(func, args, extra_functions=None):
    """Compute the result of an arithmetic function given by a functor and a list of arguments.

    :param func: functor
    :type: basestring
    :param args: arguments
    :type args: (list | tuple) of (Term | int | None)
    :param extra_functions: additional user-defined functions
    :raises: ArithmeticError if the function is unknown or if an error occurs while computing it
    :return: result of the function
    :rtype: Constant
    """
    if extra_functions is None:
        extra_functions = {}

    function = _arithmetic_functions.get((unquote(func), len(args)))
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


class InstantiationError(GroundingError):
    """Error used when performing arithmetic with a non-ground term."""
    pass


# noinspection PyShadowingBuiltins
class ArithmeticError(GroundingError):
    """Error used when an error occurs during evaluation of an arithmetic expression."""
    pass