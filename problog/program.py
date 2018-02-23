"""
problog.program - Representation of Logic Programs
--------------------------------------------------

Provides tools for loading logic programs.

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

from .errors import GroundingError
from .logic import Term, Var, Constant, AnnotatedDisjunction, Clause, And, Or, Not
from .core import transform, ProbLogObject

from .parser import DefaultPrologParser, Factory
from .core import ProbLogError

import os
import sys


class LogicProgram(ProbLogObject):
    """LogicProgram"""

    def __init__(self, source_root='.', source_files=None, line_info=None, **extra_info):
        if source_files is None:
            source_files = [None]
        if line_info is None:
            line_info = [None]
        self.source_root = source_root
        self.source_files = source_files
        self.source_parent = [None]
        self.extra_info = extra_info
        # line_info should be array, corresponding to 'source_files'.
        self.line_info = line_info

    def __iter__(self):
        """Iterator for the clauses in the program."""
        raise NotImplementedError("LogicProgram.__iter__ is an abstract method.")

    def add_clause(self, clause):
        """Add a clause to the logic program.

        :param clause: add a clause
        """
        raise NotImplementedError("LogicProgram.addClause is an abstract method.")

    def add_fact(self, fact):
        """Add a fact to the logic program.

        :param fact: add a fact
        """
        raise NotImplementedError("LogicProgram.addFact is an abstract method.")

    def __iadd__(self, clausefact):
        """Add clause or fact using the ``+=`` operator."""
        if isinstance(clausefact, Or):
            heads = clausefact.to_list()
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
            raise GroundingError("Unexpected fact '%s'" % clausefact,
                                 self.lineno(clausefact.location))
        return self

    @classmethod
    def create_from(cls, src, force_copy=False, **extra):
        """Create a LogicProgram of the current class from another LogicProgram.

        :param src: logic program to convert
        :type src: :class:`.LogicProgram`
        :param force_copy: default False, If true, always create a copy of the original logic \
        program.
        :type force_copy: bool
        :param extra: additional arguments passed to all constructors and action functions
        :returns: LogicProgram that is (externally) identical to given one
        :rtype: object of the class on which this method is invoked

        If the original LogicProgram already has the right class and force_copy is False, then \
        the original program is returned.
        """
        return cls.createFrom(src, force_copy=force_copy, **extra)

    # noinspection PyPep8Naming
    @classmethod
    def createFrom(cls, src, force_copy=False, **extra):
        """Create a LogicProgram of the current class from another LogicProgram.

        :param src: logic program to convert
        :type src: :class:`.LogicProgram`
        :param force_copy: default False, If true, always create a copy of the original logic \
        program.
        :type force_copy: bool
        :param extra: additional arguments passed to all constructors and action functions
        :returns: LogicProgram that is (externally) identical to given one
        :rtype: object of the class on which this method is invoked

        If the original LogicProgram already has the right class and force_copy is False, then \
        the original program is returned.
        """
        if not force_copy and src.__class__ == cls:
            return src
        else:
            obj = cls(**extra)
            if hasattr(src, 'extra_info'):
                obj.extra_info.update(src.extra_info)
            if hasattr(src, 'source_root'):
                obj.source_root = src.source_root
            if hasattr(src, 'source_files'):
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
        :param force_filename: always add filename even for top-level file
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
            charno = char - self.line_info[fn][i - 1]
            if fn == 0 and not force_filename:
                filename = None
            else:
                filename = self.source_files[fn]
            return filename, lineno, charno

    def to_prolog(self):
        s = ''
        for statement in self:
            s += '%s.\n' % statement
        return s


class SimpleProgram(LogicProgram):
    """LogicProgram implementation as a list of clauses."""

    def __init__(self):
        LogicProgram.__init__(self)
        self.__clauses = []

    def add_clause(self, clause):
        """Add a clause to the logic program.

        :param clause: add a clause
        """
        if type(clause) is list:
            for c in clause:
                self.__clauses.append(c)
        else:
            self.__clauses.append(clause)

    def add_fact(self, fact):
        """Add a fact to the logic program.

        :param fact: add a fact
        """
        self.__clauses.append(fact)

    def __iter__(self):
        return iter(self.__clauses)


class PrologString(LogicProgram):
    """Read a logic program from a string of ProbLog code."""

    def __init__(self, string, parser=None, factory=None, source_root='.', source_files=None,
                 identifier=0):
        self.__string = string
        self.__program = None
        self.__identifier = identifier
        lines = [self._find_lines(string)]
        if parser is None:
            if factory is None:
                factory = DefaultPrologFactory(identifier=identifier)
            else:
                factory = factory.__class__(identifier=identifier)
            self.parser = DefaultPrologParser(factory)
        else:
            self.parser = parser

        LogicProgram.__init__(self, source_root=source_root, source_files=source_files,
                              line_info=lines, factory=factory, parser=parser)

    def _program(self):
        """Parsed program"""
        if self.__program is None:
            self.__program = self.parser.parseString(self.__string)
        return self.__program

    def _find_lines(self, s):
        """Find line-end positions."""
        lines = [-1]
        f = s.find('\n')
        while f >= 0:
            lines.append(f)
            f = s.find('\n', f + 1)
        lines.append(len(s))
        return lines

    def __iter__(self):
        """Iterator for the clauses in the program."""
        return iter(self._program())

    def __getitem__(self, sl):
        program = self._program()
        return program[sl]

    def add_clause(self, clause):
        """Add a clause to the logic program.

        :param clause: add a clause
        """
        raise AttributeError('not supported')

    def add_fact(self, fact):
        """Add a fact to the logic program.

        :param fact: add a fact
        """
        raise AttributeError('not supported')


class PrologFile(PrologString):
    """LogicProgram implementation as a pointer to a Prolog file.

    :param filename: filename of the Prolog file (optional)
    :param identifier: index of the file (in case of multiple files)
    :type filename: string
    """

    def __init__(self, filename, parser=None, factory=None, identifier=0):
        if filename == '-':
            source_root = ''
            source_files = ['-']
            source_text = sys.stdin.read()
        else:
            rootfile = os.path.abspath(filename)
            source_root = os.path.dirname(filename)
            source_files = [rootfile]
            try:
                with open(filename) as f:
                    source_text = f.read()
            except IOError as err:
                raise ProbLogError(str(err))
        PrologString.__init__(self, source_text, parser=parser, factory=factory,
                              source_root=source_root, source_files=source_files,
                              identifier=identifier)

    def add_clause(self, clause):
        """Add a clause to the logic program.

        :param clause: add a clause
        """
        raise AttributeError('not supported')

    def add_fact(self, fact):
        """Add a fact to the logic program.

        :param fact: add a fact
        """
        raise AttributeError('not supported')


class PrologFactory(Factory):
    """Factory object for creating suitable objects from the parse tree."""

    def __init__(self, identifier=0):
        self.loc_id = identifier

    def build_program(self, clauses):
        return clauses

    def build_function(self, functor, arguments, location=None, **extra):
        return Term(functor, *arguments, location=(self.loc_id, location), **extra)

    def build_variable(self, name, location=None):
        return Var(name, location=(self.loc_id, location))

    def build_constant(self, value, location=None):
        return Constant(value, location=(self.loc_id, location))

    def build_binop(self, functor, operand1, operand2, function=None, location=None, **extra):
        return self.build_function("'" + functor + "'", (operand1, operand2), location=location, **extra)

    def build_directive(self, functor, operand, **extra):
        head = self.build_function('_directive', [])
        return self.build_clause(functor, [head], operand, **extra)

    def build_unop(self, functor, operand, location=None, **extra):
        if functor == '-' and operand.is_constant() and \
                (operand.is_float() or operand.is_integer()):
            return Constant(-operand.value)
        return self.build_function("'" + functor + "'", (operand,), location=location, **extra)

    def build_list(self, values, tail=None, location=None, **extra):
        if tail is None:
            current = Term('[]')
        else:
            current = tail
        for value in reversed(values):
            current = self.build_function('.', (value, current), location=location)
        return current

    def build_string(self, value, location=None):
        return self.build_constant('"' + value + '"', location=location)

    def build_cut(self, location=None):
        raise AttributeError('not supported')

    # noinspection PyUnusedLocal
    def build_clause(self, functor, operand1, operand2, location=None, **extra):
        heads = operand1
        # TODO move this to parser code
        for head in heads:
            if not type(head) == Term:
                # TODO compute correct location
                raise GroundingError("Unexpected clause head '%s'" % head)
            # elif len(heads) > 1 and head.probability is None:
            #     raise GroundingError("Non-probabilistic head in multi-head clause '%s'" % head)
        if len(heads) > 1:
            return AnnotatedDisjunction(heads, operand2, location=(self.loc_id, location))
        else:
            return Clause(operand1[0], operand2, location=(self.loc_id, location))

    # noinspection PyUnusedLocal
    def build_disjunction(self, functor, operand1, operand2, location=None, **extra):
        return Or(operand1, operand2, location=(self.loc_id, location))

    # noinspection PyUnusedLocal
    def build_conjunction(self, functor, operand1, operand2, location=None, **extra):
        return And(operand1, operand2, location=(self.loc_id, location))

    # noinspection PyUnusedLocal
    def build_not(self, functor, operand, location=None, **extra):
        return Not(functor, operand, location=(self.loc_id, location))

    def build_probabilistic(self, operand1, operand2, location=None, **extra):
        operand2.probability = operand1
        return operand2

    def _uncurry(self, term, func=None):
        if func is None:
            func = term.functor

        body = []
        current = term
        while isinstance(current, Term) and current.functor == func:
            body.append(current.args[0])
            current = current.args[1]
        body.append(current)
        return body


class ExtendedPrologFactory(PrologFactory):
    """Prolog with some extra syntactic sugar.

    Non-standard syntax:
    - Negative head literals [Meert and Vennekens, PGM 2014]: 0.5::\+a :- b.
    """

    def __init__(self, identifier=0):
        PrologFactory.__init__(self, identifier)
        self.neg_head_lits = dict()

    def _update_functors(self, t):
        """Adapt functors that appear as a negative literal to be f_p and f_n
        where f appears in the head.
        TODO: Should be implemented using a more general visitor pattern
        """
        if type(t) is Clause:
            self._update_functors(t.head)
        elif type(t) is AnnotatedDisjunction:
            self._update_functors(t.heads)
        elif type(t) is Term:
            if t.signature in self.neg_head_lits:
                t.functor = self.neg_head_lits[t.signature]['p']
        elif type(t) is Not:
            self._update_functors(t.child)
        elif type(t) is Or or type(t) is And:
            self._update_functors(t.op1)
            self._update_functors(t.op2)
        elif type(t) is None or type(t) is Var or type(t) is Constant:
            pass
        elif type(t) is list:
            for term in t:
                self._update_functors(term)
        else:
            raise Exception("Unknown type: {} -- {}".format(t, type(t)))

    def build_program(self, clauses):
        """Update functor f that appear as a negative head literal to f_p and
        :param clauses:
        :return:
        """
        # f_n
        for clause in clauses:
            self._update_functors(clause)

        # Add extra rule for a functor f that appears a as a negative head
        # literal such that:
        # f :- f_p, \+f_n.
        for k, v in self.neg_head_lits.items():
            cur_vars = [Var("V{}".format(i)) for i in range(v['c'])]
            new_clause = Clause(Term(v['f'], *cur_vars),
                                And(Term(v['p'], *cur_vars), Not('\+', Term(v['n'], *cur_vars))))
            clauses.append(new_clause)
        return clauses

    def neg_head_literal_to_pos_literal(self, literal):
        """Translate a negated literal into a positive literal and remember
        the literal to update the complete program later (in build_program).
        :param literal:
        :return:
        """
        literal = abs(literal)
        if literal.signature not in self.neg_head_lits:
            self.neg_head_lits[literal.signature] = {
                'c': literal.arity,
                'p': literal.functor + "_p",
                'n': literal.functor + "_n",
                'f': literal.functor
            }
        literal.functor = self.neg_head_lits[literal.signature]['n']
        return literal

    def build_probabilistic(self, operand1, operand2, location=None, **extra):
        """Detect probabilistic negated head literal and translate to positive literal
        :param operand1:
        :param operand2:
        :param location:
        :param extra:
        :return:
        """
        if ('unaryop' in extra and extra['unaryop'] == '\\+') or operand2.is_negated():
            operand2 = self.neg_head_literal_to_pos_literal(operand2)
        operand2.probability = operand1
        return operand2

    def build_clause(self, functor, operand1, operand2, location=None, **extra):
        """Detect deterministic head literal and translate to positive literal
        :param functor:
        :param operand1:
        :param operand2:
        :param location:
        :param extra:
        :return:
        """
        heads = operand1
        new_heads = []
        for head in heads:
            if type(head) == Not:
                new_head = self.neg_head_literal_to_pos_literal(head)
                new_heads.append(new_head)
            else:
                new_heads.append(head)
        return super(ExtendedPrologFactory, self).build_clause(functor, new_heads, operand2, location, **extra)

DefaultPrologFactory = ExtendedPrologFactory


@transform(str, LogicProgram)
def _string_to_program(source, target=None, **kwargs):
    return PrologString(source)
