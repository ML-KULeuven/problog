from __future__ import print_function

from .logic import *

from .parser import DefaultPrologParser, Factory
from .core import ProbLogError

from collections import namedtuple, defaultdict
import os, logging, sys

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
    
    def __init__(self, string, parser=None, source_root='.', source_files=None) :
        self.__string = string
        lines = self._find_lines(string)
        LogicProgram.__init__(self, source_root=source_root, source_files=source_files, line_info=lines)
        if parser is None :
            self.parser = DefaultPrologParser(PrologFactory())
        else :
            self.parser = parser
    
    def _find_lines(self, s) :
        """Find line-end positions."""
        lines = [-1]
        f = s.find('\n')
        while f >= 0 :
            lines.append(f)
            f = s.find('\n', f+1)
        lines.append(len(s))
        return lines
        
    def __iter__(self) :
        """Iterator for the clauses in the program."""
        program = self.parser.parseString(self.__string)
        return iter(program)


class PrologFile(PrologString) :
    """LogicProgram implementation as a pointer to a Prolog file.
    
    :param filename: filename of the Prolog file (optional)
    :type filename: string
    """
    
    def __init__(self, filename, parser=None) :
        if filename == '-':
            source_root = ''
            source_files = ['-']
            source_text = sys.stdin.read()
        else:
            source_root = os.path.dirname(filename)
            source_files = [ os.path.abspath(filename)]
            try :
                with open(filename) as f :
                    source_text = f.read()
            except IOError as err :
                raise ProbLogError(str(err))
        PrologString.__init__(self, source_text, parser=parser, source_root=source_root, source_files=source_files)                
        
        
class PrologFactory(Factory) :
    """Factory object for creating suitable objects from the parse tree."""
        
    def build_program(self, clauses) :
        return clauses
    
    def build_function(self, functor, arguments, location=None) :
        return Term( functor, *arguments, location=location )
        
    def build_variable(self, name, location=None) :
        return Var(name)
        
    def build_constant(self, value, location=None) :
        return Constant(value)
        
    def build_binop(self, functor, operand1, operand2, function=None, location=None, **extra) :
        return self.build_function("'" + functor + "'", (operand1, operand2), location=location)

    def build_directive(self, functor, operand, location=None, **extra) :
        head = self.build_function( '_directive', [] )
        return self.build_clause( functor, [head], operand, **extra)
            
    def build_unop(self, functor, operand, location=None, **extra) :
        return self.build_function("'" + functor + "'", (operand,) , location=location)
        
    def build_list(self, values, tail=None, location=None, **extra) :
        if tail is None :
            current = Term('[]')
        else :
            current = tail
        for value in reversed(values) :
            current = self.build_function('.', (value, current), location=location )
        return current
        
    def build_string(self, value, location=None) :
        return self.build_constant('"' + value + '"', location=location);
    
    def build_cut(self, location=None) :
        raise NotImplementedError('Not supported!')
        
    def build_clause(self, functor, operand1, operand2, location=None, **extra) :
        heads = operand1
        #heads = self._uncurry( operand1, ';' )
        if len(heads) > 1 :
            return AnnotatedDisjunction(heads, operand2, location=location)
        else :
            return Clause(operand1[0], operand2, location=location)
        
    def build_disjunction(self, functor, operand1, operand2, location=None, **extra) :
        return Or(operand1, operand2, location=location)
    
    def build_conjunction(self, functor, operand1, operand2, location=None, **extra) :
        return And(operand1, operand2, location=location)
    
    def build_not(self, functor, operand, location=None, **extra) :
        return Not(functor, operand, location=location)
        
    def build_probabilistic(self, operand1, operand2, location=None, **extra) :
        operand2.probability = operand1
        return operand2
        
    def _uncurry(self, term, func=None) :
        if func is None : func = term.functor
        
        body = []
        current = term
        while isinstance(current, Term) and current.functor == func :
            body.append(current.args[0])
            current = current.args[1]
        body.append(current)
        return body    


class ExtendedPrologFactory(PrologFactory):
    """Prolog with some extra syntactic sugar.

    Non-standard syntax:
    - Negative head literals [Meert and Vennekens, PGM 2014]:
      0.5::\+a :- b.
    """
    def __init__(self):
        self.neg_head_lits = dict()

    def update_functors(self, t):
        if type(t) is Clause:
            self.update_functors(t.head)
            self.update_functors(t.body)
        elif type(t) is AnnotatedDisjunction:
            self.update_functors(t.heads)
            self.update_functors(t.body)
        elif type(t) is Term:
            if t.signature in self.neg_head_lits:
                t.functor = self.neg_head_lits[t.signature]['p']
        elif type(t) is Not:
            self.update_functors(t.child)
        elif type(t) is Or or type(t) is And:
            self.update_functors(t.op1)
            self.update_functors(t.op2)
        elif type(t) is None or type(t) is Var or type(t) is Constant:
            pass
        elif type(t) is list :
            for term in t :
                self.update_functors(term)
        else :
            raise Exception("Unknown type: {} -- {}".format(t, type(t)))


    def build_program(self, clauses):
        # Update functor f that appear as a negative head literal to f_p and
        # f_n
        for clause in clauses:
            self.update_functors(clause)

        # Add extra rule for a functor f that appears a as a negative head
        # literal such that:
        # f :- f_p, \+f_n.
        for k,v in self.neg_head_lits.items():
            cur_vars = [Var("v{}".format(i)) for i in range(v['c'])]
            new_clause = Clause(Term(v['f'], *cur_vars), And(Term(v['p'], *cur_vars), Not('\+',Term(v['n'], *cur_vars))))
            clauses.append(new_clause)

        #logger = logging.getLogger('problog')
        #logger.debug('Transformed program:\n{}'.format('\n'.join([str(c) for c in clauses])))

        return clauses


    def build_probabilistic(self, operand1, operand2, location=None, **extra) :
        if ( 'unaryop' in extra and extra['unaryop'] == '\\+' ) or operand2.is_negative() :
            operand2 = abs(operand2)
            if not operand2.signature in self.neg_head_lits:
                self.neg_head_lits[operand2.signature] = {
                    'c': operand2.arity,
                    'p': operand2.functor+"_p",
                    'n': operand2.functor+"_n",
                    'f': operand2.functor
                }
            operand2.functor = self.neg_head_lits[operand2.signature]['n']
        operand2.probability = operand1
        return operand2

