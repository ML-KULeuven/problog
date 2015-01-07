from __future__ import print_function

"""
How to use:

    # Create a factory for your system's objects (make your own factory)
    f = Factory()

    # Create a parser
    p = PrologParser(f)
    
    # Parse from a string
    result = p.parseString( string )
    
    # Parse from a file
    result = p.parseFile( filename )
    
This code is compatible with Python2 and Python3    

"""


from pyparsing import *

from collections import defaultdict, Counter

import traceback, sys

def guarded(f) :
    """Helper function for debugging parsing problems.
    
    By default PyParsing hides all exceptions. 
    This decorator function prints a stack trace whenever the decorated 
      function throws an exception.
    """
    
    class Guard(object) :
        
        def __enter__(self) :
            pass
            
        def __exit__(self, exc_type, exc_value, tb) :
            if exc_type != None and exc_type != ParseException :
                print (str(exc_value), exc_type, file=sys.stderr)
                traceback.print_tb(tb)
    
    def res(*args, **kwdargs) :
        with Guard() :
            return f(*args, **kwdargs)
    
    return res


class PrologParser(object) :
    """
    Parser for Prolog based on PyParsing.

    This parser includes supports most of Prolog's syntax.

    Known limitations:
        - floating point number are limited to simple format (e.g. '0.50')
        - no support for postfix operators
        - no support for unquoted version infix notation of binary operators
            e.g. +(1,2) doesn't work => '+'(1,2) and 1+2 do work
            
    This parser uses a custom approach for parsing binary operators because 
     PyParsing's default support if very slow.
    
    """
    
    def __init__(self, factory) :
        self.factory = factory
        self.__operators_by_priority = defaultdict(list)
        self.__unary_operators = {}
        self.__binary_operators = {}
        
        self._init_operators()
        
        self.prepare()
        
    def _init_operators(self) :
        """Operator definitions (based on Prolog's operator specification)"""
        
        # (priority, op_tokens, format, creator, extra)
        
        self.addOperator('-->'  , 1200, 'xfx', self.factory.build_binop)
        self.addOperator(':-'   , 1200, 'xfx', self.factory.build_clause )
        self.addOperator('<-'   , 1200, 'xfx', self.factory.build_clause )    
        self.addOperator(':-'   , 1200, 'fx', self.factory.build_directive )
        self.addOperator('?-'   , 1200, 'fx', self.factory.build_unop)
        self.addOperator( ';'   , 1100, 'xfy', self.factory.build_disjunction )
        self.addOperator( '|'   , 1100, 'xfy', self.factory.build_disjunction )
        self.addOperator( '->'  , 1050, 'xfy', self.factory.build_ifthen )
        self.addOperator('::'   , 1000, 'xfx', self.factory.build_probabilistic)
        self.addOperator( '*->' , 1050, 'xfy', self.factory.build_binop )
        self.addOperator( ','   , 1000, 'xfy', self.factory.build_conjunction )
        self.addOperator( '\+'  ,  900, 'fy', self.factory.build_not )
        self.addOperator( '~'   ,  900, 'fx', self.factory.build_unop )
        
        self.addOperator('<'      , 700 , 'xfx', self.factory.build_compare_arithmetic, function=lambda a, b : a < b)
        self.addOperator('=<'     , 700 , 'xfx', self.factory.build_compare_arithmetic, function=lambda a, b : a <= b)
        self.addOperator('=:='    , 700 , 'xfx', self.factory.build_compare_arithmetic, function=lambda a, b : a == b)
        self.addOperator('>='     , 700 , 'xfx', self.factory.build_compare_arithmetic, function=lambda a, b : a >= b)
        self.addOperator('>'      , 700 , 'xfx', self.factory.build_compare_arithmetic, function=lambda a, b : a > b)
        self.addOperator('=\='    , 700 , 'xfx', self.factory.build_compare_arithmetic, function=lambda a, b : a != b)
        
        self.addOperator('@<'      , 700 , 'xfx', self.factory.build_compare_struct, function=lambda a, b : a < b)
        self.addOperator('@=<'     , 700 , 'xfx', self.factory.build_compare_struct, function=lambda a, b : a <= b)
        self.addOperator('=@='     , 700 , 'xfx', self.factory.build_compare_struct, function=lambda a, b : a == b)
        self.addOperator('@>='     , 700 , 'xfx', self.factory.build_compare_struct, function=lambda a, b : a >= b)
        self.addOperator('@>'      , 700 , 'xfx', self.factory.build_compare_struct, function=lambda a, b : a > b)
        self.addOperator('\=@='    , 700 , 'xfx', self.factory.build_compare_struct, function=lambda a, b : a != b)
        
        compare_operators = ['=', '=..', '\=', '\==', '==', 'is']
        for cmp_op in compare_operators :
            self.addOperator(cmp_op, 700 , 'xfx', self.factory.build_compare_eq)
            
        self.addOperator( ':'   , 600, 'xfy', self.factory.build_mathop2)
        self.addOperator( '+'   , 500, 'yfx', self.factory.build_mathop2, function=lambda a, b : a + b)
        self.addOperator( '-'   , 500, 'yfx', self.factory.build_mathop2, function=lambda a, b : a - b)
        self.addOperator( '/\\' , 500, 'yfx', self.factory.build_mathop2, function=lambda a, b : a & b)
        self.addOperator( '\/'  , 500, 'yfx', self.factory.build_mathop2, function=lambda a, b : a | b)
        self.addOperator( 'xor' , 500, 'yfx', self.factory.build_mathop2, function=lambda a, b : a ^ b)
        
        self.addOperator( '?'   , 500, 'fx', self.factory.build_unop )    
        
        self.addOperator( '*'   , 400 , 'yfx', self.factory.build_mathop2, function=lambda a, b : a * b)
        self.addOperator( '/'   , 400 , 'yfx', self.factory.build_mathop2, function=lambda a, b : a / b)
        self.addOperator( '//'  , 400 , 'yfx', self.factory.build_mathop2, function=lambda a, b : a // b)
        self.addOperator( 'rdiv', 400 , 'yfx', self.factory.build_mathop2) # rational number division
        self.addOperator( '<<'  , 400 , 'yfx', self.factory.build_mathop2, function=lambda a, b : a << b)
        self.addOperator( '>>'  , 400 , 'yfx', self.factory.build_mathop2, function=lambda a, b : a >> b)
        self.addOperator( 'mod' , 400 , 'yfx', self.factory.build_mathop2, function=lambda a, b : a % b)
        self.addOperator( 'rem' , 400 , 'yfx', self.factory.build_mathop2, function=lambda a, b : a % b)
        
        self.addOperator( '**'  , 200 , 'xfx', self.factory.build_mathop2, function=lambda a, b : a ** b)
        self.addOperator( '^'   , 400 , 'xfy', self.factory.build_mathop2, function=lambda a, b : a ** b)
        
        self.addOperator( '+' , 200, 'fy', self.factory.build_mathop1, function=lambda a: a )
        self.addOperator( '-' , 200, 'fy', self.factory.build_mathop1, function=lambda a: -a )
        self.addOperator( '\\', 200, 'fy', self.factory.build_mathop1, function=lambda a: ~a )
        
        self.addOperator( '.'   , 0, 'xfy', self.factory.build_list_op)        # specify so user can enter list in this notation, but don't use it as an operator
    
    def addOperator(self, operator, priority, format, creator, additional_tokens=[], **extra_args):
        self.__operators_by_priority[priority] += [operator] + additional_tokens
        for op in [operator] + additional_tokens :
            if len(format) == 3 :
                self.__binary_operators[ op ] = (operator, priority, format, creator, extra_args)
            elif len(format) == 2 :
                self.__unary_operators[ op ] = (operator, priority, format, creator, extra_args)
            else :
                raise Exception()
    
    def getOperator(self, operator, tok_before, tok_after) :
        if not self.isTerm(tok_before) : # or not self.isTerm(tok_after) :
            return self.getUnaryOperator(operator)
        else :
            return self.getBinaryOperator(operator)
        
    def isTerm(self, token) :
        return token != '' and not token in self.__unary_operators and not token in self.__binary_operators
        
    def getBinaryOperator(self, operator) :
        return self.__binary_operators[operator]

    def getUnaryOperator(self, operator) :
        return self.__unary_operators[operator]
        
    
    def operators_by_priority(self) :
        for p in sorted(self.__operators_by_priority) :
            if p > 0 :
                yield self.__operators_by_priority[p]
                
    binary_operators = property(lambda s : [ op for op in s.__binary_operators if s.__binary_operators[op][1] > 0 ] )
    unary_operators = property(lambda s : [ op for op in s.__unary_operators if s.__unary_operators[op][1] > 0 ] )                
    
    @guarded
    def _parse_clause(self, s, l, toks) :
        head = toks[0]
        body = None
        if len(toks) > 1 :
            body = toks[1]
        return self.factory.build_clause(head, body)
        
    def _combine_formats(self, format_list) :
        # TODO operator clashes for unary operators
        counts = Counter(format_list)
        if counts['xfx'] > 1 :
            raise Exception('Operator precedence clash.')
        elif counts['xfy'] > 0 and counts['yfx'] > 0 :
            raise Exception('Operator precedence clash.')
        elif counts['xfy'] > 0 :
            return 'xfy'
        elif counts['yfx'] > 0 :
            return 'yfx'
        elif counts['xfx'] > 0 :
            return 'xfx'
        elif counts['xf'] > 0 :
            return 'xf'
        elif counts['yf'] > 0 :
            return 'yf'
        elif counts['fx'] > 0 :
            return 'fx'
        elif counts['fy'] > 0 :
            return 'fy'
        else :
            raise Exception('Missing operator.')
                
    @guarded        
    def _parse_cut(self, s, l, toks) :
        return self.factory.build_cut()    
        
    @guarded
    def _parse_constant(self, s, l, toks) :
        return self.factory.build_constant(toks[0])
        
    @guarded
    def _parse_variable(self, s, l, toks) :
        return self.factory.build_variable(toks[0])
        
    @guarded
    def _parse_identifier(self, s, l, toks) :
        return str(toks[0])
        
    @guarded
    def _parse_function(self, s,l,toks) :
        return self.factory.build_function(toks[0], toks[1:])
    
    @guarded
    def _parse_program(self, s, l ,toks) :
        return self.factory.build_program(toks)
        
    @guarded
    def _create_operator2(self, operator, operand1, operand2) :
        operator, prior, format, creator, extra_args = self.getBinaryOperator(operator)
        return creator(priority=prior, format=format, functor=operator, operand1=operand1, operand2=operand2,**extra_args)
        
    @guarded
    def _create_operator1(self, operator, operand) :
        operator, prior, format, creator, extra_args = self.getUnaryOperator(operator)
        return creator(priority=prior, format=format, functor=operator, operand=operand, **extra_args)
    
    @guarded
    def _parse_arg(self, s, loc, toks) :
        """Fold a list of tokens with binary operators into a single node based 
            on operator priority."""
        
        if len(toks) == 1 :
            return toks[0]
        else :
            # find positions of highest priority operator
            max_priority = -1
            operator_locations = []
            operator_formats = []
            for i in range(0, len(toks)) :
                try :
                    op = toks[i]
                    
                    tok_before = toks[i-1] if i > 0 else ''
                    tok_after = toks[i+1] if i < len(toks)-1 else ''
                    pr,fm = self.getOperator(op, tok_before, tok_after)[1:3]
                    if pr > max_priority :
                        operator_locations = [i]
                        max_priority = pr
                        operator_formats = [fm]
                    elif pr == max_priority :
                        operator_locations.append(i)
                        operator_formats.append(fm)
                    else :
                        pass
                except KeyError :
                    pass
            try :
                operator_format = self._combine_formats(operator_formats)
            except Exception as err :
                raise ParseException(str(s),loc=loc, msg=str(err.message))
                
            if len(operator_format) == 3 :
                if operator_format == 'xfy' :
                    # fold on leftmost
                    fold_location = operator_locations[0]
                elif operator_format in ['yfx','xfx'] :
                    # fold on rightmost
                    fold_location = operator_locations[-1]
                return self._create_operator2(toks[fold_location], self._parse_arg(s,loc,toks[:fold_location]), self._parse_arg(s,loc,toks[fold_location+1:]))  
            else :
                if operator_format in ['fx', 'fy'] :
                    op_loc = operator_locations[0]
                    operator = toks[op_loc]
                    operand = toks[op_loc + 1 : ]
                    sub_toks = toks[ : op_loc] + [ self._create_operator1( operator, self._parse_arg(s,loc, operand))  ]
                else :
                    op_loc = operator_locations[-1]
                    operand = toks[ : op_loc  ]
                    sub_toks = [ self._create_operator1( operator, self._parse_arg(s,loc, operand))  ] + toks[ oploc + 1 : ]
                return self._parse_arg(s,loc,sub_toks)
                
    @guarded            
    def _parse_list(self, s, loc, toks) :
        if len(toks) == 0 :
            # Empty list
            values = []
            tail = None
        elif len(toks) > 1 and str(toks[-2]) == '|'  :
            # List with tail
            values = toks[:-2]
            tail = toks[-1]
        else :
            # Plain list
            values = toks
            tail = None
        return self.factory.build_list(values,tail=tail)
        
    @guarded
    def _parse_string(self, s, loc, toks) :
        return self.factory.build_string(toks[0])
        
    def _define_operators(self) :
        self.__binary_operator = oneOf(list(self.binary_operators))
        self.__binary_operator_in_list = oneOf([ op for op in self.__binary_operators if 0 < self.__binary_operators[op][1] < self.__binary_operators[','][1] ])
        self.__unary_operator = oneOf(list(self.unary_operators))

    def _define_basic_types(self) :
        
        # Define basic characters.
        self.__lparen = Literal("(").suppress()
        self.__rparen = Literal(")").suppress()
        self.__lbrack = Literal("[").suppress()
        self.__rbrack = Literal("]").suppress()
        self.__pipe = Literal("|")                  # keep this one in output
        self.__dot = Literal(".").suppress()
        self.__comma = Literal(",").suppress()
        
        
        # Define basic tokens
        
        # Distinction between variable and identifier by starting character
        cc_var_start = '_ABCDEFGHIJKLMNOPQRSTUVWXYZ'    # Start of variable
        cc_id_start = 'abcdefghijklmnopqrstuvwxyz'      # Start of identifier

        # <float> ::= <word of nums> "." <word of nums>
        self.__float_number = Regex(r'\d+\.(\d)+([eE]\d+)?') # Word(nums) + "." + Word(nums)
        #self.__float_number.setParseAction(lambda s, x, t : float(''.join(t)))
        self.__float_number.setParseAction(lambda s, x, t : float(t[0]))
        
        # <int> ::= <word of nums>
        self.__int_number = Word(nums)
        self.__int_number.setParseAction(lambda s, x, t : int(''.join(t)))
        
        # <constant> ::= <float> | <int>
        self.__constant = self.__float_number | self.__int_number
        self.__constant.setParseAction(self._parse_constant)
        
        # <variable> ::= ...
        self.__variable = Word(cc_var_start, alphanums + '_')
        self.__variable.setParseAction(self._parse_variable) 
        
        # <identifier> ::= ...
        self.__identifier = QuotedString("'", unquoteResults=False) | Word(cc_id_start, alphanums + '_')
        self.__identifier.setParseAction(self._parse_identifier)
        
        # <string> ::= ... double quoted
        self.__string = QuotedString('"', unquoteResults=True)
        self.__string.setParseAction(self._parse_string)
        
        # <cut> ::= "!"
        self.__cut = Literal('!')
        self.__cut.setParseAction(self._parse_cut)

    def prepare(self) :
        """Prepare the parser by initializing all tokens."""
        
        # Load basic types.
        self._define_basic_types()
        
        # Load operators.
        self._define_operators()
        
        # Some forward definitions for recursively defined tokens.  
        self.__fulllist = Forward()
        self.__func = Forward()
        self.__arg = Forward()
        self.__arg_in_list = Forward()
        
        # <emptylist> ::= "[" "]"
        self.__emptylist = self.__lbrack + self.__rbrack
        # <list> ::= <emptylist> | <fulllist>
        self.__list = self.__emptylist | self.__fulllist
        
        # Basic expression
        # <base_expr> ::= "(" <arg> ")" | <constant> | <variable> | <list> | <function> | <cut>
        self.__base_expr = ( self.__lparen + self.__arg + self.__rparen) | self.__constant | self.__variable | self.__list | self.__func | self.__cut | self.__string
        
        # Basic expression with 0 or more unary operators
        # <base_expr2> ::= (<unary_operator>)* <base_expr>
        self.__base_expr2 = ZeroOrMore(self.__unary_operator) + self.__base_expr
        
        # Combination of basic expressions with binary operators
        # <arg> ::= <base_expr2> ( <binary_operator> <base_expr2> )*
        self.__arg << self.__base_expr2 + ZeroOrMore( self.__binary_operator + self.__base_expr2 )
        self.__arg.setParseAction(self._parse_arg)
        
        # Combination of basic expressions with binary operators of priority below that of ',' (arguments in list)
        # <arg_in_list> ::= <base_expr2> ( <binary_operator> <base_expr2> )*
        self.__arg_in_list << self.__base_expr2 + ZeroOrMore( self.__binary_operator_in_list + self.__base_expr2 )
        self.__arg_in_list.setParseAction(self._parse_arg)
        
        # List of arguments
        # <arglist> ::= <arg_in_list> ("," <arg_in_list>)*
        self.__arglist = self.__arg_in_list + ZeroOrMore(self.__comma + self.__arg_in_list)

        # Function
        # <func> ::= <identifier> ( "(" <arg_list> ")" )?
        self.__func << self.__identifier + Optional( self.__lparen + self.__arglist + self.__rparen )
        self.__func.setParseAction(self._parse_function)
        
        # Tail of a Prolog list
        # <list_tail> ::= <variable> | <list>
        self.__list_tail = self.__variable | self.__list
        
        # List with values
        # <fulllist> ::= "[" <arglist> ( "|" <list_tail> )? "]"
        self.__fulllist << self.__lbrack + self.__arglist + Optional(self.__pipe + self.__list_tail) + self.__rbrack
        self.__list.setParseAction(self._parse_list)
                        
        # <statement> ::= <arg> ( ":-" )
        
        # <fact> ::= <prob> :: <func> | <func> 
        self.fact = Optional(self.__arg_in_list + Literal("::")) + self.__func
        self.fact.setParseAction(self._parse_fact)
        
        # <facts> ::= <fact> ( ";" <fact> )*
        self.facts = Forward()
        self.facts << self.fact + ZeroOrMore(Literal(";").suppress() + self.fact)
        
        # <directive> ::= ":-" <arg> <dot>
        self.directive = Literal(":-") + self.__arg + self.__dot
        
        # <clause> ::= <facts> ( ":-" <arg> )? <dot>
        self.clause = self.facts + Optional( ( Literal("<-") | Literal(":-") ) + self.__arg ) + self.__dot

        # <statement> ::= self.facts | self.clause        
        self.statement = (self.directive | self.clause)
        self.statement.setParseAction( self._parse_statement )
        
        # <program> is a list of statements
        self.program = OneOrMore(self.statement).ignore('%' + restOfLine )
        self.program.setParseAction(self._parse_program)
        
    @guarded
    def _parse_fact(self, s, loc, toks) :
        if len(toks) > 1 :
            return self.factory.build_probabilistic( functor=toks[1], operand1=toks[0], operand2=toks[2] )
        else :
            return toks[0]
    
    @guarded    
    def _parse_statement(self, s, loc, toks) :
        if len(toks) == 1 : # simple fact
            return toks[0]
        elif toks[0] == (':-') : # directive
            return self.factory.build_directive( toks[0], toks[-1] )            
        else :  
            if toks[-2] in ('<-', ':-') :
                heads = toks[:-2]
                body = toks[-1]
                func = toks[-2]
            else :
                heads = toks
                body = self.factory.build_function('true', [])
                func = ':-'
            return self.factory.build_clause( func, heads, body )            
        
    def parseToken(self, string) :
        return self.__arg.parseString(string, True)[0]
                
    def parseStatement(self, string) :
        return self.statement.parseString(string, True)[0]
        
    def parseString(self, string) :
        return self.program.parseString(string, True)[0]
        
    def parseFile(self, filename) :
        with open(filename) as f :
            return self.parseString(f.read())


def is_white(s) :
    return s in ' \t\n'


class FastPrologParser(PrologParser) :

    def parseStatement(self, string) :
        if not ':-' in string and not '<-' in string :
            # TODO Assumptions: string literals don't contain ; or ::
            disjuncts = list(map(self.parse_fact, string.split(';')))  # assume string literals don't contain ;
        
            if len(disjuncts) > 1 :
                body = self.factory.build_function('true', [])
                func = ':-'
                return self.factory.build_clause( func, disjuncts, body )            
            else :
                return disjuncts[0]
            return string
        else :
            # It's a clause: call regular parser
            return PrologParser.parseStatement(self, string)
        
        
    # def parseString(self, string) :
    #     return self.program.parseString(string, True)[0]
            
    def parseFile(self, filename) :
        # TODO assumption: no '.' in strings    
        with open(filename) as f :
            current_statement = ''
            for line in f :
                # Remove comments
                p = line.find('%')
                if p > -1 : line = line[:p]
                # Remove whitespace before and after
                line = line.strip()
                # Ignore empty lines
                if not line : continue
                # Find end-of-statement character (.)
                p = line.find('.')
                while p > -1 and p < len(line) - 1 and not is_white(line[p+1]) :
                    # Not an end-of-statement character
                    p = line.find('.', p+1)
                if p == -1 : # No end-of-statement character found
                    current_statement += ' ' + line
                else :
                    current_statement += ' ' + line[:p+1]
                    yield self.parseStatement(current_statement.strip())
                    current_statement = line[p+1:]


    def parse_ident(self, string, p) :
        p1 = p
        if string[p] == "'" :
            p = string.find("'", p+1) 
            assert(p>-1)
        else :
            while p < len(string) and ( 'a' <= string[p] <= 'z' or 'A' <= string[p] <= 'Z' or '0' <= string[p] <= '9' or string[p] == '_' ) :
                p += 1
            p -= 1
        return p+1, string[p1:p+1]
    
    def skip_ws(self, string, p) :
        while p<len(string) and is_white(string[p]) : p += 1
        return p

    def parse_atom(self, string, p=0) :
        # Can be a number -> then it starts with a number
        p = self.skip_ws(string,p)
        assert(p < len(string))    
        if '0' <= string[p] <= '9' :
            # Number
            n = ''
            while p < len(string) and '0' <= string[p] <= '9' :
                n += string[p]
                p +=1 
            if p < len(string) and string[p] == '.' :   # float
                n += string[p]
                p += 1
                while p < len(string) and '0' <= string[p] <= '9' :
                    n += string[p]
                    p +=1 
                p = self.skip_ws(string, p)
                return p, self.factory.build_constant(float(n))
            else :
                p = self.skip_ws(string, p)
                return p, self.factory.build_constant(int(n))
        elif 'A' <= string[p] <= 'Z' or string[p] == '_' :
            # Variable
            p, name = self.parse_ident(string, p)
            return p, self.factory.build_variable(name)
        elif string[p] == '[' :
            # List
            p += 1
            p = self.skip_ws(string, p)
            if string[p] == ']' :
                current = self.factory.build_list( [], tail=None )
            else :
                args = []
                p, arg = self.parse_atom(string, p)
                args.append(arg)
                p = self.skip_ws(string, p)
        
                while p < len(string) and string[p] == ',' :
                    p += 1
                    p = self.skip_ws(string, p)
                    p, arg = self.parse_atom(string, p)
                    args.append(arg)
                    p = self.skip_ws(string, p)

                tail = None
                if p < len(string) and string[p] == '|' :
                    p += 1
                    p = self.skip_ws(string, p)
                    p, tail = self.parse_atom(string, p)
                    p = self.skip_ws(string, p)

                assert(string[p] == ']')            
                current = self.factory.build_list( args, tail=tail )
        
            p += 1
            p = self.skip_ws(string, p)
        
            return p, current            
        
        else :
            # Term
            p, ident = self.parse_ident(string, p)
            p = self.skip_ws(string, p)

            if p < len(string) and string[p] == '(' :
                # Parse argument list
                p += 1
                p = self.skip_ws(string, p)
            
                args = []
                p, arg = self.parse_atom(string, p)
                args.append(arg)
                p = self.skip_ws(string, p)
            
                while p < len(string) and string[p] == ',' :
                    p += 1
                    p = self.skip_ws(string, p)
                    p, arg = self.parse_atom(string, p)
                    args.append(arg)
                    p = self.skip_ws(string, p)
            
                assert(string[p] == ')')
                p += 1
                p = self.skip_ws(string, p)
            
            else :
                args = ()
        
            return p, self.factory.build_function(ident, args)


    def parse_fact(self, string) :
        string = string.rstrip('.')
        prob_atom = string.split('::')  # assume string literals don't contain ::
        assert(len(prob_atom) <= 2)
        if len(prob_atom) == 2 :
            probstr, atomstr = prob_atom
        else :
            probstr, atomstr = None, prob_atom[0]

        pos, atom = self.parse_atom(atomstr)  
        assert(pos == len(atomstr))  
    
        if probstr :
            pos, prob = self.parse_atom(probstr)
            assert(pos == len(probstr))
            self.factory.build_probabilistic(prob, atom)
    
        return atom

DefaultPrologParser = FastPrologParser

class Factory(object) :
    """Factory object for creating suitable objects from the parse tree."""
        
    def build_program(self, clauses) :
        return '\n'.join(map(str,clauses))
    
    def build_function(self, functor, arguments) :
        return '%s(%s)' % (functor, ', '.join(map(str,arguments)))
        
    def build_variable(self, name) :
        return str(name)
        
    def build_constant(self, value) :
        return str(value)
        
    def build_binop(self, functor, operand1, operand2, function=None, **extra) :
        return self.build_function("'" + functor + "'", (operand1, operand2))
        
    def build_unop(self, functor, operand, **extra) :
        return self.build_function("'" + functor + "'", (operand,) )
        
    def build_list(self, values, tail=None, **extra) :
        if tail == None :
            return '[%s]' % (', '.join(map(str,values)))
        else :
            return '[%s | %s]' % (', '.join(map(str,values)), tail)
        
    def build_string(self, value) :
        return self.build_constant('"' + value + '"');
    
    def build_cut(self) :
        raise NotImplementedError('Not supported!')
        
    build_clause = build_binop
    build_probabilistic = build_binop
    build_disjunction = build_binop
    build_conjunction = build_binop
    build_compare_arithmetic = build_binop
    build_compare_struct = build_binop
    build_compare_eq = build_binop
    build_mathop2 = build_binop
    build_ifthen = build_binop
    build_list_op = build_binop
    
    build_not = build_unop
    build_mathop1 = build_unop
    build_directive = build_unop        
                
if __name__ == '__main__' :
    from parser import PrologParser
        
    import sys
    if sys.argv[1] == '--string' :
        result = PrologParser(Factory()).parseString(sys.argv[2])
    else :
        result = PrologParser(Factory()).parseFile(sys.argv[1])
    print(result)
