from __future__ import print_function

from .core import ParseError as CoreParseError

LINE_COMMENT = '%'
BLOCK_COMMENT_START = '/*'
BLOCK_COMMENT_END = '*/'
NEWLINE = '\n'
    
WHITESPACE = frozenset('\n\t ')

class ParseError(CoreParseError) :
    
    def __init__(self, string, message, location) :
        self.msg = message
        self.lineno, self.col, self.line = self._convert_pos(string, location)
        Exception.__init__(self, '%s (at %s:%s)' % (self.msg,self.lineno, self.col))
        
    def _convert_pos(self, string, location) :
        lineno = 1
        col = 0
        end = 0
        stop = False
        for i,x in enumerate(string) :
            if x == '\n' :
                lineno +=1 
                col = 0
                if stop :
                    break
            if i == location : 
                stop = True
            if not stop :
                col += 1
        return lineno, col, string[location-col:i+1]

class UnexpectedCharacter(ParseError) :
    
    def __init__(self, string, position) :
        char = string[position]
        ParseError.__init__(self, string, "Unexpected character '%s'" % char, position)

class UnmatchedCharacter(ParseError) :
    
    def __init__(self, string, position, length=1) :
        char = string[position:position+length]
        ParseError.__init__(self, string, "Unmatched character '%s'" % char, position)

class Token(object) :
    
    def __init__(self, string, pos, types=None, end=None, atom=True, functor=False, binop=None, unop=None, special=None, atom_action=None) :
#        if end == None : end = pos+len(string)
        self.string = string
        self.location = pos
        self.atom = atom
        self.binop = binop
        self.unop = unop
        self.special = special
        if atom :
            self.functor = functor
        else :
            self.functor = False
        self.is_comma_list = False
        self.atom_action = atom_action
        
    def is_special(self, special) :
        return self.special == special
        
    def is_atom(self) :
        return self.atom
        
    def count_options(self) :
        o = 0
        if self.atom : o += 1
        if self.binop : o += 1
        if self.unop : o += 1
        if self.functor : o += 1
        return o
        
    def list_options(self) :  # pragma: no cover
        o = ''
        if self.atom : o += 'a'
        if self.binop : o += 'b'
        if self.unop : o += 'u'
        if self.functor : o += 'f'
        return o
        
    def __repr__(self) : # pragma: no cover
        return "'%s' {%s}" % (self.string, self.list_options())
        
        
from collections import namedtuple


SPECIAL_PAREN_OPEN = 0
SPECIAL_PAREN_CLOSE = 1
SPECIAL_END = 2
SPECIAL_COMMA = 3
SPECIAL_BRACK_OPEN = 4
SPECIAL_BRACK_CLOSE = 5
SPECIAL_VARIABLE = 6
SPECIAL_FLOAT = 7
SPECIAL_INTEGER = 8
SPECIAL_PIPE = 9
SPECIAL_STRING = 10
SPECIAL_ARGLIST = 11

import re
RE_FLOAT = re.compile(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?')

def skip_to(s, pos, char) :
    end = s.find(char, pos)
    if end == -1 :
        return len(s)
    else :
        return end+1

def skip_comment_c(s, pos) :
    end = s.find(BLOCK_COMMENT_END, pos)
    if end == -1 :
        raise UnmatchedCharacter(s, pos, 2)
    return end + 2

def skip_comment_line(s, pos) :
    return skip_to(s,pos,NEWLINE)

def is_lower(c) :
    return 'a' <= c <= 'z'

def is_upper(c) :
    return 'A' <= c <= 'Z'

def is_digit(c) :
    return '0' <= c <= '9'

def is_whitespace(c) :
    return c <= ' '

class PrologParser(object) :
    
    def __init__(self, factory) :
        self.factory = factory
        self.prepare()
        
    def _skip(self, s,pos) : 
        return None, pos+1
    
    def _next_paren_open(self, s,pos) :
        try :
            return s[pos+1] == '('
        except IndexError :
            return False
            
    def _token_notsupported(self, s, pos) :
        raise UnexpectedCharacter(s, pos)
        
    def _token_dquot(self, s, pos) : 
        end = s.find('"', pos+1)
        while end != -1 and s[end-1] == '\\' :
            end = s.find('"', end+1)
        if end == -1 :
            raise UnmatchedCharacter(s, pos)
        else :
            return Token(s[pos:end+1],pos, special=SPECIAL_STRING), end+1
        
    def _token_pound(self, s, pos) :
        return Token(s[pos], pos, binop=(500,'yfx',self.factory.build_binop), functor=self._next_paren_open(s,pos) ), pos+1
    
    def _token_percent(self, s, pos) :
        return None, skip_comment_line(s,pos)
        
    def _token_squot(self, s, pos) :
        end = s.find("'", pos+1)
        while end != -1 and s[end-1] == '\\' :
            end = s.find("'", end+1)
        if end == -1 :
            raise UnmatchedCharacter(s, pos)
        else :
            return Token(s[pos:end+1],pos), end + 1
        
    def _token_paren_open(self, s, pos) : 
        return Token(s[pos], pos, atom=False, special=SPECIAL_PAREN_OPEN), pos+1
    
    def _token_paren_close(self, s, pos) : 
        return Token(s[pos], pos, atom=False, special=SPECIAL_PAREN_CLOSE), pos+1
    
    def _token_asterisk(self, s, pos) :
        if s[pos:pos+2] == '**' :
            return Token('**', pos, binop=(200,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+3] == '*->' :
            return Token('*->', pos, binop=(200,'xfy',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+3
        else :
            return Token('*', pos, binop=(400,'yfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+1

    def _token_plus(self, s, pos) :
        return Token('+', pos, binop=(500, 'yfx',self.factory.build_binop), unop=(200,'fy',self.factory.build_unop), functor=self._next_paren_open(s,pos)), pos+1

    def _token_comma(self, s, pos) : 
        return Token(',', pos, binop=(1000, 'xfy',self.factory.build_conjunction), atom=False, special=SPECIAL_COMMA), pos+1
    
    def _token_min(self, s, pos) : 
        if s[pos:pos+3] == '-->' :
            return Token('-->', pos, binop=(1200,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+3
        elif s[pos:pos+2] == '->' :
            return Token('->', pos, binop=(1050,'xfy',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        else :
            return Token('-', pos, binop=(500, 'yfx',self.factory.build_binop), unop=(200,'fy',self.factory.build_unop), functor=self._next_paren_open(s,pos)), pos+1
        
    def _token_dot(self, s, pos) : 
        if pos+1 == len(s) or is_whitespace(s[pos+1]) :
            return Token('.', pos, special=SPECIAL_END), pos+1
        elif is_digit(s[pos+1]) :
            return self._token_number(s,pos)
        elif s[pos+1] == '(' :
            return Token('.', pos, functor=self._next_paren_open(s,pos)), pos+1
        else :
            raise UnexpectedCharacter(s, pos)
        
    def _token_slash(self, s, pos) :
        if s[pos:pos+2] == '/\\' :
            return Token('/\\', pos, binop=(500,'yfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+2] == '//' :
            return Token('//', pos, binop=(400,'yfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+2] == '/*' :
            return None, skip_comment_c(s, pos)
        else :
            return Token('/', pos, binop=(400,'yfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+1
            
    def _token_colon(self, s, pos) : 
        if s[pos:pos+2] == ':-' :
            return Token(':-', pos, binop=(1200, 'xfx',self._build_clause), unop=(1200,'fx',self.factory.build_directive), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+2] == '::' :
            return Token('::', pos, binop=(1000, 'xfx',self.factory.build_probabilistic), functor=self._next_paren_open(s,pos)), pos+2
        else :
            return Token(':', pos, binop=(600, 'xfy',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+1
        
    def _token_semicolon(self, s, pos) : 
        return Token(';', pos, binop=(1100, 'xfy',self.factory.build_disjunction), functor=self._next_paren_open(s,pos)), pos+1

    def _token_less(self, s, pos) : 
        if s[pos:pos+2] == '<-' :
            return Token('<-', pos, binop=(1200,'xfx',self._build_clause), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+2] == '<<' :
            return Token('<<', pos, binop=(400,'yfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        else :
            return Token('<', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+1
            
    def _token_equal(self, s, pos) : 
        if s[pos:pos+2] == '=<' :
            return Token('=<', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+3] == '=:=' :
            return Token('=:=', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+3
        elif s[pos:pos+3] == '=\=' :
            return Token('=\=', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+3
        elif s[pos:pos+3] == '=@=' :
            return Token('=@=', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+3
        elif s[pos:pos+3] == '=..' :
            return Token('=..', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+3
        elif s[pos:pos+2] == '==' :
            return Token('==', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        else :
            return Token('=', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+1
        
    def _token_greater(self, s, pos) :
        if s[pos:pos+2] == '>>' :
            return Token('>>', pos, binop=(400,'yfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+2] == '><' :
            return Token('><', pos, binop=(500,'yfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+2] == '>=' :
            return Token('>=', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        else :
            return Token('>', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+1
    
    def _token_question(self, s, pos) : 
        raise UnexpectedCharacter(s, pos)
        
    def _token_at(self, s, pos) : 
        if s[pos:pos+2] == '@<' :
            return Token('@<', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+3] == '@=<' :
            return Token('@=<', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+3
        elif s[pos:pos+3] == '@>=' :
            return Token('@>=', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+3
        elif s[pos:pos+2] == '@>' :
            return Token('@>', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        else :
            raise UnexpectedCharacter(s, pos)
    
    def _token_bracket_open(self, s, pos) : 
        return Token('[', pos, atom=False, special=SPECIAL_BRACK_OPEN), pos+1
    
    def _token_backslash(self, s, pos) : 
        if s[pos:pos+2] == '\\\\' :
            return Token('\\\\', pos, unop=(200,'fy',self.factory.build_unop), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+2] == '\\+' :
            return Token('\\+', pos, unop=(900,'fy',self.factory.build_not), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+4] == '\\=@=' :
            return Token('\\+', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+4
        elif s[pos:pos+3] == '\\==' :
            return Token('\\==', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+3
        elif s[pos:pos+2] == '\\=' :
            return Token('\\=', pos, binop=(700,'xfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        elif s[pos:pos+2] == '\\/' :
            return Token('\\/', pos, binop=(500,'yfx',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+2
        else :
            return Token('\\', pos, unop=(200,'fy',self.factory.build_unop), functor=self._next_paren_open(s,pos)), pos+1
            
    def _token_bracket_close(self, s, pos) : 
        return Token(']', pos, atom=False, special=SPECIAL_BRACK_CLOSE), pos+1
        
    def _token_caret(self, s, pos) : 
        return Token('^', pos, binop=(400,'xfy',self.factory.build_binop), functor=self._next_paren_open(s,pos)), pos+1
    
    def _token_underscore(self, s, pos) : 
        return self._token_upper(s, pos) # Variable
    
    def _token_pipe(self, s, pos) : 
        return Token('|', pos, atom=False, binop=(1100,'xfy',self.factory.build_binop), special=SPECIAL_PIPE), pos+1
        
    def _token_tilde(self, s, pos) : 
        return Token('~', pos, unop=(900,'fx',self.factory.build_unop)), pos+1
    
    def _token_lower(self, s, pos) : 
        end = pos + 1
        s_len = len(s)
        if end < s_len :
            c = s[end]
            while c == '_' or is_lower(c) or is_upper(c) or is_digit(c) :
                end += 1
                if end >= s_len : break
                c = s[end]
        token = s[pos:end]
        kwd = self.string_operators.get(token, {})
        return Token(token, pos, functor=self._next_paren_open(s,end-1), **kwd), end
        
    def _token_upper(self, s, pos) : 
        end = pos + 1
        s_len = len(s)
        if end < s_len :
            c = s[end]
            while c == '_' or is_lower(c) or is_upper(c) or is_digit(c) :
                end += 1
                if end >= s_len : break
                c = s[end]
            return Token(s[pos:end], pos, special=SPECIAL_VARIABLE), end
        else :
            return Token(s[pos], pos, special=SPECIAL_VARIABLE), end
        
    def _token_number(self, s, pos) : 
        token = RE_FLOAT.match(s,pos).group(0)
        if token.find('.') >= 0 or token.find('e') >= 0 or token.find('E') >= 0 :
            return Token(token, pos, special=SPECIAL_FLOAT), pos+len(token)
        else :
            return Token(token, pos, special=SPECIAL_INTEGER), pos+len(token)

    def _token_action(self, char) :
        c = ord(char)
        if c < 33 :
            return self._skip # whitespace
        elif c < 48 :
            return self._token_act1[c-33]
        elif c < 58 :
            return self._token_number
        elif c < 65 :
            return self._token_act2[c-58]
        elif c < 91 :
            return self._token_upper
        elif c < 97 :
            return self._token_act3[c-91]
        elif c < 123 :
            return self._token_lower
        elif c < 127 :
            return self._token_act4[c-123]
        else :
            return None
    
    def _build_clause( self, functor, operand1, operand2, location ) :
        heads = []
        current = operand1
        while current.functor == ';' :
            heads.append(current.args[0])
            current = current.args[1]
        heads.append(current)
        return self.factory.build_clause(functor=functor, operand1=heads, operand2=operand2, location=location)
        
    def next_token(self, s, pos) :
        action = self._token_action(s[pos])
        if action is None:
            raise UnexpectedCharacter(s, pos)
        result = action(s,pos)
        if result is None : # pragma: no cover
            raise RuntimeError("Undefined action: '%s'" % action)
        else :
            return result
            
    def prepare(self) :
        self._token_act1 = [
            self._token_notsupported, # 33 !
            self._token_dquot, # 34 "
            self._token_pound, # 35 #
            self._token_notsupported, # 36 $
            self._token_percent, # 37 %
            self._token_notsupported, # 38 &
            self._token_squot, # 39 '
            self._token_paren_open, # 40 (
            self._token_paren_close, # 41 )
            self._token_asterisk, # 42 *
            self._token_plus, # 43 + 
            self._token_comma, # 44 ,
            self._token_min, # 45 -
            self._token_dot, # 46 .
            self._token_slash # 47 /
        ]
        self._token_act2 = [
            self._token_colon, # 58 :
            self._token_semicolon, # 59 ;
            self._token_less, # 60 <
            self._token_equal, # 61 =
            self._token_greater, # 62 >
            self._token_question, # 63 ?
            self._token_at, # 64 @
        ]
        self._token_act3 = [
            self._token_bracket_open, # 91 [
            self._token_backslash, # 92 \
            self._token_bracket_close,    # 93 ]
            self._token_caret, # 94 ^
            self._token_underscore, # 95 _
            self._token_notsupported, # 96 `
        ]
        self._token_act4 = [
            self._token_notsupported, # 123 {
            self._token_pipe, # 124 |
            self._token_notsupported, # 125 }
            self._token_tilde, # 126 ~
        ]
        
        self.string_operators = {
            'is' : { 'binop': (700,'xfx',self.factory.build_binop) },
            'not' : { 'unop' : (900, 'fy', self.factory.build_not), 'atom' : False },
            'xor' : { 'binop': (500,'yfx',self.factory.build_binop) },
            'rdiv' : { 'binop': (400,'yfx',self.factory.build_binop) },
            'mod' : { 'binop': (400,'yfx',self.factory.build_binop) },
            'rem' : { 'binop': (400,'yfx',self.factory.build_binop) },
            'div' : { 'binop': (400,'yfx',self.factory.build_binop) }
        }
        
    def _tokenize(self, s ) :
        s_len = len(s)
        p = 0
        while p < s_len :
            t, p = self.next_token(s, p)
            if t is not None :
                yield t
    
    def _extract_statements(self, string, s) :
        statement = []
        for token in s :
            if token.is_special(SPECIAL_END) :
                yield statement
                statement = []
            else :
                statement.append(token)
        if statement :
            raise ParseError(string, 'Incomplete statement.', len(string))
    
    
    def _parenthesis_bounds(self, string, tokens ) :
        # Find parenthesis subexpressions
        par_stack = []
        comma_stack = []
        for i, token in enumerate( tokens ) :
            if token.is_special(SPECIAL_PAREN_OPEN) :
                par_stack.append(('(',i))
                comma_stack.append([])
            elif token.is_special(SPECIAL_PAREN_CLOSE) :
                try :
                    t, s = par_stack.pop(-1)
                    if t != '(' :
                        raise UnmatchedCharacter(string, tokens[s].location)
                    yield s, i, comma_stack.pop(-1), '('
                except IndexError :
                    raise UnmatchedCharacter(string, token.location)
            elif token.is_special(SPECIAL_BRACK_OPEN) :
                par_stack.append(('[',i))
                comma_stack.append([])
            elif token.is_special(SPECIAL_BRACK_CLOSE) :
                try :
                    t, s = par_stack.pop(-1)
                    if t != '[' :
                        raise UnmatchedCharacter(string, tokens[s].location)
                    commas = comma_stack.pop(-1)
                    negs = [ x for x in commas if x < 0 ]
                    if negs :
                       assert(len(negs)==1) 
                    else :
                        commas.append(i)
                    yield s, i, map(abs,commas), '['
                except IndexError :
                    raise UnmatchedCharacter(string, token.location)
            elif token.is_special(SPECIAL_COMMA) and comma_stack :
                if comma_stack[-1] and comma_stack[-1][-1] < 0 :
                    raise UnexpectedCharacter(string, token.location)
                comma_stack[-1].append(i)
            elif token.is_special(SPECIAL_PIPE) and comma_stack :
                if comma_stack[-1] and comma_stack[-1][-1] < 0 :
                    raise UnexpectedCharacter(string, token.location)
                comma_stack[-1].append(-i)
            
        if par_stack :
            raise UnmatchedCharacter(string, tokens[par_stack[-1][1]].location)
    
    def _build_operator_free(self, tokens) :
        if len(tokens) == 1 :
            token = tokens[0]
            if isinstance(token, SubExpr) :
                if token.operator in ',.' :
                    if token.operator == ',' :
                        op = "','"
                    else :
                        op = '.' 
                    curr = None
                    for tok in reversed(token.parts) :
                        if curr == None :
                            curr = tok
                        else :
                            curr = self.factory.build_function(functor=op,arguments=(tok,curr)) #,location=tok.location)
                    return curr
                    
            elif token.is_special(SPECIAL_VARIABLE) :
                return self.factory.build_variable(token.string, location=token.location)
            elif token.is_special(SPECIAL_INTEGER) :
                return self.factory.build_constant(int(token.string), location=token.location)
            elif token.is_special(SPECIAL_FLOAT) :
                return self.factory.build_constant(float(token.string), location=token.location)
            elif token.is_special(SPECIAL_STRING) :
                return self.factory.build_string(token.string[1:-1], location=token.location)
            else :
                return self.factory.build_function(token.string, (), location=token.location)
        elif len(tokens) == 2 :
            args = [ tok for tok in tokens[1].parts ]
            return self.factory.build_function(tokens[0].string, args , location=tokens[0].location)
        else :
            assert(len(tokens)==0)
            return None
            
    def fold(self, string, operators, lo, hi, pprior=None, porder=None, level=0 ) :
        if lo >= hi : 
            return self._build_operator_free(operators[lo:hi])
        else :
            max_op = None
            max_i = None
            for i in range(lo, hi) :
                op_n = operators[i]
                op = None
                if op_n.binop :
                    op = op_n.binop
                elif op_n.unop :
                    op = op_n.unop
                if not op is None and (max_op == None or op[0] > max_op[0] or (op[0] == max_op[0] and max_op[1] == 'yfx')) :
                    max_i = i
                    max_op = op
            if max_op == None :
                return self._build_operator_free(operators[lo:hi])
            else :
                if pprior == max_op[0] and porder == 'x' :
                    raise ParseError(string, 'Operator priority clash', operators[max_i].location)
                else :
                    max_order = max_op[1]
                    if len(max_order) == 3 : # binop
                        lf = self.fold( string, operators, lo, max_i, max_op[0], max_order[0], level+1)
                        rf = self.fold( string, operators, max_i+1, hi, max_op[0], max_order[2], level+1)
                        return max_op[2]( functor=operators[max_i].string, operand1=lf, operand2=rf, location=operators[max_i].location )
                    else :  # unop
                        assert(max_i == lo)
                        lf = self.fold( string, operators, lo+1, hi, max_op[0], max_order[1], level+1 )
                        return max_op[2]( functor=operators[max_i].string, operand=lf, location=operators[max_i].location )

    def collapse(self, string, tokens ) :
        bounds = self._parenthesis_bounds(string, tokens )
        for i, j, c, t in bounds :
            i1 = i
            sub_tokens = []
            for ic in c :
                toks = self.label_tokens(string, filterl(None, tokens[i1+1:ic]))
                
                toks = self.fold(string, toks, 0, len(toks) )
                
                sub_tokens.append( toks )
                i1 = ic
            toks = self.label_tokens(string,filterl(None, tokens[i1+1:j]))
            toks = self.fold(string, toks, 0, len(toks) )
            if toks == None : toks = self.factory.build_list(())
            sub_tokens.append( toks )
            if t == '(' :
                sub_token = SubExpr(filterl(None,sub_tokens), operator=',')
            else :
                sub_token = SubExpr(filterl(None,sub_tokens), operator='.')
            tokens[i:j+1] = [ sub_token ] + [None] * (j-i) 
        toks = self.label_tokens(string,filterl(None,tokens))
        return self.fold(string, toks, 0, len(toks) )
    
    def label_tokens(self, string, tokens ) :
        l = len(tokens)-1
        p = None
        for i, t in enumerate(tokens) :
            if i == l : # Last token can not be an operator or functor
                t.unop = None
                t.binop = None
                t.functor = False
            elif t.functor and tokens[i+1].is_comma_list :
                t.atom = False
            
            if i == 0 : 
                t.binop = None  # First token can not be a binop
            elif p.functor :    
                pass
            elif p.atom :
                if not t.binop : raise ParseError(string,'Expected binary operator', t.location)
                t.unop = None
                t.atom = False
                t.functor = False
            elif p.binop :
                t.binop = None
                t.is_comma_list = False
            else :
                pass    
            
            if t.unop and t.functor :
                t.unop = None
        
            if t.unop and t.atom :
                n = tokens[i+1]
                if not n.binop :
                    t.atom = False
            
            p = t
            if t.count_options() != 1 :
                raise ParseError(string,'Ambiguous token role', t.location)
            
        return tokens
        
    def _parse_statement(self, string, tokens ) :
        return self.collapse(string, tokens)
        
    def parseString(self, string) :
        return self.factory.build_program(mapl(lambda x : self._parse_statement(string,x), self._extract_statements(string, self._tokenize(string))))
        
    def parseFile(self, filename) :
        with open(filename) as f :
            return self.parseString(f.read())

def mapl(f, l) :
    return list(map(f,l))

def filterl(f, l) :
    return list(filter(f,l))


class SubExpr(object) :
    
    def __init__(self, parts, operator='') :
        self.parts = parts
        self.operator = operator
        self.is_comma_list = (operator==',')
        self.atom = True
        self.binop = None
        self.unop = None
        self.functor = False
        self.location = (0,0)
        
        self.ast = None
        
    def count_options(self) :
        return 1
        
    def is_special(self, special) :
        return False
        
    def __repr__(self) : # pragma: no cover
        return '%s {%s}' % (self.parts, self.operator)

if __name__ == '__main__' :
    
    import sys
    for filename in sys.argv[1:] :
        print (filename)
        print ('------------------------------------')
        from problog.program import ExtendedPrologFactory
        try :
            parsed = PrologParser(ExtendedPrologFactory()).parseFile(filename)
            for s in parsed :
                print (s)
        except ParseError as e :
            print ('ParseError:', e)
        print ('====================================')
    
