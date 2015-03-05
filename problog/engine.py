from __future__ import print_function

import logging
from collections import defaultdict, namedtuple

from .program import PrologFile
from .logic import *
from .formula import LogicFormula

from .core import transform, GroundingError
from .util import Timer

@transform(LogicProgram, LogicFormula)
def ground(model, target=None, queries=None, evidence=None) :
    return DefaultEngine().ground_all(model,target, queries=queries, evidence=evidence)
    
class GenericEngine(object) :
    """Generic interface to a grounding engine."""
    
    def prepare(self, db) :
        """Prepare the given database for querying.
        Calling this method is optional.
        
        :param db: logic program
        :returns: logic program in optimized format where builtins are initialized and directives have been evaluated 
        """
        raise NotImplementedError('GenericEngine.prepare is an abstract method.')
        
    def query(self, db, term) :
        """Evaluate a query without generating a ground program.
        
        :param db: logic program
        :param term: term to query; variables should be represented as None
        :returns: list of tuples of argument for which the query succeeds.
        """
        raise NotImplementedError('GenericEngine.query is an abstract method.')
        
    def ground(self, db, term, target=None, label=None) :
        """Ground a given query term and store the result in the given ground program.
        
        :param db: logic program
        :param term: term to ground; variables should be represented as None
        :param target: target logic formula to store grounding in (a new one is created if none is given)
        :param label: optional label (query, evidence, ...)
        :returns: logic formula (target if given)
        """
        raise NotImplementedError('GenericEngine.ground is an abstract method.')
        
    def ground_all(self, db, target=None, queries=None, evidence=None) :
        """Ground all queries and evidence found in the the given database.
        
        :param db: logic program
        :param target: logic formula to ground into
        :param queries: list of queries to evaluate instead of the ones in the logic program
        :param evidence: list of evidence to evaluate instead of the ones in the logic program
        :returns: ground program
        """
        raise NotImplementedError('GenericEngine.ground_all is an abstract method.')
        

class ClauseDBEngine(GenericEngine) :
    """Parent class for all Python ClauseDB-based engines."""
    
    def __init__(self, builtins=True) :
        self.__builtin_index = {}
        self.__builtins = []
        self.__externals = {}
        
        self._unique_number = 0
        
        if builtins :
            self.loadBuiltIns()
            
    def _getBuiltIn(self, index) :
        real_index = -(index + 1)
        return self.__builtins[real_index]
        
    def addBuiltIn(self, pred, arity, func) :
        """Add a builtin."""
        sig = '%s/%s' % (pred, arity)
        self.__builtin_index[sig] = -(len(self.__builtins) + 1)
        self.__builtins.append( func )
        
    def getBuiltIns(self) :
        """Get the list of builtins."""
        return self.__builtin_index
        
    def prepare(self, db) :
        """Convert given logic program to suitable format for this engine."""
        result = ClauseDB.createFrom(db, builtins=self.getBuiltIns())
        self._process_directives( result )
        return result
        
    def execute(self, node_id, database=None, context=None, target=None, **kwdargs ) :
        raise NotImplementedError("ClauseDBEngine.execute is an abstract function.")
        
    def get_non_cache_functor(self) :
        self._unique_number += 1
        return '_nocache_%s' % self._unique_number
        
    def _process_directives(self, db) :
        """Process directives present in the database."""
        term = Term('_directive')
        directive_node = db.find( term )
        if directive_node is None : return True    # no directives
        directives = db.getNode(directive_node).children
        
        gp = LogicFormula()
        while directives :
            current = directives.pop(0)
            self.execute( current, database=db, context=self._create_context((), define=None), target=gp )
        return True
            
    def _create_context(self, content, define=None) :
        """Create a variable context."""
        return content
    
    def query(self, db, term, **kwdargs) :
        """Perform a non-probabilistic query."""
        gp = LogicFormula()
        gp, result = self._ground(db, term, gp, **kwdargs)
        return [ x for x,y in result ]
    
    def ground(self, db, term, gp=None, label=None, **kwdargs) :
        """Ground a query on the given database.
        
        :param db: logic program
        :type db: LogicProgram
        :param term: query term
        :type term: Term
        :param gp: output data structure (for incremental grounding)
        :type gp: LogicFormula
        :param label: type of query (e.g. ``query``, ``evidence`` or ``-evidence``)
        :type label: str
        """
        
        if term.is_negative() :
            negated = True
            term = -term
        else :
            negated = False
        
        gp, results = self._ground(db, term, gp, silent_fail=False, allow_vars=False, **kwdargs)
        
        for args, node_id in results :
            term_store = term.withArgs(*args)
            if negated :
                gp.addName( -term_store, -node_id, label )
            else :
                gp.addName( term_store, node_id, label )
        if not results :
            gp.addName( term, None, label )
        
        return gp
        
    def _ground(self, db, term, gp=None, level=0, silent_fail=True, allow_vars=True, assume_prepared=False, **kwdargs) :
        # Convert logic program if needed.
        if not assume_prepared :
            db = self.prepare(db)
        # Create a new target datastructure if none was given.
        if gp is None : gp = LogicFormula()
        # Find the define node for the given query term.
        clause_node = db.find(term)
        # If term not defined: fail query (no error)    # TODO add error to make it consistent?
        if clause_node is None :
            # Could be builtin?
            clause_node = db._getBuiltIn(term.signature)
        if clause_node is None : 
            if silent_fail :
                return gp, []
            else :
                raise UnknownClause(term.signature, location=db.lineno(term.location))
            
        results = self.execute( clause_node, database=db, target=gp, context=list(term.args), **kwdargs)
    
        return gp, results
        
    def ground_all(self, db, target=None, queries=None, evidence=None) :
        db = self.prepare(db)
        logger = logging.getLogger('problog')
        with Timer('Grounding'):
            if queries is None : queries = [ q[0] for q in self.query(db, Term( 'query', None )) ]
            if evidence is None : evidence = self.query(db, Term( 'evidence', None, None ))
            
            if target is None : target = LogicFormula()
            
            for query in queries :
                logger.debug("Grounding query '%s'", query)
                target = self.ground(db, query, target, label=target.LABEL_QUERY)
                logger.debug("Ground program size: %s", len(target))
            for query in evidence :
                if str(query[1]) == 'true' :
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(db, query[0], target, label=target.LABEL_EVIDENCE_POS)
                    logger.debug("Ground program size: %s", len(target))
                elif str(query[1]) == 'false' :
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(db, query[0], target, label=target.LABEL_EVIDENCE_NEG)
                    logger.debug("Ground program size: %s", len(target))
                else :
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(db, query[0], target, label=target.LABEL_EVIDENCE_MAYBE)
                    logger.debug("Ground program size: %s", len(target))
        return target
    
    def addExternalCalls(self, externals):
        self.__externals.update(externals)
        
    def getExternalCall(self, func_name):
        if self.__externals is None or not func_name in self.__externals:
            return None
        return self.__externals[func_name]
    
# Generic functions

class _UnknownClause(Exception) :
    """Undefined clause in call used internally."""
    pass
        

def instantiate( term, context, keepVars=False ) :
    """Replace variables in Term by values based on context lookup table."""
    if keepVars : 
        context = list(context)
        for i,v in enumerate(context) :
            if v is None :
                context[i] = i
    if term is None :
        return None
    elif type(term) == int :
        return context[term]
    # elif is_ground(term) :
    #     return term
    else :
        return term.apply(context)
        
        
def unify( source_value, target_value, target_context=None, location=None ) :
    """Unify two terms.
        If a target context is given, the context will be updated using the variable identifiers from the first term, and the values from the second term.
        
        :raise UnifyError: unification failed
        
    """
    if type(target_value) == int :
        if target_context != None :
            current_value = target_context[target_value]
            if current_value is None :
                target_context[target_value] = source_value
            else :
                new_value = unify_value( source_value, current_value, location=location )
                target_context[target_value] = new_value
    elif target_value is None :
        pass
    else :
        assert( isinstance(target_value, Term) )
        if source_value is None :  # a variable
            pass
        else :
            assert( isinstance( source_value, Term ) )
            if target_value.signature == source_value.signature :
                for s_arg, t_arg in zip(source_value.args, target_value.args) :
                    unify( s_arg, t_arg, target_context )
            else :
                raise UnifyError()



import os
import imp, inspect # For load_external


class NonGroundProbabilisticClause(GroundingError) : 
    
    def __init__(self, location=None) :
        self.location = location
        msg = 'Encountered non-ground probabilistic clause' 
        if self.location : msg += ' at position %s:%s' % self.location
        msg += '.'
        GroundingError.__init__(self, msg)

class _UnknownClause(Exception) :
    """Undefined clause in call used internally."""
    pass

class UnknownClause(GroundingError) :
    """Undefined clause in call."""
    
    def __init__(self, signature, location) :
        self.location = location
        msg = "No clauses found for '%s'" % signature
        if location : msg += " at position %s:%s" % location
        msg += '.'
        GroundingError.__init__(self, msg)
        
class ConsultError(GroundingError) :
    
    def __init__(self, message, location=None) :
        self.location = location
        msg = message
        if location : msg += " at position %s:%s" % location
        msg += '.'
        GroundingError.__init__(self, msg)

class UnifyError(Exception) : pass

class VariableUnification(GroundingError) : 
    """The engine does not support unification of two unbound variables."""
    
    def __init__(self, location=None) :
        self.location = location
        
        msg = 'Unification of unbound variables not supported'
        if self.location : msg += ' at position %s:%s' % self.location
        msg += '.'
        GroundingError.__init__(self, msg)


def unify_value( v1, v2, location=None ) :
    """Test unification of two values and return most specific unifier."""
    
    if is_variable(v1) :
        #if not is_ground(v2) : raise VariableUnification(location=location)
        return v2
    elif is_variable(v2) :
        #if not is_ground(v1) : raise VariableUnification(location=location)
        return v1
    elif v1.signature == v2.signature : # Assume Term
        if v1 == v2 : return v1
        return v1.withArgs(*[ unify_value(a1,a2, location=location) for a1, a2 in zip(v1.args, v2.args) ])
    else :
        raise UnifyError()


class StructSort(object) :
    
    def __init__(self, obj, *args):
        self.obj = obj
    def __lt__(self, other):
        return struct_cmp(self.obj, other.obj) < 0
    def __gt__(self, other):
        return struct_cmp(self.obj, other.obj) > 0
    def __eq__(self, other):
        return struct_cmp(self.obj, other.obj) == 0
    def __le__(self, other):
        return struct_cmp(self.obj, other.obj) <= 0  
    def __ge__(self, other):
        return struct_cmp(self.obj, other.obj) >= 0
    def __ne__(self, other):
        return struct_cmp(self.obj, other.obj) != 0


class CallModeError(GroundingError) :
    
    def __init__(self, functor, args, accepted=[], message=None, location=None) :
        if functor :
            self.scope = '%s/%s'  % ( functor, len(args) )
        else :
            self.scope = None
        self.received = ', '.join(map(self.show_arg,args))
        self.expected = [  ', '.join(map(self.show_mode,mode)) for mode in accepted  ]
        self.location = location
        msg = 'Invalid argument types for call'
        if self.scope : msg += " to '%s'" % self.scope
        if location != None : msg += ' at position %s:%s ' % (location)
        msg += ': arguments: (%s)' % self.received
        if accepted :
            msg += ', expected: (%s)' % ') or ('.join(self.expected) 
        else :
            msg += ', expected: ' + message 
        Exception.__init__(self, msg)
        
    def show_arg(self, x) :
        if x is None :
            return '_'
        else :
            return str(x)
    
    def show_mode(self, t) :
        return mode_types[t][0]


def is_ground( *terms ) :
    """Test whether a any of given terms contains a variable (recursively).
    
    :return: True if none of the arguments contains any variables.
    """
    for term in terms :
        if is_variable(term) :
            return False
        elif not term.isGround() :
            return False
    return True

def is_variable( v ) :
    """Test whether a Term represents a variable.
    
    :return: True if the expression is a variable
    """
    return v is None or type(v) == int    

def is_var(term) :
    return is_variable(term) or term.isVar()

def is_nonvar(term) :
    return not is_var(term)

def is_term(term) :
    return not is_var(term) and not is_constant(term)

def is_float_pos(term) :
    return is_constant(term) and term.isFloat()

def is_float_neg(term) :
    return is_term(term) and term.arity == 1 and term.functor == "'-'" and is_float_pos(term.args[0])

def is_float(term) :
    return is_float_pos(term) or is_float_neg(term)

def is_integer_pos(term) :
    return is_constant(term) and term.isInteger()

def is_integer_neg(term) :
    return is_term(term) and term.arity == 1 and term.functor == "'-'" and is_integer_pos(term.args[0])

def is_integer(term) :
    return is_integer_pos(term) or is_integer_neg(term)

def is_string(term) :
    return is_constant(term) and term.isString()

def is_number(term) :
    return is_float(term) and is_integer(term)

def is_constant(term) :
    return not is_var(term) and term.isConstant()

def is_atom(term) :
    return is_term(term) and term.arity == 0

def is_rational(term) :
    return False

def is_dbref(term) :
    return False

def is_compound(term) :
    return is_term(term) and term.arity > 0
    
def is_list_maybe(term) :
    """Check whether the term looks like a list (i.e. of the form '.'(_,_))."""
    return is_compound(term) and term.functor == '.' and term.arity == 2
    
def is_list_nonempty(term) :
    if is_list_maybe(term) :
        tail = list_tail(term)
        return is_list_empty(tail) or is_var(tail)
    return False

def is_fixed_list(term) :
    return is_list_empty(term) or is_fixed_list_nonempty(term)

def is_fixed_list_nonempty(term) :
    if is_list_maybe(term) :
        tail = list_tail(term)
        return is_list_empty(tail)
    return False
    
def is_list_empty(term) :
    return is_atom(term) and term.functor == '[]'
    
def is_list(term) :
    return is_list_empty(term) or is_list_nonempty(term)
    
def is_compare(term) :
    return is_atom(term) and term.functor in ("'<'", "'='", "'>'")

mode_types = {
    'i' : ('integer', is_integer),
    'I' : ('positive_integer', is_integer_pos),
    'v' : ('var', is_var),
    'n' : ('nonvar', is_nonvar),
    'l' : ('list', is_list),
    'L' : ('fixed_list', is_fixed_list),    # List of fixed length (i.e. tail is [])
    '*' : ('any', lambda x : True ),
    '<' : ('compare', is_compare ),         # < = >
    'g' : ('ground', is_ground ),
    'a' : ('atom', is_atom),
    'c' : ('callable', is_term)
}

def check_mode( args, accepted, functor=None, location=None, database=None, **k) :
    for i, mode in enumerate(accepted) :
        correct = True
        for a,t in zip(args,mode) :
            name, test = mode_types[t]
            if not test(a) : 
                correct = False
                break
        if correct : return i
    if database and location :
        location = database.lineno(location)
    else :
        location = None
    raise CallModeError(functor, args, accepted, location=location)
    
def list_elements(term) :
    elements = []
    tail = term
    while is_list_maybe(tail) :
        elements.append(tail.args[0])
        tail = tail.args[1]
    return elements, tail
    
def list_tail(term) :
    tail = term
    while is_list_maybe(tail) :
        tail = tail.args[1]
    return tail
               

def builtin_split_call( term, parts, database=None, location=None, **k ) :
    """T =.. L"""
    functor = '=..'
    # modes:
    #   <v> =.. list  => list has to be fixed length and non-empty
    #                       IF its length > 1 then first element should be an atom
    #   <n> =.. <list or var>
    #
    mode = check_mode( (term, parts), ['vL', 'nv', 'nl' ], functor=functor, **k )
    if mode == 0 :
        elements, tail = list_elements(parts)
        if len(elements) == 0 :
            raise CallModeError(functor, (term,parts), message='non-empty list for arg #2 if arg #1 is a variable', location=database.lineno(location))
        elif len(elements) > 1 and not is_atom(elements[0]) :
            raise CallModeError(functor, (term,parts), message='atom as first element in list if arg #1 is a variable', location=database.lineno(location))
        elif len(elements) == 1 :
            # Special case => term == parts[0]
            return [(elements[0],parts)]
        else :
            T = elements[0](*elements[1:])
            return [ (T , parts) ] 
    else :
        part_list = ( term.withArgs(), ) + term.args
        current = Term('[]')
        for t in reversed(part_list) :
            current = Term('.', t, current)
        try :
            L = unify_value(current, parts, location=location)
            elements, tail = list_elements(L)
            term_new = elements[0](*elements[1:])
            T = unify_value( term, term_new, location=location )
            return [(T,L)]            
        except UnifyError :
            return []

def builtin_arg(index,term,argument, **k) :
    mode = check_mode( (index,term,arguments), ['In*'], functor='arg', **k)
    index_v = int(index) - 1
    if 0 <= index_v < len(term.args) :
        try :
            arg = term.args[index_v]
            res = unify_value(arg,argument, location=location)
            return [(index,term,res)]
        except UnifyError :
            pass
    return []

def builtin_functor(term,functor,arity, **k) :
    mode = check_mode( (term,functor,arity), ['vaI','n**'], functor='functor', **k)
    
    if mode == 0 : 
        callback.newResult( Term(functor, *((None,)*int(arity)) ), functor, arity )
    else :
        try :
            func_out = unify_value(functor, Term(term.functor))
            arity_out = unify_value(arity, Constant(term.arity))
            return [(term, func_out, arity_out)]
        except UnifyError :
            pass
    return []

def builtin_true( **k ) :
    """``true``"""
    return True


def builtin_fail( **k ) :
    """``fail``"""
    return False


def builtin_eq( A, B, location=None, database=None, **k ) :
    """``A = B``
        A and B not both variables
    """
    if not is_ground(A) and not is_ground(B) :
        raise VariableUnification(location = database.lineno(location))
    else :
        try :
            R = unify_value(A,B, location=location)
            return [( R, R )]
        except UnifyError :
            return []
        except VariableUnification :
            raise VariableUnification(location = database.lineno(location))


def builtin_neq( A, B, **k ) :
    """``A \= B``
        A and B not both variables
    """
    if is_var(A) and is_var(B) :
        return False
    else :
        try :
            R = unify_value(A,B)
            return False
        except UnifyError :
            return True
        except VariableUnification :
            return False
            

def builtin_notsame( A, B, **k ) :
    """``A \== B``"""
    if is_var(A) and is_var(B) :
        return False
    # In Python A != B is not always the same as not A == B.
    else :
        return not A == B


def builtin_same( A, B, **k ) :
    """``A == B``"""
    if is_var(A) and is_var(B) :
        return True
    else :
        return A == B


def builtin_gt( A, B, **k ) :
    """``A > B`` 
        A and B are ground
    """
    mode = check_mode( (A,B), ['gg'], functor='>', **k )
    return A.value > B.value


def builtin_lt( A, B, **k ) :
    """``A > B`` 
        A and B are ground
    """
    mode = check_mode( (A,B), ['gg'], functor='<', **k )
    return A.value < B.value


def builtin_le( A, B, **k ) :
    """``A =< B``
        A and B are ground
    """
    mode = check_mode( (A,B), ['gg'], functor='=<', **k )
    return A.value <= B.value


def builtin_ge( A, B, **k ) :
    """``A >= B`` 
        A and B are ground
    """
    mode = check_mode( (A,B), ['gg'], functor='>=', **k )
    return A.value >= B.value


def builtin_val_neq( A, B, **k ) :
    """``A =\= B`` 
        A and B are ground
    """
    mode = check_mode( (A,B), ['gg'], functor='=\=', **k )
    return A.value != B.value


def builtin_val_eq( A, B, **k ) :
    """``A =:= B`` 
        A and B are ground
    """
    mode = check_mode( (A,B), ['gg'], functor='=:=', **k )
    return A.value == B.value


def builtin_is( A, B, **k ) :
    """``A is B``
        B is ground
    """
    mode = check_mode( (A,B), ['*g'], functor='is', **k )
    try :
        R = Constant(B.value)
        unify_value(A,R)
        return [(R,B)]
    except UnifyError :
        return []


def builtin_var( term, **k ) :
    return is_var(term)


def builtin_atom( term, **k ) :
    return is_atom(term)


def builtin_atomic( term, **k ) :
    return is_atom(term) or is_number(term)


def builtin_compound( term, **k ) :
    return is_compound(term)


def builtin_float( term, **k ) :
    return is_float(term)


def builtin_integer( term, **k ) :
    return is_integer(term)


def builtin_nonvar( term, **k ) :
    return not is_var(term)


def builtin_number( term, **k ) :
    return is_number(term) 


def builtin_simple( term, **k ) :
    return is_var(term) or is_atomic(term)
    

def builtin_callable( term, **k ) :
    return is_term(term)


def builtin_rational( term, **k ) :
    return is_rational(term)


def builtin_dbreference( term, **k ) :
    return is_dbref(term)  
    

def builtin_primitive( term, **k ) :
    return is_atomic(term) or is_dbref(term)


def builtin_ground( term, **k ) :
    return is_ground(term)


def builtin_is_list( term, **k ) :
    return is_list(term)

def compare(a,b) :
    if a < b :
        return -1
    elif a > b :
        return 1
    else :
        return 0
    
def struct_cmp( A, B ) :
    # Note: structural comparison
    # 1) Var < Num < Str < Atom < Compound
    # 2) Var by address
    # 3) Number by value, if == between int and float => float is smaller (iso prolog: Float always < Integer )
    # 4) String alphabetical
    # 5) Atoms alphabetical
    # 6) Compound: arity / functor / arguments
        
    # 1) Variables are smallest
    if is_var(A) :
        if is_var(B) :
            # 2) Variable by address
            return compare(A,B)
        else :
            return -1
    elif is_var(B) :
        return 1
    # assert( not is_var(A) and not is_var(B) )
    
    # 2) Numbers are second smallest
    if is_number(A) :
        if is_number(B) :
            # Just compare numbers on float value
            res = compare(float(A),float(B))
            if res == 0 :
                # If the same, float is smaller.
                if is_float(A) and is_integer(B) : 
                    return -1
                elif is_float(B) and is_integer(A) : 
                    return 1
                else :
                    return 0
        else :
            return -1
    elif is_number(B) :
        return 1
        
    # 3) Strings are third
    if is_string(A) :
        if is_string(B) :
            return compare(str(A),str(B))
        else :
            return -1
    elif is_string(B) :
        return 1
    
    # 4) Atoms / terms come next
    # 4.1) By arity
    res = compare(A.arity,B.arity)
    if res != 0 : return res
    
    # 4.2) By functor
    res = compare(A.functor,B.functor)
    if res != 0 : return res
    
    # 4.3) By arguments (recursively)
    for a,b in zip(A.args,B.args) :
        res = struct_cmp(a,b)
        if res != 0 : return res
        
    return 0

    
def builtin_struct_lt(A, B, **k) :
    return struct_cmp(A,B) < 0    

    
def builtin_struct_le(A, B, **k) :
    return struct_cmp(A,B) <= 0

    
def builtin_struct_gt(A, B, **k) :
    return struct_cmp(A,B) > 0

    
def builtin_struct_ge(A, B, **k) :
    return struct_cmp(A,B) >= 0


def builtin_compare(C, A, B, **k) :
    mode = check_mode( (C,A,B), [ '<**', 'v**' ], functor='compare', **k)
    compares = "'>'","'='","'<'" 
    c = struct_cmp(A,B)
    c_token = compares[1-c]
    
    if mode == 0 : # Given compare
        if c_token == C.functor : return [ (C,A,B) ]
    else :  # Unknown compare
        return [ (Term(c_token), A, B ) ]
    
# numbervars(T,+N1,-Nn)    number the variables TBD?

def build_list(elements, tail) :
    current = tail
    for el in reversed(elements) :
        current = Term('.', el, current)
    return current


def builtin_call_external(call, result, **k):
    from . import pypl
    mode = check_mode( (call,result), ['gv'], function='call_external', **k)

    func = k['engine'].getExternalCall(call.functor)
    if func is None:
        raise Exception('External method not known: {}'.format(call.functor))

    values = [pypl.pl2py(arg) for arg in call.args]
    computed_result = func(*values)

    return [(call, pypl.py2pl(computed_result))]


def builtin_length(L, N, **k) :
    mode = check_mode( (L,N), [ 'LI', 'Lv', 'lI', 'vI' ], functor='length', **k)
    # Note that Prolog also accepts 'vv' and 'lv', but these are unbounded.
    # Note that lI is a subset of LI, but only first matching mode is returned.
    if mode == 0 or mode == 1 :  # Given fixed list and maybe length
        elements, tail = list_elements(L)
        list_size = len(elements)
        try :
            N = unify_value(N, Constant(list_size))
            return [ ( L, N ) ]
        except UnifyError :
            return []    
    else :    # Unbounded list or variable list and fixed length.
        if mode == 2 :
            elements, tail = list_elements(L)
        else :
            elements, tail = [], L
        remain = int(N) - len(elements)
        if remain < 0 :
            raise UnifyError()
        else :
            extra = [None] * remain
        newL = build_list( elements + extra, Term('[]'))
        return [ (newL, N)]

def extract_vars(*args, **kwd) :
    counter = kwd.get('counter', defaultdict(int))
    for arg in args :
        if type(arg) == int :
            counter[arg] += 1
        elif isinstance(arg,Term) :
            extract_vars(*arg.args, counter=counter)
    return counter


def builtin_sort( L, S, **k ) :
    # TODO doesn't work properly with variables e.g. gives sort([X,Y,Y],[_]) should be sort([X,Y,Y],[X,Y])
    mode = check_mode( (L,S), [ 'L*' ], functor='sort', **k )
    elements, tail = list_elements(L)  
    # assert( is_list_empty(tail) )
    try :
        sorted_list = build_list(sorted(set(elements), key=StructSort), Term('[]'))
        S_out = unify_value(S,sorted_list)
        return [(L,S_out)]
    except UnifyError :
        return []


def builtin_between( low, high, value, **k ) :
    mode = check_mode((low,high,value), [ 'iii', 'iiv' ], functor='between', **k)
    low_v = int(low)
    high_v = int(high)
    if mode == 0 : # Check    
        value_v = int(value)
        if low_v <= value_v <= high_v :
            return [(low,high,value)]
    else : # Enumerate
        results = []
        for value_v in range(low_v, high_v+1) :
            results.append( (low,high,Constant(value_v)) ) 
        return results


def builtin_succ( a, b, **k ) :
    mode = check_mode((a,b), [ 'vI', 'Iv', 'II' ], functor='succ', **k)
    if mode == 0 :
        b_v = int(b)
        return [(Constant(b_v-1), b)]
    elif mode == 1 :
        a_v = int(a)
        return [(a, Constant(a_v+1))]
    else :
        a_v = int(a)
        b_v = int(b)
        if b_v == a_v + 1 :
            return [(a, b)]
    return []


def builtin_plus( a, b, c , **k) :
    mode = check_mode((a,b,c), [ 'iii', 'iiv', 'ivi', 'vii' ], functor='plus', **k)
    if mode == 0 :
        a_v = int(a)
        b_v = int(b)
        c_v = int(c)
        if a_v + b_v == c_v :
            return [(a,b,c)]
    elif mode == 1 :
        a_v = int(a)
        b_v = int(b)
        return [(a, b, Constant(a_v+b_v))]
    elif mode == 2 :
        a_v = int(a)
        c_v = int(c)
        return [(a, Constant(c_v-a_v), c)]
    else :
        b_v = int(b)
        c_v = int(c)
        return [(Constant(c_v-b_v), b, c)]
    return []


def atom_to_filename(atom) :
    atom = str(atom)
    if atom[0] == atom[-1] == "'" :
        atom = atom[1:-1]
    return atom

    
def builtin_consult_as_list( op1, op2, **kwdargs ) :
    check_mode( (op1,op2), ['*L'], functor='consult', **kwdargs )
    builtin_consult(op1, **kwdargs)
    if is_list_nonempty(op2) :
        builtin_consult_as_list(op2.args[0], op2.args[1], **kwdargs)
    return True
    
def builtin_consult( filename, database=None, engine=None, location=None, **kwdargs ) :
    check_mode( (filename,), 'a', functor='consult' )
    filename = os.path.join(database.source_root, atom_to_filename( filename ))
    if not os.path.exists( filename ) :
        filename += '.pl'
    if not os.path.exists( filename ) :
        raise ConsultError(location=database.lineno(location), message="Consult: file not found '%s'" % filename)
    
    # Prevent loading the same file twice
    if not filename in database.source_files : 
        database.source_files.append(filename)
        pl = PrologFile( filename )
        for clause in pl :
            database += clause
    return True


def builtin_load_external( arg, engine=None, database=None, location=None, **kwdargs ) :
    check_mode( (arg,), 'a', functor='load_external' )
    # Load external (python) files that are referenced in the model
    externals = {}
    filename = os.path.join(database.source_root, atom_to_filename( arg ))
    if not os.path.exists(filename):
          raise ConsultError(location=database.lineno(location), message="Load external: file not found '%s'" % filename)
    try :
        with open(filename, 'r') as extfile:
            ext = imp.load_module('externals', extfile, filename, ('.py', 'U', 1))
            for func_name, func in inspect.getmembers(ext, inspect.isfunction):
                externals[func_name] = func
        engine.addExternalCalls(externals)
    except ImportError :
        raise ConsultError(location=database.lineno(location), message="Error while loading external file '%s'" % filename)        
    
    return True
    
def builtin_unknown( arg, engine=None, **kwdargs) :
    check_mode( (arg,), 'a', functor='unknown')
    if arg.functor == 'fail' :
        engine.unknown = engine.UNKNOWN_FAIL
    else :
        engine.unknown = engine.UNKNOWN_ERROR
    return True
    
def select( lst, target ) :
    # TODO remove recursion?
    if lst :
        res, node = lst[0]
        lst = lst[1:]
        
        for list_rest, node_rest in select(lst, target) :
            if node == target.TRUE :  # Have to pick
                yield  (res,) + list_rest, node_rest
            else :  # Have choice
                yield (res,) + list_rest, (node,) + node_rest
                yield list_rest, (-node,) + node_rest
    else :
        yield (), (0,)
    
    
def builtin_findall( pattern, goal, result, database=None, target=None, engine=None, **kwdargs ) :
    mode = check_mode( (result,), 'vl' )
    
    findall_head = Term(engine.get_non_cache_functor(), pattern)
    findall_clause = Clause( findall_head , goal )    
    findall_db = ClauseDB(parent=database)
    findall_db += findall_clause
    results = engine.call( findall_head, subcall=True, database=findall_db, target=target, **kwdargs )
    results = [ (res[0],n) for res, n in results ]
    
    output = []
    for l,n in select(results, target) :
        node = target.addAnd(n)
        if node != None :
            res = build_list(l,Term('[]'))
            if mode == 0  :  # var
                output.append(((pattern,goal,res), node))
            else :
                try :
                    res = unify_value(res,result)
                    output.append(((pattern,goal,res), node))
                except UnifyError :
                    pass
    return output

def addStandardBuiltIns(engine, b=None, s=None, sp=None) :
    """Add Prolog builtins to the given engine."""
    
    # Shortcut some wrappers
    if b is None : b = BooleanBuiltIn
    if s is None : s = SimpleBuiltIn
    if sp is None : sp = SimpleProbabilisticBuiltIn
    
    engine.addBuiltIn('true', 0, b(builtin_true))   # -1
    engine.addBuiltIn('fail', 0, b(builtin_fail))   # -2
    engine.addBuiltIn('false', 0, b(builtin_fail))  # -3

    engine.addBuiltIn('=', 2, s(builtin_eq))        # -4
    engine.addBuiltIn('\=', 2, b(builtin_neq))      # -5

    engine.addBuiltIn('findall',3,sp(builtin_findall)) # -6

    engine.addBuiltIn('==', 2, b(builtin_same))
    engine.addBuiltIn('\==', 2, b(builtin_notsame))

    engine.addBuiltIn('is', 2, s(builtin_is))

    engine.addBuiltIn('>', 2, b(builtin_gt))
    engine.addBuiltIn('<', 2, b(builtin_lt))
    engine.addBuiltIn('=<', 2, b(builtin_le))
    engine.addBuiltIn('>=', 2, b(builtin_ge))
    engine.addBuiltIn('=\=', 2, b(builtin_val_neq))
    engine.addBuiltIn('=:=', 2, b(builtin_val_eq))

    engine.addBuiltIn('var', 1, b(builtin_var))
    engine.addBuiltIn('atom', 1, b(builtin_atom))
    engine.addBuiltIn('atomic', 1, b(builtin_atomic))
    engine.addBuiltIn('compound', 1, b(builtin_compound))
    engine.addBuiltIn('float', 1, b(builtin_float))
    engine.addBuiltIn('rational', 1, b(builtin_rational))
    engine.addBuiltIn('integer', 1, b(builtin_integer))
    engine.addBuiltIn('nonvar', 1, b(builtin_nonvar))
    engine.addBuiltIn('number', 1, b(builtin_number))
    engine.addBuiltIn('simple', 1, b(builtin_simple))
    engine.addBuiltIn('callable', 1, b(builtin_callable))
    engine.addBuiltIn('dbreference', 1, b(builtin_dbreference))
    engine.addBuiltIn('primitive', 1, b(builtin_primitive))
    engine.addBuiltIn('ground', 1, b(builtin_ground))
    engine.addBuiltIn('is_list', 1, b(builtin_is_list))
    
    engine.addBuiltIn('=..', 2, s(builtin_split_call))
    engine.addBuiltIn('arg', 3, s(builtin_arg))
    engine.addBuiltIn('functor', 3, s(builtin_functor))
    
    engine.addBuiltIn('@>',2, b(builtin_struct_gt))
    engine.addBuiltIn('@<',2, b(builtin_struct_lt))
    engine.addBuiltIn('@>=',2, b(builtin_struct_ge))
    engine.addBuiltIn('@=<',2, b(builtin_struct_le))
    engine.addBuiltIn('compare',3, s(builtin_compare))

    engine.addBuiltIn('length',2, s(builtin_length))
    engine.addBuiltIn('call_external',2, s(builtin_call_external))

    engine.addBuiltIn('sort',2, s(builtin_sort))
    engine.addBuiltIn('between', 3, s(builtin_between))
    engine.addBuiltIn('succ',2, s(builtin_succ))
    engine.addBuiltIn('plus',3, s(builtin_plus))
    
    engine.addBuiltIn('consult', 1, b(builtin_consult))
    engine.addBuiltIn('.', 2, b(builtin_consult_as_list))
    engine.addBuiltIn('load_external', 1, b(builtin_load_external))
    engine.addBuiltIn('unknown',1,b(builtin_unknown))

#from .engine_stack_opt import OptimizedStackBasedEngine as DefaultEngine
from .engine_stack import StackBasedEngine as DefaultEngine

def intersection(l1, l2) :
    i = 0
    j = 0
    n1 = len(l1)
    n2 = len(l2)
    r = []
    a = r.append
    while i < n1 and j < n2 :
        if l1[i] == l2[j] :
            a(l1[i])
            i += 1
            j += 1
        elif l1[i] < l2[j] :
            i += 1
        else :
            j += 1
    #print ('I', l1, l2, r)
    return r

class ClauseIndex(list) :
    
    def __init__(self, parent, arity) :
        self.__parent = parent
        self.__index = [ defaultdict(set) for i in range(0,arity) ]
        self.__optimized = False
        
    def optimize(self) :
        if not self.__optimized :
            self.__optimized = True
            for i in range(0,len(self.__index)) :
                arg_index = self.__index[i]
                arg_none = arg_index[None]
                self.__index[i] = { k : tuple(sorted(v | arg_none)) for k,v in arg_index.items() if k != None }
                self.__index[i][None] = tuple(sorted(arg_none))
        
    def find(self, arguments) :
        self.optimize()
        results = None
        # for i, xx in enumerate(self.__index) :
        #     print ('\t', i, xx)
        for i, arg in enumerate(arguments) :
            if arg is None or type(arg) == int or not arg.isGround() : 
                pass # Variable => no restrictions
            else :
                curr = self.__index[i].get(arg)
                if curr is None :   # No facts matching this argument exactly.
                    results = self.__index[i].get(None)
                elif results is None :  # First argument with restriction
                    results = curr
                else :  # Already have a selection
                    results = intersection(results, curr)
            if results == [] : 
                # print ('FIND', arguments, results)
                return []
        if results is None :
            # print ('FIND', arguments, 'all')
            return self
        else :
            # print ('FIND', arguments, results)
            return results
    
    def _add(self, key, item) :
        for i, k in enumerate(key) :
            self.__index[i][k].add(item)
        
    def append(self, item) :
        list.append(self, item)
        key = []
        args = self.__parent.getNode(item).args
        for arg in args :
            if isinstance(arg,Term) and arg.isGround() :
                key.append(arg)
            else :
                key.append(None)
        self._add(key, item)
        

class ClauseDB(LogicProgram) :
    """Compiled logic program.
    
    A logic program is compiled into a table of instructions.
    The types of instructions are:
    
    define( functor, arity, defs )
        Pointer to all definitions of functor/arity.
        Definitions can be: ``fact``, ``clause`` or ``adc``.
    
    clause( functor, arguments, bodynode, varcount )
        Single clause. Functor is the head functor, Arguments are the head arguments. Body node is a pointer to the node representing the body. Var count is the number of variables in head and body.
        
    fact( functor, arguments, probability )
        Single fact. 
        
    adc( functor, arguments, bodynode, varcount, parent )
        Single annotated disjunction choice. Fields have same meaning as with ``clause``, parent_node points to the parent ``ad`` node.
        
    ad( childnodes )
        Annotated disjunction group. Child nodes point to the ``adc`` nodes of the clause.

    call( functor, arguments, defnode )
        Body literal with call to clause or builtin. Arguments contains the call arguments, definition node is the pointer to the definition node of the given functor/arity.
    
    conj( childnodes )
        Logical and. Currently, only 2 children are supported.
    
    disj( childnodes )
        Logical or. Currently, only 2 children are supported.
    
    neg( childnode )
        Logical not.
                
    .. todo:: 
        
        * add annotated disjunctions (*ad*)
        * add probability field
        * remove empty nodes -> replace by None pointer in call => requires prior knowledge of builtins
    
    """
    
    _define = namedtuple('define', ('functor', 'arity', 'children', 'location') )
    _clause = namedtuple('clause', ('functor', 'args', 'probability', 'child', 'varcount', 'group', 'location') )
    _fact   = namedtuple('fact'  , ('functor', 'args', 'probability', 'location') )
    _call   = namedtuple('call'  , ('functor', 'args', 'defnode', 'location' )) 
    _disj   = namedtuple('disj'  , ('children', 'location' ) )
    _conj   = namedtuple('conj'  , ('children', 'location' ) )
    _neg    = namedtuple('neg'   , ('child', 'location' ) )
    _choice = namedtuple('choice', ('functor', 'args', 'probability', 'group', 'choice', 'location') )
    
    def __init__(self, builtins=None, parent=None) :
        LogicProgram.__init__(self)
        self.__nodes = []   # list of nodes
        self.__heads = {}   # head.sig => node index
        
        self.__builtins = builtins
        
        self.__parent = parent
        if parent is None :
            self.__offset = 0
        else :
            self.__offset = len(parent)
    
    def __len__(self) :
        return len(self.__nodes) + self.__offset
        
    def extend(self) :
        return ClauseDB(parent=self)
        
    def _getBuiltIn(self, signature) :
        if self.__builtins is None :
            if self.__parent != None :
                return self.__parent._getBuiltIn(signature)
            else :
                return None
        else :
            return self.__builtins.get(signature)
    
    def _create_index(self, arity) :
        # return []
        return ClauseIndex(self, arity)
            
    def _addAndNode( self, op1, op2, location=None ) :
        """Add an *and* node."""
        return self._appendNode( self._conj((op1,op2),location))
        
    def _addNotNode( self, op1, location=None ) :
        """Add a *not* node."""
        return self._appendNode( self._neg(op1,location) )
        
    def _addOrNode( self, op1, op2, location=None ) :
        """Add an *or* node."""
        return self._appendNode( self._disj((op1,op2),location))
    
    def _addDefineNode( self, head, childnode ) :
        define_index = self._addHead( head )
        define_node = self.getNode(define_index)
        if not define_node :
            clauses = self._create_index(head.arity)
            self._setNode( define_index, self._define( head.functor, head.arity, clauses, head.location ) )
        else :
            clauses = define_node.children
        clauses.append( childnode )
        return childnode
    
    def _addChoiceNode(self, choice, args, probability, group, location=None) :
        functor = 'ad_%s_%s' % (group, choice)
        choice_node = self._appendNode( self._choice(functor, args, probability, group, choice, location) )
        return choice_node
        
    def _addClauseNode( self, head, body, varcount, group=None ) :
        clause_node = self._appendNode( self._clause( head.functor, head.args, head.probability, body, varcount, group, head.location ) )
        return self._addDefineNode( head, clause_node )
        
    def _addCallNode( self, term ) :
        """Add a *call* node."""
        defnode = self._addHead(term, create=False)
        return self._appendNode( self._call( term.functor, term.args, defnode, term.location ) )
    
    def getNode(self, index) :
        """Get the instruction node at the given index.
        
        :param index: index of the node to retrieve
        :type index: :class:`int`
        :returns: requested node
        :rtype: :class:`tuple`
        :raises IndexError: the given index does not point to a node
        
        """
        if index < self.__offset :
            return self.__parent.getNode(index)
        else :
            return self.__nodes[index-self.__offset]
        
    def _setNode(self, index, node) :
        if index < self.__offset :
            raise IndexError('Can\'t update node in parent.')
        else :
            self.__nodes[index-self.__offset] = node
        
    def _appendNode(self, node=()) :
        index = len(self)
        self.__nodes.append( node )
        return index
    
    def _getHead(self, head) :
        node = self.__heads.get( head.signature )
        if node is None and self.__parent :
            node = self.__parent._getHead(head)
        return node
        
    def _setHead(self, head, index) :
        self.__heads[ head.signature ] = index
    
    def _addHead( self, head, create=True ) :
        node = self._getBuiltIn( head.signature )
        if node != None :
            if create :
                raise AccessError("Can not overwrite built-in '%s'." % head.signature )
            else :
                return node
        
        node = self._getHead( head )
        if node is None :
            if create :
                node = self._appendNode( self._define( head.functor, head.arity, self._create_index(head.arity), head.location) )
            else :
                node = self._appendNode()
            self._setHead( head, node )
        return node

    def find(self, head ) :
        """Find the ``define`` node corresponding to the given head.
        
        :param head: clause head to match
        :type head: :class:`.basic.Term`
        :returns: location of the clause node in the database, returns ``None`` if no such node exists
        :rtype: :class:`int` or ``None``
        """
        return self._getHead( head )
       
    def __repr__(self) :
        s = ''
        for i,n in enumerate(self.__nodes) :
            i += self.__offset
            s += '%s: %s\n' % (i,n)
        s += str(self.__heads)
        return s
        
    def _addClause(self, clause) :
        """Add a clause to the database.
        
        :param clause: Clause to add
        :type clause: :class:`.Clause`
        :returns: location of the definition node in the database
        :rtype: :class:`int`
        """
        return self._compile( clause )
    
    def _addAnnotatedDisjunction(self, clause) :
        return self._compile( clause )
    
    def _addFact( self, term) :
        variables = _AutoDict()
        new_head = term.apply(variables)
        if len(variables) == 0 :
            fact_node = self._appendNode( self._fact(term.functor, term.args, term.probability, term.location))
            return self._addDefineNode( term, fact_node )
        else :
            return self._addClause( Clause(term, Term('true')) )
    
    def _compile(self, struct, variables=None) :
        if variables is None : variables = _AutoDict()
        
        if isinstance(struct, And) :
            op1 = self._compile(struct.op1, variables)
            op2 = self._compile(struct.op2, variables)
            return self._addAndNode( op1, op2)
        elif isinstance(struct, Or) :
            op1 = self._compile(struct.op1, variables)
            op2 = self._compile(struct.op2, variables)
            return self._addOrNode( op1, op2)
        elif isinstance(struct, Not) :
            child = self._compile(struct.child, variables)
            return self._addNotNode( child, location=struct.location)
        elif isinstance(struct, AnnotatedDisjunction) :
            # Determine number of variables in the head
            new_heads = [ head.apply(variables) for head in struct.heads ]            
            head_count = len(variables)
            
            # Body arguments
            body_args = tuple(range(0,head_count))
            
            # Group id
            group = len(self.__nodes)
            
            # Create the body clause
            body_head = Term('ad_%s_body' % group, *body_args)
            body_node = self._compile(struct.body, variables)
            clause_body = self._addClauseNode( body_head, body_node, len(variables) )
            #clause_body = self._appendNode( self._clause( body_head.functor, body_head.args, None, body_node, len(variables), group=None ) )
            clause_body = self._addHead( body_head )
            for choice, head in enumerate(new_heads) :
                # For each head: add choice node
                choice_node = self._addChoiceNode(choice, body_args, head.probability, group, head.location )
                choice_call = self._appendNode( self._call( 'ad_%s_%s' % (group, choice), body_args, choice_node, head.location ) )
                body_call = self._appendNode( self._call( 'ad_%s_body' % group, body_args , clause_body, head.location ) )
                choice_body = self._addAndNode( body_call, choice_call )
                head_clause = self._addClauseNode( head, choice_body, head_count, group=group )
            return None
        elif isinstance(struct, Clause) :
            if struct.head.probability != None :
                return self._compile( AnnotatedDisjunction( [struct.head], struct.body))
            else :
                new_head = struct.head.apply(variables)
                head_count = len(variables)
                body_node = self._compile(struct.body, variables)
                return self._addClauseNode(new_head, body_node, len(variables))
        elif isinstance(struct, Term) :
            return self._addCallNode( struct.apply(variables) )
        else :
            raise ValueError("Unknown structure type: '%s'" % struct )
    
    def _create_vars(self, term) :
        if type(term) == int :
            return Var('V_' + str(term))
        else :
            args = [ self._create_vars(arg) for arg in term.args ]
            return term.withArgs(*args)
        
    def _extract(self, node_id) :
        node = self.getNode(node_id)
        if not node :
            raise ValueError("Unexpected empty node.")    
        
        groups = defaultdict(list)
        nodetype = type(node).__name__
        if nodetype == 'fact' :
            return Term(node.functor, *node.args, p=node.probability)
        elif nodetype == 'clause' :
            if clause.group != None :   # part of annotated disjunction
                groups
            else :
                head = self._create_vars( Term(node.functor,*node.args, p=node.probability) )
                return Clause( head, self._extract(node.child))
        elif nodetype == 'call' :
            func = node.functor
            args = node.args
            return self._create_vars( Term(func, *args) )
        elif nodetype == 'conj' :
            a,b = node.children
            return And( self._extract(a), self._extract(b) )
        elif nodetype == 'disj' :
            a,b = node.children
            return Or( self._extract(a), self._extract(b) )
        elif nodetype == 'neg' :
            return Not( self._extract(node.child))
            
        else :
            raise ValueError("Unknown node type: '%s'" % nodetype)    
        
    def __iter__(self) :
        clause_groups = defaultdict(list)
        for index, node in enumerate(self.__nodes) :
            index += self.__offset
            if not node : continue
            nodetype = type(node).__name__
            if nodetype == 'fact' :
                yield Term(node.functor, *node.args, p=node.probability)
            elif nodetype == 'clause' :
                if node.group is None :
                    head = self._create_vars( Term(node.functor,*node.args, p=node.probability) )
                    yield Clause( head, self._extract(node.child))
                else :
                    clause_groups[node.group].append(index)
            
            
            #if node and type(node).__name__ in ('fact', 'clause') :
               # yield self._extract( index )
        for group in clause_groups.values() :
            heads = []
            body = None
            for index in group :
                node = self.getNode(index)
                heads.append( self._create_vars( Term( node.functor, *node.args, p=node.probability)))
                if body is None :
                    body_node = self.getNode(node.child)
                    body_node = self.getNode(body_node.children[0])
                    body = self._create_vars( Term(body_node.functor, *body_node.args) )
            yield AnnotatedDisjunction(heads, body)

class AccessError(Exception) : pass            
        
class _AutoDict(dict) :
    
    def __init__(self) :
        dict.__init__(self)
        self.__record = set()
        self.__anon = 0
    
    def __getitem__(self, key) :
        if key == '_' :
            value = len(self)
            self.__anon += 1
            return value
        else :        
            value = self.get(key)
            if value is None :
                value = len(self)
                self[key] = value
            self.__record.add(value)
            return value
            
    def __len__(self) :
        return dict.__len__(self) + self.__anon
        
    def usedVars(self) :
        result = set(self.__record)
        self.__record.clear()
        return result
        
    def define(self, key) :
        if not key in self :
            value = len(self)
            self[key] = value
            
