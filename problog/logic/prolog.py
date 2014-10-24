from .engine import Engine

from .basic import Constant

# operators
#
# self.addOperator('<'      , 700 , function=lambda a, b : a < b)
# self.addOperator('=<'     , 700 , function=lambda a, b : a <= b)
# self.addOperator('=:='    , 700 , function=lambda a, b : a == b)
# self.addOperator('>='     , 700 , function=lambda a, b : a >= b)
# self.addOperator('>'      , 700 , function=lambda a, b : a > b)
# self.addOperator('=\='    , 700 , function=lambda a, b : a != b)

# self.addOperator('@<'      , 700 , function=lambda a, b : a < b)
# self.addOperator('@=<'     , 700 , function=lambda a, b : a <= b)
# self.addOperator('=@='     , 700 , function=lambda a, b : a == b)
# self.addOperator('@>='     , 700 , function=lambda a, b : a >= b)
# self.addOperator('@>'      , 700 , function=lambda a, b : a > b)
# self.addOperator('\=@='    , 700 , function=lambda a, b : a != b)

# self.addOperator( ':'   , 600, 'xfy', 
# self.addOperator( '+'   , 500, 'yfx', function=lambda a, b : a + b)
# self.addOperator( '-'   , 500, 'yfx', function=lambda a, b : a - b)
# self.addOperator( '/\\' , 500, 'yfx', function=lambda a, b : a & b)
# self.addOperator( '\/'  , 500, 'yfx', function=lambda a, b : a | b)
# self.addOperator( 'xor' , 500, 'yfx', function=lambda a, b : a ^ b)
#
#
# self.addOperator( '*'   , 400 , 'yfx', function=lambda a, b : a * b)
# self.addOperator( '/'   , 400 , 'yfx', function=lambda a, b : a / b)
# self.addOperator( '//'  , 400 , 'yfx', function=lambda a, b : a // b)
# self.addOperator( 'rdiv', 400 , 'yfx', rational number division
# self.addOperator( '<<'  , 400 , 'yfx', function=lambda a, b : a << b)
# self.addOperator( '>>'  , 400 , 'yfx', function=lambda a, b : a >> b)
# self.addOperator( 'mod' , 400 , 'yfx', function=lambda a, b : a % b)
# self.addOperator( 'rem' , 400 , 'yfx', function=lambda a, b : a % b)
#
# self.addOperator( '**'  , 200 , 'xfx', function=lambda a, b : a ** b)
# self.addOperator( '^'   , 400 , 'xfy', function=lambda a, b : a ** b)
#
# self.addOperator( '+' , 200, 'fy', function=lambda a: a )
# self.addOperator( '-' , 200, 'fy', function=lambda a: -a )
# self.addOperator( '\\', 200, 'fy', function=lambda a: ~a )    (single backslash)



# sub = builtin( engine=self, clausedb=db, args=call_args, tdb=tdb, anc=anc+[call_key], level=level)

# see http://www.deransart.fr/prolog/bips.html

def _builtin_true_0(*args, **kwdargs) :
    return [()]

def _builtin_fail_0(*args, **kwdargs) :
    return []
    
def _builtin_call_1(engine, args, tdb, clausedb, level, **kwdargs) :
    raise NotImplementedError('call/1')
    query = tdb[args[0]]
    sub = engine.query( clausedb, tdb[args[0]], level=level+1 )
    
    result = []
    for res in sub :
        with tdb :
            for a,b in zip(res, query.args) :
                tdb.unify(a,b)
            print (tdb)
            result.append( [ tdb[arg] for arg in args ] )
    return result
    
    
# cut/0   => not supported
# if-then/2 if-then/3 => not supported
# catch/3 and throw/1 => not supported

# var/1
# atom/1
# integer/1
# float/1
# atomic/1 (atom, integer, float)
# compound/1 (not atomic, not variable)
# nonvar/1
# number/1 (integer or float)

# functor/3 (Term, Name, Arity)
# arg/3
# =../2
# copy_term/2
# clause/2
# current_predicate/1

# Clause Creation and Destruction => not supported

# findall/3
# bagof/3
# setof/3

# Input and Output => not supported

# Logic an Control
# \+/1
# once/1  =>  not supported
# repeat  =>  not supported?

# Atom Processing
# atom_length/2
# atom_concat/2
# sub_atom/5
# atom_chars/2
# atom_codes/2
# char_code/2
# number_chars/2
# number_codes/2

# Evaluable functors
# +/2 -/2 */2 //2 /2 rem/2 mod/2 -/1 abs/1 sign/1 float_integer_part/1 float_fractional_part/1 float/1 floor/1 truncate/1 round/1 ceiling/1
# **/2 sin/1 cos/1 atan/1 exp/1 log/1 sqrt/1 >>/2 <</2 /\/2 \/2 \/1

def _builtin_eq_2(args, tdb, **kwdargs) :
    with tdb :
        tdb.unify(*args)
        return [ [ tdb[arg] for arg in args ] ]
    return []
    
def _builtin_noteq_2(args, tdb, **kwdargs) :
    with tdb :
        tdb.unify(*args)
        return [ ]
    return [[ tdb[arg] for arg in args ]]

def _builtin_same_2(args, tdb, **kwdargs) :
    A,B = [ tdb[x] for x in args ]
    if A == B :
        return [ [A,B] ]
    else :
        return []

def _builtin_notsame_2(args, tdb, **kwdargs) :
    A,B = [ tdb[x] for x in args ]
    if A == B :
        return []
    else :
        return [ [A,B] ]
        
def _builtin_is_2(args, tdb, **kwdargs) :
    lhs, rhs = args
    # Left can be variable or constant
    # Evaluate right hand side and unify
    raise NotImplementedError('is/2')

def _builtin_eq_arithm_2(args, tdb, **kwdargs) :
    lhs, rhs = args
    # Left can be variable or constant
    # Evaluate right hand side and unify
    raise NotImplementedError('=:=/2')


class PrologInstantiationError(Exception) : pass

class PrologTypeError(Exception) : pass

def computeFunction(func, args) :
    if func == '+' :
        return Constant(args[0].value + args[1].value)
    elif func == '-' :
        return Constant(args[0].value - args[1].value)
    elif func == '*' :
        return Constant(args[0].value * args[1].value)
        
        
    
    
    return None

def compute( value ) :
    if value.isVar() :
        raise PrologInstantiationError(value)
    elif value.isConstant() :
        if type(value.value) == str :
            raise PrologTypeError('number', value)
        else :
            return value
    else :
        args = [ compute(arg) for arg in value.args ]
        return computeFunction( value.functor, args )

def PrologEngine(*args, **kwdargs) :
    
    engine = Engine(*args, **kwdargs)
    
    addPrologBuiltins(engine)
    
    return engine    

def addPrologBuiltins(engine) :
    engine.addBuiltIn('true/0', _builtin_true_0)
    engine.addBuiltIn('fail/0', _builtin_fail_0)
    engine.addBuiltIn('call/1', _builtin_call_1)
    
    engine.addBuiltIn('=/2', _builtin_eq_2)
    engine.addBuiltIn('\=/2', _builtin_noteq_2)
    engine.addBuiltIn('==', _builtin_same_2)
    engine.addBuiltIn('\==', _builtin_notsame_2)
    
    engine.addBuiltIn('is', _builtin_is_2)
    engine.addBuiltIn('=:=', _builtin_eq_arithm_2)


