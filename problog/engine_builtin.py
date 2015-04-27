
from .logic import term2str, Term, Clause, Constant, term2list, list2term
from .program import PrologFile
from .core import GroundingError
from .engine_unify import unify_value, is_variable, UnifyError, substitute_simple

import os
import imp  # For load_external
import inspect  # For load_external


class CallModeError(GroundingError):
    """
    Represents an error in builtin argument types.
    """

    def __init__(self, functor, args, accepted=[], message=None, location=None):
        if functor:
            self.scope = '%s/%s' % (functor, len(args))
        else:
            self.scope = None
        self.received = ', '.join(map(self._show_arg, args))
        self.expected = [', '.join(map(self._show_mode, mode)) for mode in accepted]
        msg = 'Invalid argument types for call'
        if self.scope:
            msg += " to '%s'" % self.scope
        msg += ': arguments: (%s)' % self.received
        if accepted:
            msg += ', expected: (%s)' % ') or ('.join(self.expected)
        else:
            msg += ', expected: ' + message
        GroundingError.__init__(self, msg, location)

    def _show_arg(self, x):
        return term2str(x)

    def _show_mode(self, t):
        return mode_types[t][0]


class StructSort(object):
    """
    Comparator of terms based on structure.
    """

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


def is_ground(*terms):
    """Test whether a any of given terms contains a variable (recursively).
    :param terms:
    :return: True if none of the arguments contains any variables.
    """
    for term in terms:
        if is_variable(term):
            return False
        elif not term.isGround():
            return False
    return True


def is_var(term):
    return is_variable(term) or term.isVar()


def is_nonvar(term):
    return not is_var(term)


def is_term(term):
    return not is_var(term) and not is_constant(term)


def is_float_pos(term):
    return is_constant(term) and term.isFloat()


def is_float_neg(term):
    return is_term(term) and term.arity == 1 and term.functor == "'-'" and is_float_pos(term.args[0])


def is_float(term):
    return is_float_pos(term) or is_float_neg(term)


def is_integer_pos(term):
    return is_constant(term) and term.isInteger()


def is_integer_neg(term):
    return is_term(term) and term.arity == 1 and term.functor == "'-'" and is_integer_pos(term.args[0])


def is_integer(term):
    return is_integer_pos(term) or is_integer_neg(term)


def is_string(term):
    return is_constant(term) and term.isString()


def is_number(term):
    return is_float(term) and is_integer(term)


def is_constant(term):
    return not is_var(term) and term.isConstant()


def is_atom(term):
    return is_term(term) and term.arity == 0


def is_rational(term):
    return False


def is_dbref(term):
    return False


def is_compound(term):
    return is_term(term) and term.arity > 0


def is_list_maybe(term):
    """
    Check whether the term looks like a list (i.e. of the form '.'(_,_)).
    :param term:
    :return:
    """
    return is_compound(term) and term.functor == '.' and term.arity == 2


def is_list_nonempty(term):
    if is_list_maybe(term):
        tail = list_tail(term)
        return is_list_empty(tail) or is_var(tail)
    return False


def is_fixed_list(term):
    return is_list_empty(term) or is_fixed_list_nonempty(term)


def is_fixed_list_nonempty(term):
    if is_list_maybe(term):
        tail = list_tail(term)
        return is_list_empty(tail)
    return False


def is_list_empty(term):
    return is_atom(term) and term.functor == '[]'


def is_list(term):
    return is_list_empty(term) or is_list_nonempty(term)


def is_compare(term):
    return is_atom(term) and term.functor in ("'<'", "'='", "'>'")


mode_types = {
    'i': ('integer', is_integer),
    'I': ('positive_integer', is_integer_pos),
    'f': ('float', is_float),
    'v': ('var', is_var),
    'n': ('nonvar', is_nonvar),
    'l': ('list', is_list),
    'L': ('fixed_list', is_fixed_list),    # List of fixed length (i.e. tail is [])
    '*': ('any', lambda x: True),
    '<': ('compare', is_compare),         # < = >
    'g': ('ground', is_ground),
    'a': ('atom', is_atom),
    'c': ('callable', is_term)
}


def check_mode(args, accepted, functor=None, location=None, database=None, **kwdargs):
    """
    Checks the arguments against a list of accepted types.
    :param args: arguments to check
    :type args: tuple of Term
    :param accepted: list of accepted combination of types (see mode_types)
    :type accepted: list of basestring
    :param functor: functor of the call (used for error message)
    :param location: location of the call (used for error message)
    :param database: database (used for error message)
    :param kwdargs: additional arguments (not used)
    :return: the index of the first mode in accepted that matches the arguments
    :rtype: int
    """
    for i, mode in enumerate(accepted):
        correct = True
        for a, t in zip(args, mode):
            name, test = mode_types[t]
            if not test(a):
                correct = False
                break
        if correct:
            return i
    if database and location:
        location = database.lineno(location)
    else:
        location = None
    raise CallModeError(functor, args, accepted, location=location)


def list_elements(term):
    """
    Extract elements from a List term.
    Ignores the list tail.
    :param term: term representing a list
    :type term: Term
    :return: elements of the list
    :rtype: list of Term
    """
    elements = []
    tail = term
    while is_list_maybe(tail):
        elements.append(tail.args[0])
        tail = tail.args[1]
    return elements, tail


def list_tail(term):
    """
    Extract the tail of the list.
    :param term: Term representing a list
    :type term: Term
    :return: tail of the list
    :rtype: Term
    """
    tail = term
    while is_list_maybe(tail):
        tail = tail.args[1]
    return tail


def builtin_split_call(term, parts, database=None, location=None, **kwdargs):
    """
    Implements the '=..' builtin operator.
    :param term:
    :param parts:
    :param database:
    :param location:
    :param kwdargs:
    :return:
    """
    functor = '=..'
    # modes:
    #   <v> =.. list  => list has to be fixed length and non-empty
    #                       IF its length > 1 then first element should be an atom
    #   <n> =.. <list or var>
    #
    mode = check_mode((term, parts), ['vL', 'nv', 'nl'], functor=functor, **kwdargs)
    if mode == 0:
        elements, tail = list_elements(parts)
        if len(elements) == 0:
            raise CallModeError(functor, (term, parts), message='non-empty list for arg #2 if arg #1 is a variable',
                                location=database.lineno(location))
        elif len(elements) > 1 and not is_atom(elements[0]):
            raise CallModeError(functor, (term, parts), message='atom as first element in list if arg #1 is a variable',
                                location=database.lineno(location))
        elif len(elements) == 1:
            # Special case => term == parts[0]
            return [(elements[0], parts)]
        else:
            term_part = elements[0](*elements[1:])
            return [(term_part, parts)]
    else:
        part_list = (term.withArgs(), ) + term.args
        current = Term('[]')
        for t in reversed(part_list):
            current = Term('.', t, current)
        try:
            local_values = {}
            list_part = unify_value(current, parts, local_values)
            elements, tail = list_elements(list_part)
            term_new = elements[0](*elements[1:])
            term_part = unify_value(term, term_new, local_values)
            return [(term_part, list_part)]
        except UnifyError:
            return []


def builtin_arg(index, term, arguments, **kwdargs):
    mode = check_mode((index, term, arguments), ['In*'], functor='arg', **kwdargs)
    index_v = int(index) - 1
    if 0 <= index_v < len(term.args):
        try:
            arg = term.args[index_v]
            res = unify_value(arg, arguments, {})
            return [(index, term, res)]
        except UnifyError:
            pass
    return []


def builtin_functor(term, functor, arity, **kwdargs):
    mode = check_mode((term, functor, arity), ['vaI', 'n**'], functor='functor', **kwdargs)

    if mode == 0:
        kwdargs.get('callback').newResult(Term(functor, *((None,)*int(arity))), functor, arity)
    else:
        try:
            values = {}
            func_out = unify_value(functor, Term(term.functor), values)
            arity_out = unify_value(arity, Constant(term.arity), values)
            return [(term, func_out, arity_out)]
        except UnifyError:
            pass
    return []


def builtin_true(**kwdargs):
    """``true``"""
    return True


def builtin_fail(**kwdargs):
    """``fail``"""
    return False


def builtin_eq(arg1, arg2, **kwdargs):
    """``A = B``
        A and B not both variables
    """
    try:
        result = unify_value(arg1, arg2, {})
        return [(result, result)]
    except UnifyError:
        return []
    # except VariableUnification:
    #     raise VariableUnification(location = database.lineno(location))


def builtin_neq(arg1, arg2, **kwdargs):
    """``A \= B``
        A and B not both variables
    """
    try:
        result = unify_value(arg1, arg2, {})
        return False
    except UnifyError:
        return True


def builtin_notsame(arg1, arg2, **kwdargs):
    """``A \== B``"""
    return not arg1 == arg2


def builtin_same(arg1, arg2, **kwdargs):
    """``A == B``"""
    return arg1 == arg2


def builtin_gt(arg1, arg2, **kwdargs):
    """``A > B``
        A and B are ground
    """
    mode = check_mode((arg1, arg2), ['gg'], functor='>', **kwdargs)
    return arg1.value > arg2.value


def builtin_lt(arg1, arg2, **kwdargs):
    """``A > B``
        A and B are ground
    """
    mode = check_mode((arg1, arg2), ['gg'], functor='<', **kwdargs)
    return arg1.value < arg2.value


def builtin_le( A, B, **k ):
    """``A =< B``
        A and B are ground
    """
    mode = check_mode( (A,B), ['gg'], functor='=<', **k )
    return A.value <= B.value


def builtin_ge( A, B, **k ):
    """``A >= B``
        A and B are ground
    """
    mode = check_mode( (A,B), ['gg'], functor='>=', **k )
    return A.value >= B.value


def builtin_val_neq( A, B, **k ):
    """``A =\= B``
        A and B are ground
    """
    mode = check_mode( (A,B), ['gg'], functor='=\=', **k )
    return A.value != B.value


def builtin_val_eq( A, B, **k ):
    """``A =:= B``
        A and B are ground
    """
    mode = check_mode( (A,B), ['gg'], functor='=:=', **k )
    return A.value == B.value


def builtin_is( A, B, **k ):
    """``A is B``
        B is ground
    """
    mode = check_mode( (A,B), ['*g'], functor='is', **k )
    try:
        R = Constant(B.value)
        unify_value(A, R, {})
        return [(R,B)]
    except UnifyError:
        return []


def builtin_var( term, **k ):
    return is_var(term)


def builtin_atom( term, **k ):
    return is_atom(term)


def builtin_atomic( term, **k ):
    return is_atom(term) or is_number(term)


def builtin_compound( term, **k ):
    return is_compound(term)


def builtin_float( term, **k ):
    return is_float(term)


def builtin_integer( term, **k ):
    return is_integer(term)


def builtin_nonvar( term, **k ):
    return not is_var(term)


def builtin_number( term, **k ):
    return is_number(term)


def builtin_simple( term, **k ):
    return is_var(term) or is_atomic(term)


def builtin_callable( term, **k ):
    return is_term(term)


def builtin_rational( term, **k ):
    return is_rational(term)


def builtin_dbreference( term, **k ):
    return is_dbref(term)


def builtin_primitive( term, **k ):
    return is_atomic(term) or is_dbref(term)


def builtin_ground( term, **k ):
    return is_ground(term)


def builtin_is_list( term, **k ):
    return is_list(term)


def compare(a,b):
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


def struct_cmp( A, B ):
    # Note: structural comparison
    # 1) Var < Num < Str < Atom < Compound
    # 2) Var by address
    # 3) Number by value, if == between int and float => float is smaller (iso prolog: Float always < Integer )
    # 4) String alphabetical
    # 5) Atoms alphabetical
    # 6) Compound: arity / functor / arguments

    # 1) Variables are smallest
    if is_var(A):
        if is_var(B):
            # 2) Variable by address
            return compare(A,B)
        else:
            return -1
    elif is_var(B):
        return 1
    # assert( not is_var(A) and not is_var(B) )

    # 2) Numbers are second smallest
    if is_number(A):
        if is_number(B):
            # Just compare numbers on float value
            res = compare(float(A),float(B))
            if res == 0:
                # If the same, float is smaller.
                if is_float(A) and is_integer(B):
                    return -1
                elif is_float(B) and is_integer(A):
                    return 1
                else:
                    return 0
        else:
            return -1
    elif is_number(B):
        return 1

    # 3) Strings are third
    if is_string(A):
        if is_string(B):
            return compare(str(A),str(B))
        else:
            return -1
    elif is_string(B):
        return 1

    # 4) Atoms / terms come next
    # 4.1) By arity
    res = compare(A.arity,B.arity)
    if res != 0: return res

    # 4.2) By functor
    res = compare(A.functor,B.functor)
    if res != 0: return res

    # 4.3) By arguments (recursively)
    for a,b in zip(A.args,B.args):
        res = struct_cmp(a,b)
        if res != 0: return res

    return 0


def builtin_struct_lt(A, B, **k):
    return struct_cmp(A,B) < 0


def builtin_struct_le(A, B, **k):
    return struct_cmp(A,B) <= 0


def builtin_struct_gt(A, B, **k):
    return struct_cmp(A,B) > 0


def builtin_struct_ge(A, B, **k):
    return struct_cmp(A,B) >= 0


def builtin_compare(C, A, B, **k):
    mode = check_mode( (C,A,B), [ '<**', 'v**' ], functor='compare', **k)
    compares = "'>'","'='","'<'"
    c = struct_cmp(A,B)
    c_token = compares[1-c]

    if mode == 0: # Given compare
        if c_token == C.functor: return [ (C,A,B) ]
    else:  # Unknown compare
        return [ (Term(c_token), A, B ) ]

# numbervars(T,+N1,-Nn)    number the variables TBD?

def build_list(elements, tail):
    current = tail
    for el in reversed(elements):
        current = Term('.', el, current)
    return current


class UnknownExternal(GroundingError):
    """Undefined clause in call."""

    def __init__(self, signature, location):
        GroundingError.__init__(self, "Unknown external function '%s'" % signature, location)


def builtin_call_external(call, result, database=None, location=None, **k):
    from . import pypl
    mode = check_mode( (call,result), ['gv'], function='call_external', database=database, location=location, **k)

    func = k['engine'].get_external_call(call.functor)
    if func is None:
        raise UnknownExternal(call.functor, database.lineno(location))

    values = [pypl.pl2py(arg) for arg in call.args]
    computed_result = func(*values)

    return [(call, pypl.py2pl(computed_result))]


def builtin_length(L, N, **k):
    mode = check_mode( (L,N), [ 'LI', 'Lv', 'lI', 'vI' ], functor='length', **k)
    # Note that Prolog also accepts 'vv' and 'lv', but these are unbounded.
    # Note that lI is a subset of LI, but only first matching mode is returned.
    if mode == 0 or mode == 1:  # Given fixed list and maybe length
        elements, tail = list_elements(L)
        list_size = len(elements)
        try:
            N = unify_value(N, Constant(list_size), {})
            return [ ( L, N ) ]
        except UnifyError:
            return []
    else:    # Unbounded list or variable list and fixed length.
        if mode == 2:
            elements, tail = list_elements(L)
        else:
            elements, tail = [], L
        remain = int(N) - len(elements)
        if remain < 0:
            raise UnifyError()
        else:
            extra = [None] * remain
        newL = build_list( elements + extra, Term('[]'))
        return [ (newL, N)]

# def extract_vars(*args, **kwd):
#     counter = kwd.get('counter', defaultdict(int))
#     for arg in args:
#         if type(arg) == int:
#             counter[arg] += 1
#         elif isinstance(arg,Term):
#             extract_vars(*arg.args, counter=counter)
#         else:
#            raise VariableUnification()
#     return counter


def builtin_sort( L, S, **k ):
    # TODO doesn't work properly with variables e.g. gives sort([X,Y,Y],[_]) should be sort([X,Y,Y],[X,Y])
    mode = check_mode( (L,S), [ 'L*' ], functor='sort', **k )
    elements, tail = list_elements(L)
    # assert( is_list_empty(tail) )
    try:
        sorted_list = build_list(sorted(set(elements), key=StructSort), Term('[]'))
        S_out = unify_value(S, sorted_list, {})
        return [(L,S_out)]
    except UnifyError:
        return []


def builtin_between(low, high, value, **k):
    """
    Implements the between/3 builtin.
   :param low:
   :param high:
   :param value:
   :param k:
   :return:
    """
    mode = check_mode((low, high, value), ['iii', 'iiv'], functor='between', **k)
    low_v = int(low)
    high_v = int(high)
    if mode == 0:  # Check
        value_v = int(value)
        if low_v <= value_v <= high_v:
            return [(low, high, value)]
    else:  # Enumerate
        results = []
        for value_v in range(low_v, high_v+1):
            results.append((low, high, Constant(value_v)))
        return results


def builtin_succ(a, b, **kwdargs):
    """
    Implements the succ/2 builtin.
   :param a: input argument
   :param b: output argument
   :param kwdargs: additional arguments
   :return:
    """
    mode = check_mode((a, b), ['vI', 'Iv', 'II'], functor='succ', **kwdargs)
    if mode == 0:
        b_v = int(b)
        return [(Constant(b_v-1), b)]
    elif mode == 1:
        a_v = int(a)
        return [(a, Constant(a_v+1))]
    else:
        a_v = int(a)
        b_v = int(b)
        if b_v == a_v + 1:
            return [(a, b)]
    return []


def builtin_plus(a, b, c, **kwdargs):
    """
    Implements the plus/3 builtin.
   :param a: first argument
   :param b: second argument
   :param c: result argument
   :param kwdargs: additional arguments
   :return:
    """
    mode = check_mode((a, b, c), ['iii', 'iiv', 'ivi', 'vii'], functor='plus', **kwdargs)
    if mode == 0:
        a_v = int(a)
        b_v = int(b)
        c_v = int(c)
        if a_v + b_v == c_v:
            return [(a, b, c)]
    elif mode == 1:
        a_v = int(a)
        b_v = int(b)
        return [(a, b, Constant(a_v+b_v))]
    elif mode == 2:
        a_v = int(a)
        c_v = int(c)
        return [(a, Constant(c_v-a_v), c)]
    else:
        b_v = int(b)
        c_v = int(c)
        return [(Constant(c_v-b_v), b, c)]
    return []


def atom_to_filename(atom):
    """
    Translate an atom to a filename.
   :param atom: filename as atom
   :return: filename as string
   :rtype: basestring
    """
    atom = str(atom)
    if atom[0] == atom[-1] == "'":
        atom = atom[1:-1]
    return atom


class ConsultError(GroundingError):
    """Error during consult"""

    def __init__(self, message, location):
        GroundingError.__init__(self, message, location)


def builtin_consult_as_list(op1, op2, **kwdargs):
    """
    Implementation of consult/1 using list notation.
   :param op1: first element in the list
   :param op2: tail of the list
   :param kwdargs: additional arugments
   :return: True
    """
    # TODO make non-recursive
    check_mode((op1, op2), ['*L'], functor='consult', **kwdargs)
    builtin_consult(op1, **kwdargs)
    if is_list_nonempty(op2):
        builtin_consult_as_list(op2.args[0], op2.args[1], **kwdargs)
    return True


def builtin_consult(filename, database=None, **kwdargs):
    """
    Implementation of consult/1 builtin.
    A file will be loaded only once.
   :param filename: filename to load into the database
   :param database: database containing the current logic program.
   :param kwdargs: additional arguments
   :return: True
    """
    check_mode((filename,), 'a', functor='consult', **kwdargs)
    filename = os.path.join(database.source_root, atom_to_filename(filename))
    if not os.path.exists(filename):
        filename += '.pl'
    if not os.path.exists(filename):
        raise ConsultError(location=database.lineno(kwdargs.get('location')),
                           message="Consult: file not found '%s'" % filename)

    # Prevent loading the same file twice
    if filename not in database.source_files:
        database.source_files.append(filename)
        pl = PrologFile(filename)
        for clause in pl:
            database += clause
    return True


def builtin_load_external( arg, engine=None, database=None, location=None, **kwdargs ):
    check_mode( (arg,), 'a', functor='load_external' )
    # Load external (python) files that are referenced in the model
    externals = {}
    filename = os.path.join(database.source_root, atom_to_filename( arg ))
    if not os.path.exists(filename):
          raise ConsultError(location=database.lineno(location), message="Load external: file not found '%s'" % filename)
    try:
        with open(filename, 'r') as extfile:
            ext = imp.load_module('externals', extfile, filename, ('.py', 'U', 1))
            for func_name, func in inspect.getmembers(ext, inspect.isfunction):
                externals[func_name] = func
        engine.add_external_calls(externals)
    except ImportError:
        raise ConsultError(location=database.lineno(location), message="Error while loading external file '%s'" % filename)

    return True

def builtin_unknown( arg, engine=None, **kwdargs):
    check_mode( (arg,), 'a', functor='unknown')
    if arg.functor == 'fail':
        engine.unknown = engine.UNKNOWN_FAIL
    else:
        engine.unknown = engine.UNKNOWN_ERROR
    return True


def select_sublist(lst, target):
    """
    Enumerate all possible selection of elements from a list.
    This function is used to generate all solutions to findall/3.
    An element must be selected if it is TRUE in the target formula.
   :param lst: list to select elements from
   :type lst: list of tuple
   :param target: data structure containing truth value of nodes
   :type target: LogicFormula
   :return: generator of sublists
    """
    l = len(lst)

    # Generate an array that indicates the decision bit for each element in the list.
    # If an element is deterministically true, then no decision bit is needed.
    choice_bits = [None] * l
    x = 0
    for i in range(0, l):
        if lst[i][1] != target.TRUE:
            choice_bits[i] = x
            x += 1
    # We have 2^x distinct lists. Each can be represented as a number between 0 and n.
    n = (1 << x)

    while n >= 0:
        # Generate the list of positive values and node identifiers
        sublist = [lst[i] for i in range(0, l) if choice_bits[i] is None or n & 1 << choice_bits[i]]
        # Generate the list of negative node identifiers
        sublist_no = tuple([-lst[i][1] for i in range(0, l) if not (choice_bits[i] is None or n & 1 << choice_bits[i])])
        if sublist:
            terms, nodes = zip(*sublist)
        else:
            # Empty list.
            terms, nodes = (), ()
        yield terms, nodes + sublist_no + (0,)
        n -= 1


def builtin_findall(pattern, goal, result, database=None, target=None, engine=None, **kwdargs):
    """
    Implementation of findall/3 builtin.
   :param pattern: pattern to extract
   :type pattern: Term
   :param goal: goal to evaluate
   :type goal: Term
   :param result: list to store results
   :type result: Term
   :param database: database holding logic program
   :type database: ClauseDB
   :param target: logic formula in which to store the result
   :type target: LogicFormula
   :param engine: engine that is used for evaluation
   :type engine: ClauseDBEngine
   :param kwdargs: additional arguments from engine
   :return: list results (tuple of lists and node identifiers)
    """
    # Check the modes.
    mode = check_mode((pattern, goal, result,), ('*cv', '*cl'), database=database, **kwdargs)

    findall_head = Term(engine.get_non_cache_functor(), pattern)
    findall_clause = Clause( findall_head, goal)
    findall_db = database.extend()
    findall_db += findall_clause

    class _TranslateToNone(object):
        def __getitem__(self, item):
            return None
    findall_head = substitute_simple(findall_head, _TranslateToNone())

    results = engine.call(findall_head, subcall=True, database=findall_db, target=target, **kwdargs)
    results = [(res[0], n) for res, n in results]
    output = []
    for l, n in select_sublist(results, target):
        node = target.addAnd(n)
        if node is not None:
            res = build_list(l, Term('[]'))
            if mode == 0:  # var
                args = (pattern, goal, res)
                output.append((args, node))
            else:
                try:
                    res = unify_value(res, result, {})
                    args = (pattern, goal, res)
                    output.append((args, node))
                except UnifyError:
                    pass
    return output


class problog_export(object):

    database = None

    @classmethod
    def add_function(cls, name, in_args, out_args, function):
        if cls.database is not None:
            cls.database.add_extern(name, in_args+out_args, function)

    def __init__(self, *args, **kwdargs):
        # TODO check if arguments are in order: input first, output last
        self.input_arguments = [a[1:] for a in args if a[0] == '+']
        self.output_arguments = [a[1:] for a in args if a[0] == '-']

    def _convert_input(self, a, t):
        if t == 'str':
            return str(a)
        elif t == 'int':
            return int(a)
        elif t == 'float':
            return float(a)
        elif t == 'list':
            return term2list(a)
        else:
            raise ValueError("Unknown type specifier '%s'!" % t)

    def _extract_callmode(self):
        callmode = ''
        for t in self.input_arguments:
            if t == 'str':
                callmode += 'a'
            elif t == 'int':
                callmode += 'i'
            elif t == 'float':
                callmode += 'f'
            elif t == 'list':
                callmode += 'L'
            else:
                raise ValueError("Unknown type specifier '%s'!" % t)
        for t in self.output_arguments:
            callmode += 'v'
        return callmode

    def _convert_output(self, a, t):
        if t == 'str':
            return Term(a)
        elif t == 'int':
            return Constant(a)
        elif t == 'float':
            return Constant(a)
        elif t == 'list':
            return list2term(a)
        else:
            raise ValueError("Unknown type specifier '%s'!" % t)

    def _convert_inputs(self, args):
        return [self._convert_input(a, t) for a, t in zip(args, self.input_arguments)]

    def _convert_outputs(self, args):
        return [self._convert_output(a, t) for a, t in zip(args, self.output_arguments)]

    def __call__(self, function):
        def _wrapped_function(*args, **kwdargs):
            check_mode(args, [self._extract_callmode()], function.__name__, **kwdargs)
            # TODO check that output arguments are variables
            converted_args = self._convert_inputs(args)
            result = function(*converted_args)
            if len(self.output_arguments) == 1:
                result = [result]
            result = args[:len(self.input_arguments)] + tuple(self._convert_outputs(result))
            return [result]
        problog_export.add_function(function.__name__, len(self.input_arguments), len(self.output_arguments), _wrapped_function)
        return function


def builtin_use_module(filename, engine=None, database=None, location=None, **kwdargs ):
    filename = os.path.join(database.source_root, atom_to_filename(filename))
    try:
        load_external_module(database, filename)
    except IOError as err:
        raise ConsultError('Error while reading external library: %s' % str(err), database.lineno(location))
    return True


def load_external_module(database, filename):
    import imp
    problog_export.database = database
    with open(filename, 'r') as extfile:
        imp.load_module('externals', extfile, filename, ('.py', 'U', 1))
