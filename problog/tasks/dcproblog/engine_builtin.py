import itertools

from problog.engine import _ReplaceVar, substitute_call_args
from problog.engine_builtin import check_mode, _builtin_possible
from problog.engine_unify import UnifyError, unify_value
from problog.engine_stack import NODE_TRUE, NODE_FALSE
from problog.logic import Term, Constant, Var, term2list, list2term, _arithmetic_functions, unquote, is_ground

from .formula import LogicFormulaHAL
from .logic import SymbolicConstant, ValueDimConstant, ValueExpr, DensityConstant


def _builtin_is(a, b, engine=None, **kwdargs):
    check_mode((a, b), ['*g'], functor='is', **kwdargs)
    try:
        b_values = evaluate_arithemtics(b, engine=engine, **kwdargs)
        results =  []
        for b_value in b_values:
            if b_value[0] is None:
                continue
            else:
                if isinstance(b_value[0],(int,float)):
                    constant_val = Constant(b_value[0])
                else:
                    constant_val = b_value[0]
                results.append(((constant_val, b), b_value[1]))
        return results
    except UnifyError:
        return []

def _builtin_lt(arg1, arg2, engine=None, target=None, **kwdargs):
    check_mode((arg1, arg2), ['gg'], functor='<', **kwdargs)
    functor = "<"
    ab_values = make_comparison_args(arg1, arg2, engine=engine, target=target, **kwdargs)
    result = make_comparison(functor, ab_values, engine=engine, target=target, **kwdargs)
    return result
def _builtin_gt(arg1, arg2, engine=None, target=None, **kwdargs):
    check_mode((arg1, arg2), ['gg'], functor='>', **kwdargs)
    functor = ">"
    ab_values = make_comparison_args(arg1, arg2, engine=engine, target=target, **kwdargs)
    result = make_comparison(functor, ab_values, engine=engine, target=target, **kwdargs)
    return result
def _builtin_le(arg1, arg2, engine=None, target=None, **kwdargs):
    check_mode((arg1, arg2), ['gg'], functor='=<', **kwdargs)
    functor = "<="
    ab_values = make_comparison_args(arg1, arg2, engine=engine, target=target, **kwdargs)
    result = make_comparison(functor, ab_values, engine=engine, target=target, **kwdargs)
    return result
def _builtin_ge(arg1, arg2, engine=None, target=None, **kwdargs):
    check_mode((arg1, arg2), ['gg'], functor='>=', **kwdargs)
    functor = ">="
    ab_values = make_comparison_args(arg1, arg2, engine=engine, target=target, **kwdargs)
    result = make_comparison(functor, ab_values, engine=engine, target=target, **kwdargs)
    return result

def _builtin_observation(term, observation, engine=None, target=None, database=None, **kwdargs):
    check_mode((term, observation), ['gg'], functor='observation_builtin', target=target, **kwdargs)
    functor="observation"

    target, d_nodes = engine._ground(database, Term('~', term, Var('Distribution')), target)
    observation_node = 0
    for d_node in d_nodes:
        distribution = get_distribution(d_node, engine=engine, database=database, target=target, **kwdargs)
        identifier = "observation_of({})".format(target.get_density_name(term,distribution[1]))
        arg1 = target.create_ast_representation(distribution[0])
        arg2 = target.create_ast_representation(observation)
        cvariables = set()
        for a in (arg1,arg2):
            cvariables = cvariables.union(a.cvariables)
        probability = SymbolicConstant(functor, args=(arg1,arg2), cvariables=cvariables)

        o_node = target.add_atom(identifier, probability)
        if distribution[1] is None:
            o_node = None
        elif distribution[1]:
            o_node = target.add_and((o_node,distribution[1]))


        if not observation_node:
            observation_node = o_node
        else:
            observation_node = target.add_or((observation_node,o_node))

    result =  [((term,observation), observation_node)]
    return result



# def _builtin_density(term, args=(), target=None, engine=None, callback=None, transform=None, **kwdargs):
#     check_mode( (term,), ['c'], functor='density_builtin')
#     actions = []
#     try:
#         node_ids = target.density_nodes[term]
#     except:
#         raise ValueError("Cannot query density of discrete random variable ({}).".format(term))
#     target.density_queries[term] = set()
#     for nid in node_ids:
#         if nid in target.density_node_body:
#             body_node = target.density_node_body[nid]
#         else:
#             body_node = target.TRUE
#         density_name =  target.get_density_name(term, nid)
#         density = DensityConstant(density_name)
#         target.add_name(density, body_node, target.LABEL_QUERY)
#         target.density_queries[term].add(density)
#     actions += callback.notifyComplete()
#     return False, actions
#
# def _builtin_free(free_variable, args=(), target=None, engine=None, callback=None, transform=None, **kwdargs):
#     check_mode( (free_variable,), ['c'], functor='free')
#     actions = []
#     target.free_variables.add(free_variable)
#     actions += callback.notifyResult((free_variable,), is_last=False)
#     actions += callback.notifyComplete()
#     return True, actions
#
# def _builtin_free_list(free_variables, args=(), target=None, engine=None, callback=None, transform=None, **kwdargs):
#     check_mode( (free_variables,), ['l'], functor='free_list')
#     free_variables = term2list(free_variables)
#     actions = []
#     for v in free_variables:
#         target.free_variables.add(v)
#     actions += callback.notifyResult((free_variables,), is_last=False)
#     actions += callback.notifyComplete()
#     return True, actions






def get_value_type(value):
    if isinstance(value, int):
        return "point"
    else:
        return "vector"

def create_value(value, value_functor, value_args, value_name, value_type):
    if value_type=="point":
        dimensions = 1
        return ValueExpr(value_functor, value_args, value_name, dimensions)
    else:
        value_list  = term2list(value)
        dimensions = len(value_list)
        return ValueExpr(value_functor, value_args, value_name, dimensions)

def get_value_terms(value, value_type):
    value_terms = []
    for dv in value.dimension_values:
        value_terms.append(dv)
    if value_type=="point":
        value_terms = value_terms[0]
    else:
        value_terms = [vt for vt in value_terms]
        value_terms = list2term(value_terms)
    return value_terms


def make_comparison(functor, ab_values, engine=None, target=None, **kwdargs):
    result = []
    for args in ab_values:
        if isinstance(args[0][0], (int,float)) and isinstance(args[1][0], (int,float)):
            test = None
            a_value = args[0][0]
            b_value = args[1][0]
            if functor=="<":
                test = a_value<b_value
            elif functor==">":
                test = a_value>b_value
            elif functor=="<=":
                test = a_value<=b_value
            elif functor==">=":
                test = a_value>=b_value
            else:
                #TODO add other cases
                assert False
            if test:
                result.append(((a_value,b_value),NODE_TRUE))
            else:
                result.append(((a_value,b_value),NODE_FALSE))
        elif args[0] is None or args[1] is None:
            result.append((NODE_FALSE,0))
        else:
            arg1 = target.create_ast_representation(args[0][0])
            arg2 = target.create_ast_representation(args[1][0])
            body_node1 = args[0][1]
            body_node2 = args[1][1]
            if body_node1 and body_node2:
                body_node = target.add_and((body_node1,body_node2))
            elif body_node1:
                body_node = body_node1
            else:
                body_node = body_node2
            sym_args = (arg1,arg2)
            cvariables = set()
            for a in sym_args:
                cvariables = cvariables.union(a.cvariables)
            symbolic_condition = SymbolicConstant(functor, args=sym_args, cvariables=cvariables)

            hashed_symbolic = hash(str(symbolic_condition))
            con_node = target.add_atom(identifier=hashed_symbolic, probability=symbolic_condition, source=None)
            if body_node:
                pass_node = target.add_and((body_node, con_node))
            else:
                pass_node = con_node
            result.append((sym_args,pass_node))
    return result

def make_comparison_args(arg1, arg2, engine=None, **kwdargs):
    a_values = evaluate_arithemtics(arg1, engine=engine, **kwdargs)
    b_values = evaluate_arithemtics(arg2, engine=engine, **kwdargs)
    ab_values = list(itertools.product(a_values, b_values))
    return ab_values


def compute_function(term, database=None, target=None, engine=None, **kwdargs):
    """
    this function was originally in problog.logic.py
    """
    func = term.functor
    args = term.args

    function = _arithmetic_functions.get((unquote(func), len(args)))
    if function is None:
        function = extra_functions.get((unquote(func), len(args)))
        if function is None:
            raise ArithmeticError("Unknown function '%s'/%s" % (func, len(args)))
    try:
        value_lists = []
        for arg in args:
            value_lists.append(evaluate_arithemtics(arg, database=database, target=target, engine=engine, **kwdargs))
        args_list = list(itertools.product(*value_lists))
        coinjoined_args_list = []
        for args in args_list:
            cargs = ()
            body_node = 0
            for a, n in args:
                cargs += (a,)
                if n and not body_node:
                    body_node = n
                elif n:
                    body_node = target.add_and(n, body_node)
            coinjoined_args_list.append((cargs, body_node))

        # if None in :
        #     return [None]
        # else:
        result = []
        for cargs in coinjoined_args_list:
            result.append((function(*cargs[0]), cargs[1]))
        return result
    except ValueError as err:
        raise ArithmeticError(err.message)
    except ZeroDivisionError:
        raise ArithmeticError("Division by zero.")

#FIXME
def create_value2(value_functor, value_args, value_name, value_type):
    if value_type=="point":
        dimensions = 1
        return ValueExpr(value_functor, value_args, value_name, dimensions)


def get_distribution(distribution_node, target=None, engine=None, callback=None, transform=None, **kwdargs):
    value_type="point"

    rv = distribution_node[0][0]
    distribution = distribution_node[0][1]
    node_id = distribution_node[1]

    value_name =  target.get_density_name(rv, node_id)
    if value_name in target.density_values:
        value = target.density_values[value_name]
        # ?is this doing anything here?
    else:
        value_functor = distribution.functor
        value_args = [target.create_ast_representation(a) for a in distribution.args]

        value = create_value2(value_functor, value_args, value_name, value_type)
        target.density_values[value_name] = value
    value_terms = get_value_terms(value, value_type)


    result = (value_terms, node_id)

    return result


def evaluate_arithemtics(term_expression , engine=None, database=None, target=None, **kwdargs):
    b = term_expression
    b_values = []
    if isinstance(b, Constant):
        b_values.append((b.functor,0))
    elif isinstance(term_expression, SymbolicConstant):
        b_values.append((term_expression,0))
    elif isinstance(b, Term) and (unquote(b.functor), b.arity) in _arithmetic_functions:
        b_values = compute_function(b, engine=engine, database=database, target=target, **kwdargs)
    else:
        term = Term("~", b, Var("Distribution"))
        assert is_ground(b)
        term = term.apply(_ReplaceVar())  # replace Var(_) by integers
        target, results = engine._ground(database, term, target,  subcall=True)

        for r in results:
            b_values.append(get_distribution(r, engine=engine, database=database, target=target, **kwdargs))
    return b_values
