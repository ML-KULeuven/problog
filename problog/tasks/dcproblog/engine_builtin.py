from problog.engine_builtin import check_mode
from problog.engine_unify import UnifyError, unify_value
from problog.engine_stack import NODE_TRUE, NODE_FALSE
from problog.logic import Term, Constant, term2list, list2term

from .formula import LogicFormulaHAL
from .logic import SymbolicConstant, ValueDimConstant, ValueExpr, DensityConstant




def _builtin_density(term, args=(), target=None, engine=None, callback=None, transform=None, **kwdargs):
    check_mode( (term,), ['c'], functor='density_builtin')
    actions = []
    try:
        node_ids = target.density_nodes[term]
    except:
        raise ValueError("Cannot query density of discrete random variable ({}).".format(term))
    target.density_queries[term] = set()
    for nid in node_ids:
        if nid in target.density_node_body:
            body_node = target.density_node_body[nid]
        else:
            body_node = target.TRUE
        density_name =  target.get_density_name(term, nid)
        density = DensityConstant(density_name)
        target.add_name(density, body_node, target.LABEL_QUERY)
        target.density_queries[term].add(density)
    actions += callback.notifyComplete()
    return False, actions

def _builtin_free(free_variable, args=(), target=None, engine=None, callback=None, transform=None, **kwdargs):
    check_mode( (free_variable,), ['c'], functor='free')
    actions = []
    target.free_variables.add(free_variable)
    actions += callback.notifyResult((free_variable,), is_last=False)
    actions += callback.notifyComplete()
    return True, actions

def _builtin_free_list(free_variables, args=(), target=None, engine=None, callback=None, transform=None, **kwdargs):
    check_mode( (free_variables,), ['l'], functor='free_list')
    free_variables = term2list(free_variables)
    actions = []
    for v in free_variables:
        target.free_variables.add(v)
    actions += callback.notifyResult((free_variables,), is_last=False)
    actions += callback.notifyComplete()
    return True, actions

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

def _builtin_as(value, term, args=(), target=None, engine=None, callback=None, transform=None, **kwdargs):
    check_mode( (value,term), ['*g'], functor='as')
    #TODO make this function dependent on term
    value_type = get_value_type(value)
    try:
        node_ids = target.density_nodes[term]
    except:
        raise ValueError("The value of a discrete random variable ({}) is not defined.".format(term))
    actions = []
    for nid in node_ids:
        node = target.get_node(nid)
        value_name =  target.get_density_name(term, nid)
        if value_name in target.density_values:
            value = target.density_values[value_name]
            #?is this doing anything here?
        else:
            probability = node.probability
            value_functor = probability.functor
            value_args = [target.create_ast_representation(a) for a in probability.args]

            value = create_value(value, value_functor, value_args, value_name, value_type)
            target.density_values[value_name] = value

        value_terms = get_value_terms(value, value_type)
        term_pass = (Constant(value_terms), term)
        if nid in target.density_node_body:
            pass_node = target.density_node_body[nid]
        else:
            pass_node = 0
        actions += callback.notifyResult(term_pass, node=pass_node, is_last=False)
    actions += callback.notifyComplete()
    return True, actions

def conditionCallback(functor, arg1, arg2, **kwdargs):
    actions = []
    target = kwdargs["target"]
    callback = kwdargs["callback"]
    arg1 = target.create_ast_representation(arg1)
    arg2 = target.create_ast_representation(arg2)
    args = (arg1,arg2)
    cvariables = set()
    for a in args:
        cvariables = cvariables.union(a.cvariables)
    symbolic_condition = SymbolicConstant(functor, args=args, cvariables=cvariables)
    hashed_symbolic = hash(str(symbolic_condition))
    con_node = target.add_atom(identifier=hashed_symbolic, probability=symbolic_condition, source=None)
    args = kwdargs['engine'].create_context((arg1,arg2), parent=kwdargs['context'])
    actions += callback.notifyResult(args, node=con_node, is_last=True, parent=None)
    return True, actions

def booleanCallback(test, call, arg0, arg1, **kwdargs):
    callback = kwdargs['callback']
    if test:
        args = kwdargs['engine'].create_context((arg0,arg1), parent=kwdargs['context'])
        if kwdargs['target'].flag('keep_builtins'):
            call = functor
            name = Term(call, *args)
            node = kwdargs['target'].add_atom(name, None, None, name=name, source='builtin')
            return True, callback.notifyResult(args, node, True)
        else:
            return True, callback.notifyResult(args, NODE_TRUE, True)
    else:
        return True, callback.notifyComplete()

def _builtin_gt(arg1, arg2, engine=None, **kwdargs):
    """``A > B``
        A and B are ground
    """
    check_mode((arg1, arg2), ['gg'], functor='>', **kwdargs)
    a_value = arg1.compute_value(engine.functions)
    b_value = arg2.compute_value(engine.functions)
    if a_value is None or b_value is None:
        return False
    elif isinstance(a_value, (int, float)) and isinstance(b_value, (int, float)):
        return booleanCallback(a_value>b_value, '>', a_value, b_value, engine=engine, **kwdargs)
    else:
        return conditionCallback(">", a_value, b_value, engine=engine, **kwdargs)

def _builtin_lt(arg1, arg2, engine=None, **kwdargs):
    """``A < B``
        A and B are ground
    """
    check_mode((arg1, arg2), ['gg'], functor='<', **kwdargs)
    a_value = arg1.compute_value(engine.functions)
    b_value = arg2.compute_value(engine.functions)
    if a_value is None or b_value is None:
        return False
    elif isinstance(a_value, (int, float)) and isinstance(b_value, (int, float)):
        return booleanCallback(a_value<b_value, '<', a_value, b_value, engine=engine, **kwdargs)
    else:
        return conditionCallback("<", a_value, b_value, engine=engine, **kwdargs)

def _builtin_le(arg1, arg2, engine=None, **kwdargs):
    """``A =< B``
        A and B are ground
    """
    check_mode((arg1, arg2), ['gg'], functor='=<', **kwdargs)
    a_value = arg1.compute_value(engine.functions)
    b_value = arg2.compute_value(engine.functions)
    if a_value is None or b_value is None:
        return False
    elif isinstance(a_value, (int, float)) and isinstance(b_value, (int, float)):
        return booleanCallback(a_value<=b_value, '=<', a_value, b_value, engine=engine, **kwdargs)
    else:
        return conditionCallback("<=", a_value, b_value, engine=engine, **kwdargs)

def _builtin_ge(arg1, arg2, engine=None, **kwdargs):
    """``A >= B``
        A and B are ground
    """
    check_mode((arg1, arg2), ['gg'], functor='>=', **kwdargs)
    a_value = arg1.compute_value(engine.functions)
    b_value = arg2.compute_value(engine.functions)
    if a_value is None or b_value is None:
        return False
    elif isinstance(a_value, (int, float)) and isinstance(b_value, (int, float)):
        return booleanCallback(a_value>=b_value, '>=', a_value, b_value, engine=engine, **kwdargs)
    else:
        return conditionCallback(">=", a_value, b_value, engine=engine, **kwdargs)
#
#
# def _builtin_val_neq(a, b, engine=None, **k):
#     """``A =\= B``
#         A and B are ground
#     """
#     check_mode((a, b), ['gg'], functor='=\=', **k)
#     a_value = a.compute_value(engine.functions)
#     b_value = b.compute_value(engine.functions)
#     if a_value is None or b_value is None:
#         return False
#     else:
#         return a_value != b_value
#
#
# def _builtin_val_eq(a, b, engine=None, **k):
#     """``A =:= B``
#         A and B are ground
#     """
#     check_mode((a, b), ['gg'], functor='=:=', **k)
#     a_value = a.compute_value(engine.functions)
#     b_value = b.compute_value(engine.functions)
#     if a_value is None or b_value is None:
#         return False
#     else:
#         return a_value == b_value
#

def _builtin_observation(value, observation, engine=None, **kwdargs):
    check_mode((value, observation), ['gg'], functor='observation', **kwdargs)
    assert isinstance(value.functor, ValueDimConstant)

    v_value = value.compute_value(engine.functions)
    o_value = observation.compute_value(engine.functions)
    if v_value is None or o_value is None:
        return False
    else:
        return conditionCallback("observation", v_value, o_value, engine=engine, **kwdargs)




#
# def _builtin_is(a, b, engine=None, **k):
#     """``A is B``
#         B is ground
#
#         @param a:
#         @param b:
#         @param engine:
#         @param k:
#     """
#     check_mode((a, b), ['*g'], functor='is', **k)
#     try:
#         b_value = b.compute_value(engine.functions)
#         # print(type(b_value))
#         # print(engine.functions)
#         if b_value is None:
#             return []
#         else:
#             r = Constant(b_value)
#             unify_value(a, r, {})
#             return [(r, b)]
#     except UnifyError:
#         return []
