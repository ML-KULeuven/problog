"""
problog.engine_unify - Unification
----------------------------------

Implementation of unification for the grounding engine.

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
from .logic import is_variable


class UnifyError(Exception):
    """Unification error (used and handled internally)."""
    pass


class _SubstitutionWrapper(object):
    def __init__(self, subst):
        self.subst = subst

    def __getitem__(self, key):
        if key in self.subst:
            return self.subst[key]
        else:
            return key


def substitute_all(terms, subst):
    """

    :param terms:
    :param subst:
    :return:
    """
    subst = _SubstitutionWrapper(subst)
    result = []
    for term in terms:
        if is_variable(term):
            result.append(subst[term])
        else:
            result.append(term.apply(subst))
    return result


def instantiate(term, context):
    """Replace variables in Term by values based on context lookup table.

    :param term:
    :param context:
    :return:
    """
    if is_variable(term):
        return context[term]
    else:
        return term.apply(context)


class OccursCheck(GroundingError):
    def __init__(self):
        GroundingError.__init__(self, 'Infinite unification')


def unify_value(value1, value2, source_values):
    """
    Unify two values that exist in the same context.
    :param value1:
    :param value2:
    :param source_values:
    :return:
    """
    # Variables are negative numbers or None
    # Naive implementation (no occurs check)

    if is_variable(value1) and is_variable(value2):
        if value1 == value2:
            return value1
        elif value1 is None:
            return value2
        if value2 is None:  # second one is anonymous
            return value1
        else:
            # Two named variables: unify their source_values
            value = unify_value(source_values.get(value1), source_values.get(value2), source_values)
            if value is None:
                value = max(value1, value2)
            source_values[value1] = value
            source_values[value2] = value
            # Return the one of lowest rank: negative numbers, so max
            return value
    elif is_variable(value1):
        if value1 is None:
            return value2
        else:
            if value1 in value2.variables():
                raise OccursCheck()
            value = unify_value(source_values.get(value1), value2, source_values)
            source_values[value1] = value
            return value
    elif is_variable(value2):
        if value2 is None:
            return value1
        else:
            if value2 in value1.variables():
                raise OccursCheck()
            value = unify_value(source_values.get(value2), value1, source_values)
            source_values[value2] = value
            return value
    elif value1.signature == value2.signature:  # Assume Term
        return value1.with_args(*[unify_value(a1, a2, source_values)
                                  for a1, a2 in zip(value1.args, value2.args)])
    else:
        raise UnifyError()


class _ContextWrapper(object):
    def __init__(self, context):
        self.context = context
        self.numbers = {}
        self.translate = {None: None}
        self.num_count = 0

    def __getitem__(self, key):
        if key is None:
            return None
        elif key < 0:
            value = self.numbers.get(key)
            if value is None:
                self.num_count -= 1
                value = self.num_count
                self.numbers[key] = value
            self.translate[value] = key
        else:
            value = self.context[key]
            if value is None or type(value) == int:
                if value is not None:
                    key = value
                    okey = value
                else:
                    okey = None
                value = self.numbers.get(key)
                if value is None:
                    self.num_count -= 1
                    value = self.num_count
                    self.numbers[key] = value
                self.translate[value] = okey

        if not is_variable(value):
            value = value.apply(self)
        return value


def substitute_call_args(terms, context):
    """

    :param terms:
    :param context:
    :return:
    """
    result = []
    cw = _ContextWrapper(context)
    for term in terms:
        if term is None:
            cw.num_count -= 1
            result.append(cw.num_count)
        elif type(term) == int:
            v = cw[term]
            result.append(v)
        else:
            result.append(term.apply(cw))
    return result, cw.translate


def substitute_head_args(terms, context):
    """
    Extract the clause head arguments from the clause context.
    :param terms: head arguments. These can contain variables >0.
    :param context: clause context. These can contain variable <0.
    :return: input terms where variables are substituted by their values in the context
    """
    result = []
    for term in terms:
        result.append(substitute_simple(term, context))
    return result


def substitute_simple(term, context):
    """

    :param term:
    :param context:
    :return:
    """
    if term is None:
        return None
    elif type(term) == int:
        return context[term]
    else:
        return term.apply(context)


def unify_call_head(call_args, head_args, target_context):
    """
    Unify argument list from clause call and clause head.
    :param call_args: arguments of the call
    :param head_args: arguments of the head
    :param target_context: list of values of variables in the clause
    :raise UnifyError: unification failed
    """
    source_values = {}  # contains the values unified to the variables in the call arguments
    for call_arg, head_arg in zip(call_args, head_args):
        _unify_call_head_single(call_arg, head_arg, target_context, source_values)
    result = substitute_all(target_context, source_values)
    return result


def _unify_call_head_single(source_value, target_value, target_context, source_values):
    """
    Unify a call argument with a clause head argument.
    :param source_value: value occuring in clause call
    :type source_value: Term or variable. All variables are represented as negative numbers.
    :param target_value: value occuring in clause head
    :type target_value: Term or variable. All variables are represented as positive numbers \
    corresponding to positions
    in target_context.
    :param target_context: values of the variables in the current context. Output argument.
    :param source_values:
    :raise UnifyError: unification failed

    The values in the target_context contain only variables from the source context \
    (i.e. negative numbers) (or Terms).
    Initially values are set to None.
    """
    if is_variable(target_value):  # target_value is variable (integer >= 0)
        assert type(target_value) == int and target_value >= 0
        target_context[target_value] = \
            unify_value(source_value, target_context[target_value], source_values)
    else:  # target_value is a Term (which can still contain variables)
        if is_variable(source_value):  # source value is variable (integer < 0)
            if source_value is None:
                pass
            else:
                assert type(source_value) == int and source_value < 0
                # This means that *all* occurrences of source_value should unify with the same value.
                # We keep track of the value for each source_value, and unify the target_value with the current value.
                # Note that we unify two values in the same scope.
                source_values[source_value] = \
                    unify_value(source_values.get(source_value), target_value, source_values)
        else:  # source value is a Term (which can still contain variables)
            if target_value.signature == source_value.signature:
                # When signatures match, recursively unify the arguments.
                for s_arg, t_arg in zip(source_value.args, target_value.args):
                    _unify_call_head_single(s_arg, t_arg, target_context, source_values)
            else:
                raise UnifyError()


class _VarTranslateWrapper(object):
    def __init__(self, var_translate, min_var):
        self.base = var_translate
        self.min_var = min_var
#         self.min_var = min([min_var] + self.values())

    def __getitem__(self, item):
        if item in self.base:
            return self.base[item]
        else:
            self.min_var -= 1
            self.base[item] = self.min_var
            return self.min_var

    def values(self):
        return [x for x in self.base.values() if x is not None]


def unify_call_return(call_args, head_args, target_context, var_translate, min_var, mask=None):
    """
    Unify argument list from clause call and clause head.
    :param call_args: arguments of the call
    :param head_args: arguments of the head
    :param target_context: list of values of variables in the clause
    :param var_translate:
    :raise UnifyError: unification failed
    """
    if mask is None:
        mask = [True] * len(call_args)

    var_translate = _VarTranslateWrapper(var_translate, min_var)
    source_values = {}  # contains the values unified to the variables in the call arguments
    for call_arg, head_arg, mask_arg in zip(call_args, head_args, mask):
        if mask_arg:
            # Translate the variables in the source value (V3) to negative variables in current context (V2)
            call_arg = substitute_simple(call_arg, var_translate)
            _unify_call_return_single(call_arg, head_arg, target_context, source_values)
    result = substitute_all(target_context, source_values)
    return result


def _unify_call_return_single(source_value, target_value, target_context, source_values):
    """

    :param source_value: value from call result. Can contain variables < 0.
    :param target_value: value from call itself. Can contain variables >= 0.
    :param target_context:
    :param source_values:
    :return:
    """
    if is_variable(target_value):
        if target_value is None:
            pass
        elif target_value >= 0:
            # Get current value of target.
            current_value = target_context[target_value]
            # Three possibilities: None, negative variable (from V2), Term (with variables from V2)
            if current_value is None:
                target_context[target_value] = source_value
            elif type(current_value) == int:
                # Negative variable (from V2)
                source_values[current_value] = unify_value(
                    source_values.get(current_value), source_value, source_values)
            else:
                # Term (with variables from V2)
                target_context[target_value] = unify_value(
                    current_value, source_value, source_values)
        else:
            # Not possible. Target comes from static variables
            assert not target_value < 0
    else:
        # Target value is Term (which can contain variables)
        if is_variable(source_value):  # source value is variable (integer < 0)
            assert type(source_value) == int and source_value < 0
            source_values[source_value] = \
                unify_value(source_values.get(source_value), target_value, source_values)
        else:  # source value is a Term (which can still contain variables)
            if target_value.signature == source_value.signature:
                # When signatures match, recursively unify the arguments.
                for s_arg, t_arg in zip(source_value.args, target_value.args):
                    _unify_call_return_single(s_arg, t_arg, target_context, source_values)
            else:
                raise UnifyError()
