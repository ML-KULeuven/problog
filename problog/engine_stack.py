"""
problog.engine_stack - Stack-based implementation of grounding engine
---------------------------------------------------------------------

Default implementation of the ProbLog grounding engine.

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

import sys

from problog.context import State, Context, FixedContext, get_state
from problog.engine_messages import MessageFIFO, MessageOrderD, MessageOrderDrc
from .engine import ClauseDBEngine, substitute_simple
from .engine import NonGroundProbabilisticClause
from .engine import is_variable
from .engine_builtin import add_standard_builtins
from .eval_nodes import *


class InvalidEngineState(Exception):
    pass


# Define mapping from strings to the EvalNode classes, to map node_type strings to factory functions
node_types = {'fact': EvalFact,
              'conj': EvalAnd,
              'disj': EvalOr,
              'neg': EvalNot,
              'define': EvalDefine,
              'call': EvalCall,
              'clause': EvalClause,
              'choice': EvalChoice,
              'builtin': EvalBuiltIn,
              'extern': EvalExtern}


class StackBasedEngine(ClauseDBEngine):
    def __init__(self, label_all=False, **kwdargs):
        ClauseDBEngine.__init__(self, **kwdargs)

        self.cycle_root = None
        self.pointer = 0
        self.stack_size = 128
        self.stack = [None] * self.stack_size

        self.stats = [0] * 10

        self.debug = False
        self.trace = False

        self.label_all = label_all

        self.debugger = kwdargs.get('debugger')

        self.unbuffered = kwdargs.get('unbuffered')
        self.rc_first = kwdargs.get('rc_first', False)

        self.full_trace = kwdargs.get('full_trace')

        self.ignoring = set()

    def eval(self, node_id, **kwdargs):
        # print (kwdargs.get('parent'))
        database = kwdargs['database']

        # Skip not included or excluded nodes, or if parent is ignoring new results
        if self.should_skip_node(node_id, **kwdargs):
            return self.skip(node_id, **kwdargs)

        if node_id < 0:
            # Builtin node
            node = self.get_builtin(node_id)
            exec_func = EvalBuiltIn
        else:
            node = database.get_node(node_id)
            node_type = type(node).__name__
            exec_func = node_types.get(node_type)

            if exec_func is None:
                if self.unknown == self.UNKNOWN_FAIL:
                    return self.skip(node_id, **kwdargs)
                else:
                    raise UnknownClauseInternal()

        return exec_func.eval(engine=self, node_id=node_id, node=node, **kwdargs)

    def should_skip_node(self, node_id, **kwdargs):
        include_ids = kwdargs.get('include')
        exclude_ids = kwdargs.get('exclude')
        # If we are only looking at certain 'included' ids, check if it is included
        # OR If we are excluding certain ids, check if it is in the excluded id list.
        # OR The parent node is ignoring new results, so there is no point in generating them.
        return include_ids is not None and node_id not in include_ids \
               or exclude_ids is not None and node_id in exclude_ids \
               or kwdargs.get('parent') in self.ignoring

    def skip(self, node_id, **kwdargs):
        return [complete(kwdargs['parent'], kwdargs.get('identifier'))]

    def load_builtins(self):
        add_built_ins(self)

    def add_simple_builtin(self, predicate, arity, function):
        return self.add_builtin(predicate, arity, SimpleBuiltIn(function))

    def grow_stack(self):
        self.stack += [None] * self.stack_size
        self.stack_size *= 2

    def reset_stack(self):
        self.stack_size = 128
        self.stack = [None] * self.stack_size

    def add_record(self, record):
        if self.pointer >= self.stack_size:
            self.grow_stack()
        self.stack[self.pointer] = record
        self.pointer += 1

    def notify_cycle(self, childnode):
        # Optimization: we can usually stop when we reach a node on_cycle.
        #   However, when we swap the cycle root we need to also notify the old cycle root
        #    up to the new cycle root.
        assert self.cycle_root is not None
        root = self.cycle_root.pointer
        # childnode = self.stack[child]
        current = childnode.parent
        actions = []
        while current != root:
            if current is None:
                raise IndirectCallCycleError(
                    location=childnode.database.lineno(childnode.node.location))
            exec_node = self.stack[current]
            if exec_node.on_cycle:
                break
            new_actions = exec_node.createCycle()
            actions += new_actions
            current = exec_node.parent
        return actions

    def check_cycle(self, child, parent):
        current = child
        while current > parent:
            exec_node = self.stack[current]
            if exec_node.on_cycle:
                break
            if isinstance(exec_node, EvalNot):
                raise NegativeCycle(location=exec_node.database.lineno(exec_node.node.location))
            current = exec_node.parent

    def _transform_act(self, action):
        if action[0] in 'rc':
            return action
        else:
            return action[:2] + (action[3]['parent'], action[3]['context'], action[3]['identifier'])

    def init_message_stack(self):
        if self.unbuffered:
            if self.rc_first:
                return MessageOrderDrc(self)
            else:
                return MessageOrderD(self)
        else:
            return MessageFIFO(self)

    def in_cycle(self, pointer):
        """Check whether the node at the given pointer is inside a cycle.

        :param pointer:
        :return:
        """
        if pointer is None:
            return False
        elif self.cycle_root is None:
            return False
        elif pointer == self.cycle_root.pointer:
            return True
        else:
            node = self.stack[pointer]
            res = node.on_cycle or self.in_cycle(node.parent)
            return res

    def find_cycle(self, child, parent, force=False):
        root_encountered = None
        cycle = []
        while child is not None:
            cycle.append(child)
            childnode = self.stack[child]
            if hasattr(childnode, 'siblings'):
                for s in childnode.siblings:
                    cycle_rest = self.find_cycle(s, parent, force=force)
                    if cycle_rest:
                        return cycle + cycle_rest
            child = childnode.parent
            if child == parent:
                return cycle
            if self.cycle_root is not None and child == self.cycle_root.pointer:
                root_encountered = len(cycle)
        # if force:
        #     return cycle
        # else:
        if root_encountered is not None:
            return cycle[:root_encountered]
        else:
            return None

    def notify_cycle(self, cycle):
        actions = []
        for current in cycle[1:]:
            exec_node = self.stack[current]
            actions += exec_node.createCycle()
        return actions
        #
        #
        # assert self.cycle_root is not None
        # root = self.cycle_root.pointer
        # # childnode = self.stack[child]
        # current = childnode.parent
        # actions = []
        # while current != root:
        #     if current is None:
        #         raise IndirectCallCycleError(
        #             location=childnode.database.lineno(childnode.node.location))
        #     exec_node = self.stack[current]
        #     if exec_node.on_cycle:
        #         break
        #     new_actions = exec_node.createCycle()
        #     actions += new_actions
        #     current = exec_node.parent
        # return actions

    def is_real_cycle(self, child, parent):
        return bool(self.find_cycle(child, parent))

    def execute_init(self, node_id, target=None, database=None, is_root=None, **kwargs):

        # Initialize the cache/table.
        # This is stored in the target ground program because
        # node ids are only valid in that context.
        if not hasattr(target, '_cache'):
            target._cache = DefineCache(database.dont_cache)

        # Retrieve the list of actions needed to evaluate the top-level node.
        # parent = kwdargs.get('parent')
        # kwdargs['parent'] = parent

        initial_actions = self.eval(node_id, parent=None, database=database, target=target,
                                    is_root=is_root, **kwargs)

        return initial_actions

    def execute(self, node_id, target=None, database=None, subcall=False,
                is_root=False, name=None, **kwdargs):
        """
        Execute the given node.
        :param node_id: pointer of the node in the database
        :param subcall: indicates whether this is a toplevel call or a subcall
        :param target: target datastructure for storing the ground program
        :param database: database containing the logic program to ground
        :param kwdargs: additional arguments
        :return: results of the execution
        """
        # Find out debugging mode.
        self.trace = kwdargs.get('trace')
        self.debug = kwdargs.get('debug') or self.trace
        debugger = self.debugger

        # Initialize the cache/table.
        # This is stored in the target ground program because
        # node ids are only valid in that context.
        if not hasattr(target, '_cache'):
            target._cache = DefineCache(database.dont_cache)

        # Retrieve the list of actions needed to evaluate the top-level node.
        # parent = kwdargs.get('parent')
        # kwdargs['parent'] = parent

        initial_actions = self.eval(node_id, parent=None, database=database, target=target,
                                    is_root=is_root, **kwdargs)

        # Initialize the action stack.
        actions = self.init_message_stack()
        actions += reversed(initial_actions)
        solutions = []

        # Main loop: process actions until there are no more.
        while actions:
            if self.full_trace:
                self.printStack()
                print(actions)
            # Pop the next action.
            # An action consists of 4 parts:
            #   - act: the type of action (r, c, e)
            #   - obj: the pointer on which to call the action
            #   - args: the arguments of the action
            #   - context: the execution context

            if self.cycle_root is not None and actions.cycle_exhausted():
                if self.full_trace:
                    print('CLOSING CYCLE')
                    sys.stdin.readline()
                # for message in actions:   # TODO cache
                #     parent = actions._msg_parent(message)
                #     print (parent, self.in_cycle(parent))
                next_actions = self.cycle_root.closeCycle(True)
                actions += reversed(next_actions)
            else:
                act, obj, args, context = actions.pop()
                if debugger:
                    debugger.process_message(act, obj, args, context)

                if obj is None:
                    # We have reached the top-level.
                    if act == 'r':
                        # A new result is available
                        solutions.append((args[0], args[1]))
                        if name is not None:
                            negated, term, label = name
                            term_store = term.with_args(*args[0])
                            if negated:
                                target.add_name(-term_store, -args[1], label)
                            else:
                                target.add_name(term_store, args[1], label)

                        if args[3]:
                            # Last result received
                            if not subcall and self.pointer != 0:  # pragma: no cover
                                # ERROR: the engine stack should be empty.
                                self.printStack()
                                raise InvalidEngineState('Stack not empty at end of execution!')
                            if not subcall:
                                # Clean up the stack to save memory.
                                self.reset_stack()
                            return solutions
                    elif act == 'c':
                        # Indicates completion of the execution.
                        return solutions
                    else:
                        # ERROR: unknown message
                        raise InvalidEngineState('Unknown message!')
                else:
                    # We are not at the top-level.
                    if act == 'e':
                        # Never clean up in this case because 'obj' doesn't contain a pointer.
                        cleanup = False
                        # We need to execute another node.
                        # if self.cycle_root is not None and context['parent'] < self.cycle_root.pointer:
                        #     print ('Cycle exhausted indeed:', len(actions) + 1)
                        #     # There is an active cycle and we are about to execute a node
                        #     # outside that cycle.
                        #     # We first need to close the cycle.
                        #     next_actions = self.cycle_root.closeCycle(True) + [
                        #         (act, obj, args, context)]
                        # else:
                        try:
                            # Evaluate the next node.
                            # if exclude is not None and obj in exclude:
                            #     next_actions = self.skip(obj, **context)
                            #     obj = self.pointer
                            # elif include is not None and obj not in include:
                            #     next_actions = self.skip(obj, **context)
                            #     obj = self.pointer
                            # else:
                            next_actions = self.eval(obj, **context)
                            obj = self.pointer
                        except UnknownClauseInternal:
                            # An unknown clause was encountered.
                            # TODO why is this handled here?
                            call_origin = context.get('call_origin')
                            if call_origin is None:
                                sig = 'unknown'
                                raise UnknownClause(sig, location=None)
                            else:
                                loc = database.lineno(call_origin[1])
                                raise UnknownClause(call_origin[0], location=loc)
                    else:
                        # The message is 'r' or 'c'. This means 'obj' should be a valid pointer.
                        try:
                            # Retrieve the execution node from the stack.
                            exec_node = self.stack[obj]
                        except IndexError:  # pragma: no cover
                            self.printStack()
                            raise InvalidEngineState('Non-existing pointer: %s' % obj)
                        if exec_node is None:  # pragma: no cover
                            print(act, obj, args)
                            self.printStack()
                            raise InvalidEngineState('Invalid node at given pointer: %s' % obj)

                        if act == 'r':
                            # A new result was received.
                            cleanup, next_actions = exec_node.new_result(*args, **context)
                        elif act == 'c':
                            # A completion message was received.
                            cleanup, next_actions = exec_node.complete(*args, **context)
                        else:  # pragma: no cover
                            raise InvalidEngineState('Unknown message')

                    if not actions and not next_actions and self.cycle_root is not None:
                        if self.full_trace:
                            print('CLOSE CYCLE')
                            sys.stdin.readline()
                        # If there are no more actions and we have an active cycle, we should close the cycle.
                        next_actions = self.cycle_root.closeCycle(True)
                    # Update the list of actions.
                    actions += list(reversed(next_actions))

                    # Do debugging.
                    if self.debug:  # pragma: no cover
                        self.printStack(obj)
                        if act in 'rco':
                            print(obj, act, args)
                        print([(a, o, x) for a, o, x, t in actions[-10:]])
                        if self.trace:
                            a = sys.stdin.readline()
                            if a.strip() == 'gp':
                                print(target)
                            elif a.strip() == 'l':
                                self.trace = False
                                self.debug = False
                    if cleanup:
                        self.cleanup(obj)

        if subcall:
            call_origin = kwdargs.get('call_origin')
            if call_origin is not None:
                call_origin = database.lineno(call_origin[1])
            raise IndirectCallCycleError()
        else:
            # This should never happen.
            self.printStack()  # pragma: no cover
            print('Actions:', actions)
            print('Collected results:', solutions)  # pragma: no cover
            raise InvalidEngineState('Engine did not complete correctly!')  # pragma: no cover

    def cleanup(self, obj):
        """
        Remove the given node from the stack and lower the pointer.
        :param obj: pointer of the object to remove
        :type obj: int
        """

        self.ignoring.discard(obj)

        if self.cycle_root and self.cycle_root.pointer == obj:
            self.cycle_root = None
        self.stack[obj] = None
        while self.pointer > 0 and self.stack[self.pointer - 1] is None:
            self.pointer -= 1

    def call(self, query, database, target, transform=None, parent=None, context=None, **kwdargs):
        node_id = database.find(query)
        if node_id is None:
            node_id = database.get_builtin(query.signature)
            if node_id is None:
                raise UnknownClause(query.signature, database.lineno(query.location))

        return self.execute(node_id, database=database, target=target,
                            context=self.create_context(query.args, parent=context), **kwdargs)

    def call_intern(self, query, parent_context=None, **kwdargs):
        if query.is_negated():
            negated = True
            neg_func = query.functor
            query = -query
        elif query.functor in ('not', '\+') and query.arity == 1:
            negated = True
            neg_func = query.functor
            query = query.args[0]
        else:
            negated = False
        database = kwdargs.get('database')
        node_id = database.find(query)
        if node_id is None:
            node_id = database.get_builtin(query.signature)
            if node_id is None:
                raise UnknownClause(query.signature, database.lineno(query.location))

        call_args = range(0, len(query.args))
        call_term = query.with_args(*call_args)
        call_term.defnode = node_id
        call_term.child = node_id

        if negated:
            def func(result):
                return Term(neg_func, Term(call_term.functor, *result)),

            kwdargs['transform'].addFunction(func)

            return EvalNot.eval(engine=self, node_id=None, node=call_term,
                                context=self.create_context(query.args, parent=parent_context), **kwdargs)
        else:
            return EvalCall.eval(engine=self, node_id=None, node=call_term,
                                 context=self.create_context(query.args, parent=parent_context), **kwdargs)

    def printStack(self, pointer=None):  # pragma: no cover
        print('===========================')
        for i, x in enumerate(self.stack):
            if (pointer is None or pointer - 20 < i < pointer + 20) and x is not None:
                if i == pointer:
                    print('>>> %s: %s' % (i, x))
                elif self.cycle_root is not None and i == self.cycle_root.pointer:
                    print('ccc %s: %s' % (i, x))
                else:
                    print('    %s: %s' % (i, x))

    def propagate_evidence(self, db, target, functor, args, resultnode):
        if hasattr(target, 'lookup_evidence'):
            if resultnode in target.lookup_evidence:
                return target.lookup_evidence[resultnode]
            else:
                neg = target.negate(resultnode)
                if neg in target.lookup_evidence:
                    return target.negate(target.lookup_evidence[neg])
                else:
                    return resultnode
        else:
            return resultnode

    def context_min_var(self, context):
        min_var = 0
        for c in context:
            if is_variable(c):
                if c is not None and c < 0:
                    min_var = min(min_var, c)
            else:
                variables = [v for v in c.variables() if v is not None]
                if variables:
                    min_var = min(min_var, min(variables))
        return min_var

    def handle_nonground(self, location=None, database=None, node=None, **kwdargs):
        raise NonGroundProbabilisticClause(location=database.lineno(node.location))

    def create_context(self, content, define=None, parent=None, state=None):
        """Create a variable context."""

        con = Context(content)
        if state is not None:
            con.state = state
        elif not con.state:
            con.state = get_state(parent)
        if con.state is None:
            con.state = State()
        return con

    def _clone_context(self, context, parent=None, state=None):
        con = Context(context, state=state)
        if not con.state:
            con.state = get_state(parent)
        return con

    def _fix_context(self, context):
        return FixedContext(context)


class NestedDict(object):
    def __init__(self):
        self.__base = {}

    def __getitem__(self, key):
        p_key, s_key = key
        p_key = (p_key, len(s_key))
        s_key = list(s_key) + [get_state(s_key)]
        elem = self.__base[p_key]
        for s in s_key:
            elem = elem[s]
        return elem

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        p_key, s_key = key
        p_key = (p_key, len(s_key))
        s_key = list(s_key) + [get_state(s_key)]
        try:
            elem = self.__base[p_key]
            for s in s_key:
                elem = elem[s]
            return True
        except KeyError:
            return False

    def __setitem__(self, key, value):
        p_key, s_key = key
        p_key = (p_key, len(s_key))
        s_key = list(s_key) + [get_state(s_key)]
        if s_key:
            elem = self.__base.get(p_key)
            if elem is None:
                elem = {}
                self.__base[p_key] = elem
            for s in s_key[:-1]:
                elemN = elem.get(s)
                if elemN is None:
                    elemN = {}
                    elem[s] = elemN
                elem = elemN
            elem[s_key[-1]] = value
        else:
            self.__base[p_key] = value

    def __delitem__(self, key):
        p_key, s_key = key
        p_key = (p_key, len(s_key))
        s_key = list(s_key) + [get_state(s_key)]
        if s_key:
            elem = self.__base[p_key]
            elems = [(p_key, self.__base, elem)]
            for s in s_key[:-1]:
                elem_n = elem[s]
                elems.append((s, elem, elem_n))
                elem = elem_n
            del elem[s_key[-1]]  # Remove last element
            for s, e, ec in reversed(elems):
                if len(ec) == 0:
                    del e[s]
                else:
                    break
        else:
            del self.__base[p_key]

    def __str__(self):  # pragma: no cover
        return str(self.__base)


class VarReindex(object):

    def __init__(self):
        self.v = 0
        self.n = {}

    def __getitem__(self, var):
        if var is None:
            return var
        else:
            if var in self.n:
                return self.n[var]
            else:
                self.v -= 1
                self.n[var] = self.v
                return self.v
        # else:
        #     return var


class DefineCache(object):
    def __init__(self, dont_cache):
        self.__non_ground = NestedDict()
        self.__ground = NestedDict()
        self.__active = NestedDict()
        self.__dont_cache = dont_cache

    def reset(self):
        self.__non_ground = NestedDict()
        self.__ground = NestedDict()

    def _reindex_vars(self, goal):
        ri = VarReindex()
        return goal[0], [substitute_simple(g, ri) for g in goal[1]]

    def is_dont_cache(self, goal):
        return goal[0][:9] == '_nocache_' or (goal[0], len(goal[1])) in self.__dont_cache

    def activate(self, goal, node):
        self.__active[self._reindex_vars(goal)] = node

    def deactivate(self, goal):
        del self.__active[self._reindex_vars(goal)]

    def get_eval_node(self, goal):
        return self.__active.get(self._reindex_vars(goal))

    def __setitem__(self, goal, results):
        if self.is_dont_cache(goal):
            return
        # Results
        functor, args = goal
        if is_ground(*args):
            if results:
                # assert(len(results) == 1)
                res_key = next(iter(results.keys()))
                key = (functor, res_key)
                self.__ground[key] = results[res_key]
            else:
                key = (functor, args)
                self.__ground[key] = NODE_FALSE  # Goal failed
        else:
            goal = self._reindex_vars(goal)
            res_keys = list(results.keys())
            self.__non_ground[goal] = results
            all_ground = True
            for res_key in res_keys:
                key = (functor, res_key)
                all_ground &= is_ground(*res_key)
                if not all_ground:
                    break

            # TODO caching might be incorrect if program contains var(X) or nonvar(X) or ground(X).
            if all_ground:
                for res_key in res_keys:
                    key = (functor, res_key)
                    self.__ground[key] = results[res_key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, goal):
        functor, args = goal
        if is_ground(*args):
            return [(args, self.__ground[goal])]
        else:
            goal = self._reindex_vars(goal)
            # res_keys = self.__non_ground[goal]
            return self.__non_ground[goal].items()

    def __delitem__(self, goal):
        functor, args = goal
        if is_ground(*args):
            del self.__ground[goal]
        else:
            goal = self._reindex_vars(goal)
            del self.__non_ground[goal]

    def __contains__(self, goal):
        functor, args = goal
        if is_ground(*args):
            return goal in self.__ground
        else:
            goal = self._reindex_vars(goal)
            return goal in self.__non_ground

    def __str__(self):  # pragma: no cover
        return '%s\n%s' % (self.__non_ground, self.__ground)


class BooleanBuiltIn(object):
    """Simple builtin that consist of a check without unification. \
      (e.g. var(X), integer(X), ... )."""

    def __init__(self, base_function):
        self.base_function = base_function

    def __call__(self, *args, **kwdargs):
        callback = kwdargs.get('callback')
        if self.base_function(*args, **kwdargs):
            args = kwdargs['engine'].create_context(args, parent=kwdargs['context'])
            if kwdargs['target'].flag('keep_builtins'):
                call = kwdargs['call_origin'][0].split('/')[0]
                name = Term(call, *args)
                node = kwdargs['target'].add_atom(name, None, None, name=name, source='builtin')
                return True, callback.notifyResult(args, node, True)
            else:
                return True, callback.notifyResult(args, NODE_TRUE, True)
        else:
            return True, callback.notifyComplete()

    def __str__(self):  # pragma: no cover
        return str(self.base_function)


class SimpleProbabilisticBuiltIn(object):
    """Simple builtin that does cannot be involved in a cycle or require engine information and has 0 or more results."""

    def __init__(self, base_function):
        self.base_function = base_function

    def __call__(self, *args, **kwdargs):
        callback = kwdargs.get('callback')
        results = self.base_function(*args, **kwdargs)
        output = []
        if results:
            for i, result in enumerate(results):
                output += callback.notifyResult(kwdargs['engine'].create_context(result[0], parent=result[0]),
                                                result[1], i == len(results) - 1)
            return True, output
        else:
            return True, callback.notifyComplete()

    def __str__(self):  # pragma: no cover
        return str(self.base_function)


def add_built_ins(engine):
    add_standard_builtins(engine, BooleanBuiltIn, SimpleBuiltIn, SimpleProbabilisticBuiltIn)
