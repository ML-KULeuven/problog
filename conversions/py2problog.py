#!/usr/bin/env python3
# encoding: utf-8
"""
py2problog.py

Trace a Python program to generate a probabilistic model that represents
the Python program. The generated model can be fed to ProbLog to perform
probabilistic inference.

Dependencies:
- python 3: https:python.org
- attrs: https://attrs.readthedocs.io

Created by Wannes Meert on 17-11-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""

# https://pymotw.com/2/sys/tracing.html#sys-tracing

import sys
import logging
import re
from collections import deque, Counter
import random
import hashlib
import inspect
import itertools
import functools
import attr

print("WARNING: py2problog.py is an experimental and incomplete library, use with caution.")

# Settings


@attr.s(slots=True)
class Settings:
    memoization = attr.ib(default=True)
    uniform_prob = attr.ib(default=True)
    nb_samples = attr.ib(default=200)
    graph_prob = attr.ib(default=False)
    use_trace = attr.ib(default=True)

settings = Settings()


# Global variables

root_nodes = list()
trace_stack = None
current_node = None
pf_cache = dict()
node_cache = None
memoization_cache = None
cnt_created_nodes = 0
cnt_probfact = 0
iteration = 0
is_running = False
exhaustive_search = False
deterministic_funcs = set()
stop_tracing = 0
probabilistic_funcs = set()

logger = logging.getLogger('be.kuleuven.be.dtai.problog')


# Probabilistic primitives

class ProbFactObject:
    def __init__(self, prob, name=None):
        global cnt_probfact
        if name is None:
            self.name = "pf_{}".format(cnt_probfact)
        else:
            self.name = name
        self.prob = prob
        cnt_probfact += 1
        self.nb_calls = 0

    def nb_values(self):
        return 2

    def get_value(self, idx):
        if idx == 0:
            return False
        elif idx == 1:
            return True
        return None

    def get_idx(self, value):
        if value is True or value == 1:
            return 1
        elif value is False or value == 0:
            return 0
        return None

    def __call__(self):
        if not settings.use_trace:
            process_pf_call(self)
        result = self.call()
        if not settings.use_trace:
            process_pf_return(result)
        return result

    def call(self):
        if settings.memoization and self in memoization_cache:
            return memoization_cache[self]
        if settings.uniform_prob:
            # With every iteration becomes more uniform
            # Initial bias towards actual program, final bias towards uniform
            # probabilities such that also unlikely paths are followed
            # This heuristic can be improved by looking at the FuncNode graph
            # statistics
            prob = self.prob + (0.5-self.prob)*iteration/settings.nb_samples
            # print("{}::{}".format(prob,self.name))
        else:
            prob = self.prob
        rvalue = True if random.random() < prob else False
        memoization_cache[self] = rvalue
        self.nb_calls += 1
        return rvalue

    def __str__(self):
        return "{}::{}".format(self.prob, self.name)


def ProbFact(prob=None, name=None):
    """Create a Probabilistic Fact.
    If an Probabilistic Fact with that name has already been defined before
    (or in a previous iteration) the same object is returned.

    :param prob: Probability 0<=prob<=1.0
    :param name: Name of probabilistic fact
    :returns: A ProbFactObject instance
    """
    global pf_cache
    global is_running
    if prob is None:
        prob = 0.5
    if exhaustive_search and is_running:
        print("ERROR: Exhaustive search only works if all probabilistic facts are defined before running the query")
        sys.exit(1)
    if name is None and is_running:
        print("ERROR: When defining a probabilistic fact while running, it should be given a name")
        # If no name is given it will be a different probfact in every
        # iteration instead of the same variable
        sys.exit(1)
    if name in pf_cache:
        logger.debug('Prob fact already exists: {}'.format(name))
        pf = pf_cache[name]
        return pf
    pf = ProbFactObject(prob, name)
    pf_cache[pf.name] = pf
    return pf


class ADObject:
    def __init__(self, probs, name=None):
        global cnt_probfact
        if name is None:
            self.name = "ad_{}".format(cnt_probfact)
        else:
            self.name = name
        self.probs = probs
        cnt_probfact += 1
        self.nb_calls = 0

    def nb_values(self):
        return len(self.probs)

    def get_value(self, idx):
        return self.probs[idx][1]

    def get_idx(self, value):
        idx = None
        for pr, v in self.probs:
            if v == value:
                return idx
        return idx

    def __call__(self):
        if not settings.use_trace:
            process_pf_call(self)
        result = self.call()
        if not settings.use_trace:
            process_pf_return(result)
        return result

    def call(self):
        if settings.memoization and self in memoization_cache:
            return memoization_cache[self]
        r = random.random()
        s = 0
        rvalue = None
        for pr, rv in self.probs:
            if settings.uniform_prob:
                prob = pr + (1/len(self.probs)-pr)*iteration/settings.nb_samples
                # print("{}::{}={}".format(prob,self.name,rvalue))
            else:
                prob = pr
            s += prob
            if r <= s:
                rvalue = rv
                break
        memoization_cache[self] = rvalue
        self.nb_calls += 1
        return rvalue

    def __str__(self):
        return "; ".join(["{}::{}={}".format(pr, self.name, rvalue) for pr, rvalue in self.probs])


def AD(probs, name=None):
    """Create an Annotated Disjunction.
    If an Annotated Disjunction with that name has already been defined before
    (or in a previous iteration) the same object is returned.

    :param probs: List of tuples (probability, return value)
        The probability is a number between 0.0 and 1.0 and the sum should not
        exceed 1.0. The return value can be any Python primitive (int, str,
        float, bool)
    :param name: Name of the annotated disjunction
    :returns: A ADObject instance
    """
    global pf_cache
    global is_running
    if exhaustive_search and is_running:
        print("ERROR: Exhaustive search only works if all annotated disjunctions are defined before running the query")
        sys.exit(1)
    if name is None and is_running:
        print("ERROR: When defining an annotated disjunction while running, it should be given a name")
        # If no name is given it will be a different AD in every iteration
        # instead of the same variable
        sys.exit(1)
    if name in pf_cache:
        logger.debug('Prob fact already exists: {}'.format(name))
        pf = pf_cache[name]
        return pf
    pf = ADObject(probs, name)
    pf_cache[pf.name] = pf
    return pf

dot_re = re.compile("[\[\]()= ,<>.':-]")


def clean_dot(name):
    # print('{} -> {}'.format(name, dot_re.sub("_", name)))
    return dot_re.sub("_", name)


class FuncNode:
    def __init__(self, pred, terms):
        global cnt_created_nodes
        global iteration
        cnt_created_nodes += 1
        self.pred = pred  # str
        self.terms = terms  # dict
        self.rvalue = None  # str/number
        self.bodies = dict()  # dict(body_hash, body)
        self.body = set()  # set(FuncNode)
        self.visited = 0
        self.first_visit = iteration + 1
        self.printed = False
        self.pf = None
        self.root = False

        for k, v in self.terms.items():
            primitives = (int, str, bool, float)
            if type(v) in (list, tuple):
                for vv in v:
                    if type(vv) not in primitives:
                        print("WARNING: py2problog can not yet deal properly with non-primitive data types in a "
                              "sequence ({}={}, {})".format(k, v, type(v)))
            elif type(v) not in primitives:
                print("WARNING: py2problog can not yet deal properly with non-primitive data types"
                      " ({}={}, {})".format(k, v, type(v)))

    def add_to_body(self, node):
        self.body.add(node)

    def merge_bodies(self, other):
        """Only necessary if running multiple runs"""
        # print('merge: {} -- {}'.format(self, other))
        self.bodies.update(other.bodies)
        self.visited += other.visited
        self.first_visit = min(self.first_visit, other.first_visit)

    def set_rvalue(self, rvalue):
        self.rvalue = rvalue
        self.visited += 1
        if len(self.body) == 0:
            return
        h = 1
        for b in self.body:
            h *= (1779033703 + 2*b.__hash__())
        if h not in self.bodies:
            self.bodies[h] = self.body
        self.body = set()

    def dot(self):
        if self.printed:
            return ""
        d = "  {} [label=\"{}\\n{} - {}\"];\n".format(clean_dot(self.unique_name()), self.unique_name(),
                                                      self.visited, self.first_visit)
        if self.pf is None and len(self.bodies) == 0:
            d += "  {}_{} [label=\"True\"];\n".format(clean_dot(self.unique_name()), 0)
            d += "  {} -> {}_{};\n".format(clean_dot(self.unique_name()), clean_dot(self.unique_name()), 0)
        for i, (h, bs) in enumerate(self.bodies.items()):
            d += "  {}_{} [label=\"^\"];\n".format(clean_dot(self.unique_name()), i)
            d += "  {} -> {}_{};\n".format(clean_dot(self.unique_name()), clean_dot(self.unique_name()), i)
            for b in bs:
                d += "  {}_{} -> {};\n".format(clean_dot(self.unique_name()), i, clean_dot(b.unique_name()))
                d += b.dot()

        self.printed = True
        return d

    def problog(self):
        if self.printed:
            return ""
        if self.pf is not None:
            return ""
        d = ""
        if self.pf is None and len(self.bodies) == 0:
            d += "{}.\n".format(self.problog_name())
        for i, (h, bs) in enumerate(self.bodies.items()):
            body = ", ".join([b.problog_name() for b in bs])
            d += "{} :- {}.\n".format(self.problog_name(), body)
            for b in bs:
                d += b.problog()
        self.printed = True
        return d

    def unique_name(self):
        terms = []
        for k, v in self.terms.items():
            # Only focus on actual values of simple objects
            # just an experiment, not enough
            terms.append("{}={}".format(k, v))
        terms.sort()
        term_str = ",".join(terms)
        s = "{}({})".format(self.pred, term_str)
        # if self.rvalue is not None:
        s += "={}".format(self.rvalue)
        return s

    def problog_name(self):
        if self.pf is not None:
            if type(self.pf) == ProbFactObject:
                s = "{}p_{}".format("" if self.rvalue else "\\+", self.pf.name)
            elif type(self.pf) == ADObject:
                s = "p_{}({})".format(self.pf.name, str(self.rvalue).lower())
            else:
                s = None
            return s
        terms = []
        for k, v in self.terms.items():
            # Only focus on actual values of simple objects
            # just an experiment, not enough
            terms.append("{},t_{}".format(k, clean_dot(str(v))))
        terms.sort()
        terms.append("{}".format(str(self.rvalue).lower()))
        term_str = ",".join(terms)
        s = "p_{}({})".format(clean_dot(self.pred), term_str)
        return s

    def __str__(self):
        return self.unique_name()

    def __hash__(self):
        if self.rvalue is None:
            raise Exception("ERROR: a hash for a FuncNode ({}) should not be called before it is complete".format(self))
        # h = int(hashlib.sha1(self.unique_name().encode()).hexdigest(), 16) % (10 ** 16)
        h = int(hashlib.sha1(self.unique_name().encode()).hexdigest(), 16)
        # print('{} -> {}'.format(self.unique_name(), h))
        return h

    def __eq__(self, other):
        return self.unique_name() == other.unique_name()


def deterministic(func):
    """Decorator to indicate that a function is deterministic and we can stop
    tracing when entering this function.

    The decorated function and it descedants should be detereministic.

    It is not necessary to indicate deterministic functions as being
    deterministic but it reduces the size of the underlying graph and speeds
    up inference.
    """
    deterministic_funcs.add(func.__name__)  # TODO: name is too simple

    def deterministic_wrapper(*args, **kwargs):
        if not settings.use_trace:
            sig = inspect.signature(func)
            func_name = func.__name__
            terms = sig.bind(*args, **kwargs).arguments
            process_def_call(func_name, terms)
        result = func(*args, **kwargs)
        if not settings.use_trace:
            process_def_return(result)
        return result
    return deterministic_wrapper


def probabilistic(func):
    """Decorator"""
    probabilistic_funcs.add(func.__name__)  # TODO: name is too simple

    def probabilistic_wrapper(*args, **kwargs):
        if not settings.use_trace:
            sig = inspect.signature(func)
            func_name = func.__name__
            terms = sig.bind(*args, **kwargs).arguments
            process_def_call(func_name, terms)
        result = func(*args, **kwargs)
        if not settings.use_trace:
            process_def_return(result)
        return result
    return probabilistic_wrapper


def process_def_call(func_name, terms):
    global current_node
    new_node = FuncNode(func_name, terms)
    trace_stack.append(current_node)
    current_node = new_node


def process_def_return(arg):
    global current_node
    if callable(arg):
        # returned function
        parent_node = trace_stack.pop()
        parent_node.body.update(current_node.body)
        current_node = parent_node
        return p2p_trace_calls
    current_node.set_rvalue(arg)
    if current_node.__hash__() in node_cache:
        node_cache[current_node.__hash__()].merge_bodies(current_node)
        current_node = node_cache[current_node.__hash__()]
    else:
        node_cache[current_node.__hash__()] = current_node
    parent_node = trace_stack.pop()
    parent_node.add_to_body(current_node)
    current_node = parent_node


def process_pf_call(pf):
    global current_node
    new_node = FuncNode(pf.name, {})
    new_node.pf = pf
    trace_stack.append(current_node)
    current_node = new_node


def process_pf_return(arg):
    global current_node
    current_node.set_rvalue(arg)
    if current_node.__hash__() in node_cache:
        node_cache[current_node.__hash__()].merge_bodies(current_node)
        current_node = node_cache[current_node.__hash__()]
    else:
        node_cache[current_node.__hash__()] = current_node
    parent_node = trace_stack.pop()
    parent_node.add_to_body(current_node)
    current_node = parent_node


def p2p_trace_calls(frame, event, arg):
    global current_node
    global node_cache
    global stop_tracing
    global deterministic_funcs
    global trace_stack
    co = frame.f_code
    func_name = co.co_name
    func_fn = co.co_filename
    if sys is not None:
        if sys.prefix in func_fn:
            return p2p_trace_calls
        if sys.exec_prefix in func_fn:
            return p2p_trace_calls
    if "<list" in func_name:
        # # Ignore system functions (e.g. list iterators)
        return p2p_trace_calls
    func_line_no = frame.f_lineno

    if 'py2problog.py' in func_fn:
        if func_name == "__call__":
            # TODO: how can we force more diverse samples? Or should we
            #       exhaustively iterate over all combinations of known
            #       probabilistic facts?
            this = frame.f_locals['self']
            if type(this) == ProbFactObject or type(this) == ADObject:
                if event == 'call':
                    logger.debug('call pf: {}'.format(this.name))
                    process_pf_call(this)
                    return p2p_trace_calls
                elif event == 'return':
                    logger.debug('return pf: {}'.format(arg))
                    process_pf_return(arg)
                    return
        return

    if event == 'call' or event == 'c_call':
        logger.debug('{}: {}.{}.{} -> {}'.format(event, func_fn, func_name, func_line_no, frame.f_locals))
        if stop_tracing > 0:
            stop_tracing += 1
            logger.debug("Not tracing call ({}->{})".format(stop_tracing-1, stop_tracing))
            return
        if func_name in deterministic_funcs:
            logger.debug("Stop tracing at {}".format(func_name))
            stop_tracing = 1
        # context should be stored
        argvalues = inspect.getargvalues(frame)
        # print(type(frame.f_code))
        # print(frame.f_code.co_names)
        # print(argvalues)
        if func_name == "<lambda>":
            # return p2p_trace_calls
            # print(inspect.getclosurevars(frame.f_code))
            terms = frame.f_locals
        else:
            terms = {k: v for k, v in frame.f_locals.items() if k in argvalues.args}
        # Line number is part of the name to differentiate between lambda funcs
        process_def_call(func_name+"_"+str(func_line_no), terms)
        return p2p_trace_calls

    elif event == 'return' or event == 'c_return':
        logger.debug('{}: {}_{}'.format(event, arg, func_line_no))
        if stop_tracing > 0:
            stop_tracing -= 1
            if stop_tracing > 1:
                logger.debug('Not tracing return ({}->{})'.format(stop_tracing+1, stop_tracing))
                return
        process_def_return(arg)
        return p2p_trace_calls

    elif event == 'line':
        # logger.debug('{}: {}.{}'.format(event, func_name, func_line_no))
        return

    elif event == 'exception':
        logger.debug('exception: {}'.format(arg))
        return

    else:
        logger.debug('Unknown event: {}'.format(event))


def dot():
    for node in node_cache.values():
        node.printed = False
    s = "digraph call_graph {\n"
    for root_node in root_nodes:
        s += root_node.dot()
    s += "}\n"
    return s


def problog():
    """Generate a ProbLog program representing the probabilistic model that
    results from the preceding calls of query.

    :returns: String represting a ProbLog program
    """
    for node in node_cache.values():
        node.printed = False
    s = "%% ProbLog program from Python\n"
    s += "\n%% Facts\n"
    for pf in pf_cache.values():
        if type(pf) == ProbFactObject:
            s += "{}::p_{}.\n".format(pf.prob, pf.name)
        elif type(pf) == ADObject:
            s += "; ".join(["{}::p_{}({})".format(pr, pf.name, str(v).lower()) for pr, v in pf.probs])+".\n"
    s += "\n%% Rules\n"
    for root_node in root_nodes:
        for i, (h, bs) in enumerate(root_node.bodies.items()):
            for b in bs:
                s += b.problog()
    # s += root_node.problog()+"\n"
    s += "\n%% Queries\n"
    for root_node in root_nodes:
        for i, (h, bs) in enumerate(root_node.bodies.items()):
            for b in bs:
                s += "query({}).\n".format(b.problog_name())
    return s


def last_useful_iteration():
    i = 0
    for node in node_cache.values():
        i = max(node.first_visit, i)
    return i


def reset_memoization():
    global memoization_cache
    memoization_cache = dict()


def reset():
    """Reset the caches required to remember results from multiple runs."""
    global node_cache
    global trace_stack
    global cnt_created_nodes
    cnt_created_nodes = 0
    reset_memoization()
    trace_stack = deque()
    node_cache = dict()

reset()


def setup():
    global trace_stack
    global is_running
    reset_memoization()
    trace_stack = deque()
    is_running = True
    if settings.use_trace:
        sys.settrace(p2p_trace_calls)


def teardown():
    global is_running
    sys.settrace(None)
    is_running = False
    print("--- STATS ---")
    print("Number of iterations: {}".format(settings.nb_samples))
    print("Created nodes: {}".format(cnt_created_nodes))
    print("Nodes in cache: {}".format(0 if node_cache is None else len(node_cache)))
    print("ProbFacts in cache: {}".format(len(pf_cache)))
    print("Last iteration that added a node: {}".format(last_useful_iteration()))
    print("-------------")


def query(fun, args=None):
    """Run a query.
    Can be called multiple times.

    :param fun: Function
    :args: Tuple of arguments for the given function
    """
    return query_sampling([(fun, args)])


def query_sampling(funcs):
    """Run set of queries with sampling.

    The adviced way to run py2problog.
    Not guaranteed to contain all paths. You can call this function multiple
    times.

    :param funcs: Funcs is a list of tuples with (func, (arg1, arg2, ...)) or
        (func, None)
    """
    global current_node
    global memoization_cache
    global iteration
    global root_nodes
    setup()
    if not settings.use_trace:
        new_funcs = []
        for fun, args in funcs:
            if fun.__name__ != "probabilistic_wrapper":
                new_funcs.append((deterministic(fun), args))
            else:
                new_funcs.append((fun, args))
        funcs = new_funcs
    for fun_idx, (fun, args) in enumerate(funcs):
        root_node = FuncNode("root_{}".format(len(root_nodes)), {})
        root_node.root = True
        current_node = root_node
        root_nodes.append(root_node)
        if args is None:
            args = tuple()
        for iteration in range(iteration, iteration+settings.nb_samples):
            print('Iteration {}'.format(iteration))
            if trace_stack is not None and len(trace_stack) > 0:
                raise Exception('Stack should be empty')
            result = fun(*args)
            memoization_cache = dict()
        root_node.set_rvalue(1)
        node_cache[root_node.__hash__()] = root_node
    teardown()
    # global node_cache
    # for k,v in node_cache.items():
    #     print("{} -> {}".format(v, v.bodies))


def query_es(fun, args=None):
    """Run query with exhaustive search over all defined probabilistic facts.

    This is intractable in practice! It is only here for testing purposes.
    Use the query() definition.
    """
    global exhaustive_search
    global current_node
    global memoization_cache
    global iteration
    exhaustive_search = True
    if args is None:
        args = tuple()
    setup()
    root_node = FuncNode("root", {})
    root_node.root = True
    current_node = root_node
    root_nodes.append(root_node)
    memoization_cache = dict()
    pfs = list(pf_cache.values())
    print('Iterating over {} probabilistic facts'.format(len(pfs)))
    pf_values = [0]*len(pfs)
    total_values = 1
    for pf in pfs:
        total_values *= pf.nb_values()
        memoization_cache[pf] = pf.get_value(0)
    needle = len(pf_values) - 1
    for iteration in range(total_values):
        print('Iteration {}: {}'.format(iteration, ','.join([str(pf.name)+'='+str(v) for pf,v in zip(pfs,pf_values)])))
        result = fun(*args)
        needle = len(pf_values) - 1
        pf = pfs[needle]
        while pf_values[needle] == pf.nb_values() - 1:
            pf_values[needle] = 0
            memoization_cache[pf] = pf.get_value(0)
            needle -= 1
            pf = pfs[needle]
        pf_values[needle] += 1
        memoization_cache[pf] = pf.get_value(pf_values[needle])
    root_node.set_rvalue(1)
    node_cache[root_node.__hash__()] = root_node
    teardown()
    exhaustive_search = False


def count_samples(func, repeat=None):
    """Generate samples by running the given function multiple times.

    This is a naive approach for sampling. It is only here for testing
    purposes. Use the query() definition.

    :param func: Function for which we will sample the output
    :param repeat: Number of samples to generate (default is
        settings.nb_samples)
    """
    if repeat is None:
        repeat = settings.nb_samples
    uniform_prob = settings.uniform_prob
    settings.uniform_prob = False

    def func2():
        reset_memoization()
        return func()
    cnt = Counter((func2() for i in range(repeat)))
    settings.uniform_prob = uniform_prob
    return cnt


def hist_samples(func, repeat=None, title=None):
    """Plot histogram for the outcomes of the given function.

    This is a naive approach for sampling based on count_samples().
    It is only here for testing purposes. Use the query() definition.
    """
    if repeat is None:
        repeat = settings.nb_samples
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.style.use('ggplot')
    # %matplotlib inline 
    labels, values = zip(*count_samples(func, repeat=repeat).items())
    total = sum(values)
    # print('total: {}'.format(total))
    # print('values: {}'.format(values))
    values = [value/total*100 for value in values]
    indexes = range(len(labels))
    width = 1
    fig = plt.figure(figsize=(10,0.4*len(labels)))
    plt.barh(indexes, values, width)
    plt.yticks([i+width*0.5 for i in indexes], ["{:>10}".format(str(l)) for l in labels])
    plt.xlim(0, 100)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: '{:.0f}%'.format(x)))
    if title is not None:
        plt.title(title)
    plt.show(block=True)
