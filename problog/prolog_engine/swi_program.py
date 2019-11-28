import re
from collections import defaultdict
from pathlib import Path
from time import time

from problog.core import ProbLogObject
from problog.logic import unquote, term2list, ArithmeticError
from problog.prolog_engine.swip import parse
from problog.prolog_engine.threaded_prolog import ThreadedProlog


def handle_prob(prob):
    '''
    Turns the probability into a canonical form. If probability is none, returns 1.0.
    If probability is an integer, it returns a variable as returned by handle_var.
    If the probability can be cast to a float, return the float.
    Else, cast it to a string and return that.
    :param prob:
    :return:
    '''
    if prob is None:
        return 1.0
    elif type(prob) is int:
        return handle_var(prob)
    try:
        prob = float(prob)
        return prob
    except ArithmeticError:
        return str(prob)


def handle_var(a):
    '''Turn the variable into a canonical string format.'''
    if type(a) is int:
        if a < 0:
            return 'X{}'.format(-a)
        else:
            return 'A{}'.format(a + 1)
    else:
        return str(a)


def handle_functor(func, args=None):
    '''Turn the functor into a canonical format.'''
    if type(func) is str:
        if args is not None and len(args) > 0:
            return '{}({})'.format(unquote(func), ','.join(handle_var(a) for a in args))
        else:
            return unquote(func)
    else:
        return str(func.with_args(*func.args, *args))


class SWIProgram(ProbLogObject):

    def __init__(self, db, consult='engine2.pl'):
        self.facts = []
        self.clauses = []
        self.db = db
        self.ad_heads = defaultdict(list)
        self.i = 0
        self.index_dict = dict()
        self.groups = dict()
        # Warning: Has to be evaluated before the creation of the Prolog engine
        self.parse_directives()
        self.prolog = ThreadedProlog()
        self.d = dict()
        self.prolog.consult(str(Path(__file__).parent / consult))
        self.parse_db()

    def to_str(self, node):
        ntype = type(node).__name__
        if ntype == 'conj':
            children = [self.to_str(self.db.get_node(c)) for c in node.children]
            children = [c for c in children]
            if len(children) > 1:
                return '({})'.format(','.join(children))
            else:
                return str(children[0])
        elif ntype == 'call':
            return handle_functor(node.functor, node.args)
        elif ntype == 'neg':
            return 'neg({})'.format(self.to_str(self.db.get_node(node.child)))
        return ntype + '_unhandled'

    def new_entry(self):
        self.i += 1
        return self.i

    def add_fact(self, node):
        i = self.new_entry()
        new_fact = (i, handle_prob(node.probability), handle_functor(node.functor, node.args))
        self.facts.append(new_fact)
        self.index_dict[i] = new_fact
        self.prolog.assertz('fa({},{},{})'.format(*self.facts[-1]))
        print('fa({},{},{})'.format(*self.facts[-1]))

    def add_clause(self, node):
        i = self.new_entry()
        body = self.to_str(self.db.get_node(node.child))
        self.clauses.append(
            (i, handle_functor(node.functor, node.args), body))
        self.prolog.assertz('cl({},{})'.format(self.clauses[-1][1], self.clauses[-1][2]))
        print('cl({},{})'.format(self.clauses[-1][1], self.clauses[-1][2]))

    def add_choice(self, node):
        self.add_fact(node)

    def add_directive(self, node):
        print(node)
        print(vars(node))
        pass

    def get_lines(self):
        lines = ['cl({},{})'.format(c[1], c[2]) for c in self.clauses]
        lines += ['fa({},{},{})'.format(*c) for c in self.facts]
        return lines

    def __str__(self):
        return '\n'.join(l + '.' for l in self.get_lines())

    def construct_node(self, node, target, neg=False):
        if type(node) is list:  # The node is a list if there's several proofs for the node
            if neg:
                # If the definition is for a negated node, the list represents and and of nots
                return target.add_and([target.negate(self.construct_node(c, target)) for c in node])
            else:
                # If the definition is not negated, it represents an or
                return target.add_or([self.construct_node(c, target) for c in node])
        try:
            return self.d[node]
        except KeyError:
            if node.functor == '::':  # If the definition is a fact
                name = node.args[2]
                group = None
                if name.functor == 'choice':
                    group = name.args[0], name.args[3:]
                    print('group=',group)
                p = float(node.args[1])  # Get its probability
                # add an atom to the formula
                k = target.add_atom(target.get_next_atom_identifier(), p, name=name, group=group)
                self.d[node] = k
                return k
            elif node.functor == ':-':  # If the definition is a clause
                return self.construct_node(node.args[1], target)  # Recursively construct a node for the body
            elif node.functor == 'neg':
                # try:
                #     # If there's a definition for the negated node (i.e. not the negation of an atom)
                #     # Lookup the node in the logical formula
                #     return d[node]
                # except KeyError:
                #     # Else, its the negation of an atom. Construct it recursively
                return target.negate(self.construct_node(node.args[0], target))
            elif node.functor == ',':  # The definition is an and
                return target.add_and([self.construct_node(c, target) for c in node.args])
            elif node.functor == 'true':  # Determinstically true
                return target.TRUE
            elif node.functor == 'builtin':  # Proven with a builtin
                # return target.add_atom(target.get_next_atom_identifier(), True, name=
                return target.TRUE
            elif node.functor == 'foreign':  # Proven through a foreign predicate
                # return target.add_atom(target.get_next_atom_identifier(), True, name=
                return target.TRUE
            elif node.functor == 'call':  # Proven through a meta call, construct it recursively for the node that is called
                return self.construct_node(node.args[0], target)
            else:
                return target.FALSE  # Node is not present in the graph, so deterministically false

    def get_children(self, term):
        if term.functor == ',':
            children = []
            for c in term.args:
                children += self.get_children(c)
            return children
        elif term.functor == 'neg':
            return self.get_children(term.args[0])
        else:
            return [term]

    def build_formula(self, proofs, target):
        # Keys are nodes, and the value is a list of all definitions of that node
        nodes = defaultdict(list)
        # Keys are nodes, and the value is a list of all nodes that this node depends on directly
        dependencies = defaultdict(list)
        # Keys are nodes, and the values area the keys of the nodes in the ground logic formula
        for p in proofs:  # Loop over all elements of the proof
            if p.functor == '::':  # If it's a fact
                nodes[p.args[2]].append(p)  # Add it to the definitions
            elif p.functor == ':-':  # If it's a clause
                children = self.get_children(p.args[1])  # Recursively determine its children
                dependencies[p.args[0]] += children  # Add these to its dependencies
                nodes[p.args[0]].append(p)  # Add it to the definition

        for k in dependencies:
            # Only keep the dependencies for which there is a definition. If a dependency has no dependency,
            # it means there's no proof for it and thus deterministically false
            dependencies[k] = [n for n in dependencies[k] if n in nodes]
        while nodes:  # While nodes is not empty
            new_sol = False  # Keeps track if a new node has been added to the ground formula
            keys = list(nodes)  # List of keys of the dictionary
            for k in keys:
                if len(dependencies[k]) == 0:  # If the node has no more undefined children
                    neg = k.functor == 'neg'  # Check whether the current node is negated
                    self.d[k] = self.construct_node(nodes[k], target, neg=neg)  # Construct the node
                    for k2 in dependencies:  # Remove the node from undefined dependency from all nodes
                        try:
                            dependencies[k2].remove(k)
                        except ValueError:
                            pass
                    new_sol = True  # A node was added
                    # Remove the node from the dict as it's fully processed
                    del (nodes[k])
                    del (dependencies[k])
            if not new_sol:
                # No node could be added in this iteration. So all nodes have unfulfilled dependencies
                raise NotImplemented('Cycles are not yet supported in build_formula')
        return self.d

    def add_proofs(self, proofs, ground_queries, target):
        target.names = dict()
        d = self.build_formula(proofs, target)
        print(d)
        for q in ground_queries:
            key = d[q]
            target.add_name(q, key, label=target.LABEL_QUERY)
        return target

    def query(self, query, profile=0):
        query = str(query)

        # TODO: replace this, a bit hacky
        # Replaces $VAR(X) with actual variables
        # Needed when specified queries are non ground
        print('query=', query)
        query = re.sub(r'\$VAR\((.*?)\)', r'X\1', query)
        print('query=', query)
        #

        if profile > 0:
            start = time()
            if profile > 1:
                query = 'profile((between(1,100,_),{},fail);true)'.format(query)
        result = list(self.prolog.query(query))
        if profile > 0:
            print('Query: {} answered in {} seconds'.format(query, time() - start))
        if len(result) == 1:
            out = {}
            for k in result[0]:
                v = result[0][k]
                if type(v) is list:
                    out[k] = [p for p in term2list(parse(result[0][k]))]
                else:
                    out[k] = parse(v)
            return out
        else:
            raise (Exception('Expected exactly one result, got {}'.format(len(result))))

    def parse_directives(self):
        """
        Parse the directives (before the creation of the Prolog engine)
        :return:
        """
        if self.db is not None:
            for n in self.db.iter_nodes():
                ntype = type(n).__name__
                if ntype == 'call':
                    self.parse_call(n)

    def parse_call(self, node, directives=True):
        """
        Parse a call node
        :param node: The node to parse
        :param directives: Is True if the directives have to be parsed
        :return:
        """
        if directives and node.functor == "_use_module":
            filename = node.args[1]
            self.db.use_module(filename=filename, predicates=None, location=node.location)

    def parse_db(self):
        """
        Parse the database (ClauseDB) into a valid SWI-Prolog
        :return: Nothing (Update the current object)
        """
        if self.db is not None:
            for n in self.db.iter_nodes():
                ntype = type(n).__name__
                if ntype == 'fact':
                    self.add_fact(n)
                elif ntype == 'call':
                    self.parse_call(n, directives=False)
                elif ntype == 'clause':
                    if not n.functor == '_directive':
                        self.add_clause(n)
                elif ntype == 'choice':
                    self.add_choice(n)
