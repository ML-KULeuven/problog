from collections import defaultdict
from pathlib import Path
from time import time

from problog.prolog_engine.swip import parse
from problog.prolog_engine.threaded_prolog import ThreadedProlog

from problog.core import ProbLogObject
from problog.logic import unquote, term2list, ArithmeticError, Term


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

        # Warning: Has to be evaluated before the creation of the Prolog engine
        self.parse_directives()
        self.prolog = ThreadedProlog()

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

    def add_clause(self, node):
        i = self.new_entry()
        body = self.to_str(self.db.get_node(node.child))
        self.clauses.append(
            (i, handle_functor(node.functor, node.args), body))
        self.prolog.assertz('cl({},{})'.format(self.clauses[-1][1], self.clauses[-1][2]))

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

    def construct_node(self, node, target, d, neg=False):
        if type(node) is list:
            if neg:
                return target.add_and([target.negate(self.construct_node(c, target, d)) for c in node])
            else:
                return target.add_or([self.construct_node(c, target, d) for c in node])
        if node.functor == '::':
            p = float(node.args[1])
            return target.add_atom(target.get_next_atom_identifier(), p, name=node.args[2])
        elif node.functor == ':-':
            return self.construct_node(node.args[1], target, d)
        elif node.functor == 'neg':
            try:
                return d[node]
            except KeyError:
                return target.negate(self.construct_node(node.args[0], target, d))
        elif node.functor == ',':
            return target.add_and([self.construct_node(c, target, d) for c in node.args])
        elif node.functor == 'true':
            return target.TRUE
        elif node.functor == 'builtin':
            # return target.add_atom(target.get_next_atom_identifier(), True, name=
            return True
        elif node.functor == 'foreign':
            # return target.add_atom(target.get_next_atom_identifier(), True, name=
            return True
        elif node.functor == 'call':
            return self.construct_node(node.args[0],target,d)
        else:
            try:
                return d[node]
            except KeyError:
                return target.FALSE

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
        nodes = defaultdict(list)
        dependencies = defaultdict(list)
        d = dict()
        for p in proofs:
            if p.functor == '::':
                nodes[p.args[2]].append(p)
            elif p.functor == ':-':
                children = self.get_children(p.args[1])
                nodes[p.args[0]].append(p)
                dependencies[p.args[0]] += children
        for k in dependencies:
            dependencies[k] = [n for n in dependencies[k] if n in nodes]
        while nodes:  # While nodes is not empty
            new_sol = False
            keys = list(nodes)
            for k in keys:
                if len(dependencies[k]) == 0:  # If the node has no more undefined children
                    neg = k.functor == 'neg'
                    d[k] = self.construct_node(nodes[k], target, d, neg=neg)  # Construct the node
                    for k2 in dependencies:  # Remove the node from undefined dependency from all nodes
                        try:
                            dependencies[k2].remove(k)
                        except ValueError:
                            pass
                    new_sol = True
                    del (nodes[k])
                    del (dependencies[k])
            if not new_sol:  # No nodes without undefined children, so we have a cycle!
                break
        return d

    # def build_formula(self, proof, target):
    #     if not hasattr(target, 'd'):
    #         target.d = dict()
    #     t = proof.functor
    #     if t == ':-':
    #         name, body = proof.args
    #         key = self.build_formula(name, target)
    #         # if str(body) not in target.d:
    #         if True:
    #             body_key = self.build_formula(body, target)
    #             target.add_disjunct(key, body_key)
    #             target.d[str(body)] = body_key
    #         return key
    #     elif t == ',':
    #         body = proof.args
    #         new = target.add_and([self.build_formula(b, target) for b in body])
    #         return new
    #     elif t == 'neg':
    #         negated = proof.args[0]
    #         return -self.build_formula(negated, target)
    #     # elif t == ';':
    #     #     body = proof.args
    #     #     new = target.add_or([self.build_formula(b, target) for b in body])
    #     #     return new
    #     elif t == 'true':
    #         return target.TRUE
    #     elif t == '::':
    #         id, p, name = proof.args
    #         key = self.build_formula(name, target)
    #         if id not in target.d:
    #             p = float(proof.args[1])
    #             fact_key = target.add_atom(target.get_next_atom_identifier(), p, name=name)
    #             target.add_disjunct(key, fact_key)
    #             target.d[id] = fact_key
    #         return key
    #     else:
    #         try:
    #             return target.d[str(proof)]
    #         except KeyError:
    #             key = target.add_or([], placeholder=True, readonly=False, name=proof)
    #             target.d[str(proof)] = key
    #             return key

    def add_proofs(self, proofs, ground_queries, target):
        target.names = dict()
        d = self.build_formula(proofs, target)
        for q in ground_queries:
            key = d[q]
            target.add_name(q, key, label=target.LABEL_QUERY)
        return target

    def query(self, query, profile=0):
        query = str(query)
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
                    raise Exception('choices not implemented')
