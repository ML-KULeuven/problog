from collections import defaultdict
from pathlib import Path
from time import time

from problog.core import ProbLogObject
from problog.logic import unquote, term2list, ArithmeticError, Term, Constant
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

    def __init__(self, db, consult='engine.pl'):
        self.facts = []
        self.clauses = []
        self.db = db
        self.ad_heads = defaultdict(list)
        self.i = 0
        self.index_dict = dict()
        self.prolog = ThreadedProlog()
        self.prolog.consult(str(Path(__file__).parent / consult))
        self.parse_db()

    def to_str(self, node):
        ntype = type(node).__name__
        if ntype == 'conj':
            children = [self.to_str(self.db.get_node(c)) for c in node.children]
            children = [c for c in children if c != 'true']
            return ','.join(children)
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
        if body == 'true':
            body = ''
        self.clauses.append(
            (i, handle_functor(node.functor, node.args), body))
        self.prolog.assertz('cl({},({}))'.format(self.clauses[-1][1], self.clauses[-1][2]))

    def add_directive(self, node):
        print(node)
        print(vars(node))
        pass

    def get_lines(self):
        lines = ['cl({},({}))'.format(c[1], c[2]) for c in self.clauses]
        lines += ['fa({},{},{})'.format(*c) for c in self.facts]
        return lines

    def __str__(self):
        return '\n'.join(l + '.' for l in self.get_lines())

    def build_formula(self, proof, target):
        if not hasattr(target, 'd'):
            target.d = dict()
        t = proof.functor
        if t == ':-':
            name, body = proof.args
            key = self.build_formula(body, target)
            target.add_name(name, key)
            return key
        elif t == ',':
            body = proof.args
            new = target.add_and([self.build_formula(b, target) for b in body])
            return new
        elif t == ';':
            body = proof.args
            new = target.add_or([self.build_formula(b, target) for b in body])
            return new
        elif t == 'true':
            return target.TRUE
        elif t == '::':
            id = int(proof.args[0])
            name = proof.args[2]
            if id not in target.d:
                p = float(proof.args[1])
                key = target.add_atom(target.get_next_atom_identifier(), p, name=name)
                target.d[id] = key
            return target.d[id]
        else:
            raise Exception('Unhandled node type ' + str(proof))

    def add_proofs(self, proofs, target):
        target.names = dict()
        for proof in proofs:
            query, proof = proof.args
            key = self.build_formula(proof, target)
            if not target.is_trivial():
                target.add_name(query, key, label=target.LABEL_QUERY)
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
                    print([p for p in term2list(parse(result[0][k]))])
                    out[k] = [p for p in term2list(parse(result[0][k]))]
                else:
                    out[k] = parse(v)
            print(out)
            return out
        else:
            raise (Exception('Expected exactly one result, got {}'.format(len(result))))

    def parse_call(self, node):
        pass

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
                    self.parse_call(n)
                elif ntype == 'clause':
                    if not n.functor == '_directive':
                        self.add_clause(n)
                elif ntype == 'choice':
                    raise Exception('choices not implemented')
