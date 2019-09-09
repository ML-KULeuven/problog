from collections import defaultdict
from pathlib import Path
from problog.core import ProbLogObject
from problog.logic import unquote, term2list, ArithmeticError, Term, Constant
from problog.prolog_engine.swip import query, parse

# with open(Path(__file__).parent / 'engine_kbest.pl') as f:
#     k_engine = f.read()


def handle_prob(prob):
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
    if type(a) is int:
        if a < 0:
            return 'X{}'.format(-a)
        else:
            return 'A{}'.format(a + 1)
    else:
        return str(a)


def handle_functor(func, args=None):
    if type(func) is str:
        if args is not None and len(args) > 0:
            return '{}({})'.format(unquote(func), ','.join(handle_var(a) for a in args))
        else:
            return unquote(func)
    else:
        return str(func.with_args(*func.args, *args))


class SWIProgram(ProbLogObject):

    def __init__(self, db):
        self.facts = []
        self.clauses = []
        self.db = db
        self.ad_heads = defaultdict(list)
        self.i = 0
        if self.db is not None:
            for n in self.db.iter_nodes():
                ntype = type(n).__name__
                if ntype == 'fact':
                    self.add_fact(n)
                elif ntype == 'clause':
                    self.add_clause(n)
                elif ntype == 'choice':
                    self.add_choice(n)

    def to_str(self, node):
        ntype = type(node).__name__
        if ntype == 'conj':
            return ','.join(self.to_str(self.db.get_node(c)) for c in node.children)
        elif ntype == 'call':
            return handle_functor(node.functor, node.args)
        elif ntype == 'neg':
            return 'neg({})'.format(self.to_str(self.db.get_node(node.child)))
        return ntype + '_unhandled'

    def add_fact(self, node):
        self.i += 1
        self.facts.append((handle_prob(node.probability), handle_functor(node.functor, node.args), self.i))

    def add_clause(self, node):
        # prob = node.probability if node.group is None else None
        self.clauses.append(
            (handle_functor(node.functor, node.args),
             self.to_str(self.db.get_node(node.child))))

    def add_choice(self, node):
        self.i += 1
        self.ad_heads[node.group].append(
            (handle_prob(node.probability), handle_functor(node.functor, node.args), self.i))

    def get_lines(self):
        lines = ['rule({},[{}])'.format(*c) for c in self.clauses]
        lines += ['fact({},{},{})'.format(*c) for c in self.facts]
        for ad in self.ad_heads:
            if len(self.ad_heads[ad]) == 1:
                lines.append('fact({},{},{})'.format(*self.ad_heads[ad][0]))
            else:
                lines.append('ad([' + ','.join('p({},{},{})'.format(*head) for head in self.ad_heads[ad]) + '])')
        return lines

    def __str__(self):
        return '\n'.join(l + '.' for l in self.get_lines())

    def get_proofs(self, q, k):
        if k is None:
            res = query(Path(__file__).parent / 'engine_kbest.pl', 'top({}, Proofs)'.format(q), asserts=self.get_lines())
        else:
            res = query(Path(__file__).parent / 'engine_kbest.pl', 'top({},{}, Proofs)'.format(k, q), asserts=self.get_lines())

        proofs = []
        for r in res:
            proofs.append(r.args)
        return proofs

    def build_formula(self, proof, target):
        type = proof.functor
        if type == 'fact':
            p, name, i = proof.args
            i = str(i)
            p = float(p)
            if p > 1.0 - 1e-8:
                p = None
            group = None
            if name.functor == 'choice':
                i = str(name)
                group = name.args[0], name.args[3:]
            # key = target.add_atom(i, p, name=name, group=group)
            try:
                key = target.get_node_by_name(name)
            except KeyError:
                key = target.add_or([], placeholder=True, readonly=False)
                target.add_name(name, key)
            target.add_disjunct(key, target.add_atom(i, p, group=group))
            # target.add_name(name, key)
            return key
        elif type == 'and':
            name, body = proof.args
            body = term2list(body, False)
            try:
                key = target.get_node_by_name(name)
            except KeyError:
                key = target.add_or([], placeholder=True, readonly=False)
                target.add_name(name, key)
            target.add_disjunct(key, target.add_and([self.build_formula(b, target) for b in body]))
            return key
        elif type == 'neg':
            return -self.build_formula(proof.args[0], target)
        elif type == 'cycle':
            name, = proof.args
            return target.get_node_by_name(name)
        # elif type == 'foreign':
        #     name, = proof.args
        #     return target.add_atom(1000, None, name=name)
        elif type == 'builtin':
            return target.TRUE
        elif type == 'neural_fact':
            p, name, i, net, args = proof.args
            i = str(i)
            # p = float(p)
            # if p > 1.0 - 1e-8:
            #     p = None
            p = Term('nn', Term(net), args, Constant(p))
            key = target.add_atom(target.get_next_atom_identifier(), p, name=name)
            target.add_name(name, key)
            return key
        else:
            raise Exception('Unhandled node type ' + str(proof))

    def ground(self, query, target, k=None):
        proofs = self.get_proofs(query, k)
        # print('{} proofs found'.format(len(proofs)))
        query = parse(query)
        if len(proofs) == 0:  # query is determinstically false, add trivial
            target.add_name(query, target.FALSE, label=target.LABEL_QUERY)
        for q, proof in proofs:
            # if len(proof) == 0:  # query is deterministically true
            #     target.add_name(q, target.TRUE, label=target.LABEL_QUERY)

            key = self.build_formula(proof, target)
            target.add_name(q, key, label=target.LABEL_QUERY)
        return target
