from collections import defaultdict
from pathlib import Path

from problog.core import ProbLogObject
from problog.logic import unquote, term2list
from problog.parser import PrologParser
from problog.program import ExtendedPrologFactory
from problog.prolog_engine.swip import run_string

parser = PrologParser(ExtendedPrologFactory())
with open(Path(__file__).parent / 'engine_kbest.pl') as f:
    k_engine = f.read()


def parse(to_parse):
    return parser.parseString(str(to_parse) + '.')[0]


def handle_prob(prob):
    if prob is None:
        return 1.0
    elif type(prob) is int:
        return handle_var(prob)
    return float(prob)


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
            lines.append('ad([' + ','.join('p({},{},{})'.format(*head) for head in self.ad_heads[ad]) + '])')
        return lines

    def __str__(self):
        return '\n'.join(l + '.' for l in self.get_lines())

    def get_proofs(self, query, k):
        if k is None:
            k = 100  # TODO FIX!
        file = str(self) + '\n' + k_engine
        res = run_string(file, 'top({},{})'.format(k, query))
        res = parse(res)
        res = term2list(parse(res), deep=False)

        proofs = []
        for r in res:
            proofs.append(r.args)
        return proofs
        # except PrologError as e:
        #     if 'unknown_clause' in str(e):
        #         raise UnknownClause(str(e), 0)

    def build_formula(self, proof, target, label=None):
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
                group = name.args[0], name.args[1].args[3:]
            key = target.add_atom(i, p, name=name, group=group)
            target.add_name(name, key, label=label)
            return key
        elif type == 'and':
            name, body = proof.args
            body = term2list(body, False)
            try:
                key = target.get_node_by_name(name)
            except KeyError:
                key = target.add_or([], placeholder=True, readonly=False)
                target.add_name(name, key, label=label)
            target.add_disjunct(key, target.add_and([self.build_formula(b, target) for b in body]))
            return key
        elif type == 'cycle':
            name, = proof.args
            return target.get_node_by_name(name)
        else:
            raise Exception('Unhandled node type '+str(proof))
        # name, body = proof.args
        # body = term2list(body, False)
        # if type == 'and':
        #     if len(body) == 0:
        #         if name.functor == 'p':
        #             p = float(name.args[0])
        #             if p > 1.0 - 1e-8:
        #                 p = None
        #             id = str(name)
        #             group = None
        #             if name.args[1].functor == 'choice':
        #                 id = str(name.args[1])
        #                 group = name.args[1].args[0], name.args[1].args[3:]
        #             return target.add_atom(id, p, name=name, group=group)
        #         else:
        #             return target.get_node(str(name))
        #     else:
        #         return target.add_and([self.build_formula(b, target) for b in body])  # %, name=name)
        # elif type == 'neg':
        #     return - target.add_and([self.build_formula(b, target) for b in body])  # , name=name)

    def ground(self, query, target, k=None):

        proofs = self.get_proofs(query, k)
        proof_keys = defaultdict(list)
        query = parse(query)
        if len(proofs) == 0:  # query is determinstically false, add trivial
            target.add_name(query, target.FALSE, label=target.LABEL_QUERY)
        for q, proof in proofs:
            # if len(proof) == 0:  # query is deterministically true
            #     target.add_name(q, target.TRUE, label=target.LABEL_QUERY)

            proof_keys[q].append(self.build_formula(proof, target, label=target.LABEL_QUERY))
        # for q in proof_keys:
        #     key = target.add_or(proof_keys[q])
        #     target.add_name(q, key, label=target.LABEL_QUERY)
        return target
