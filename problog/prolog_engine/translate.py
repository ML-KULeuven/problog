from collections import defaultdict
from pathlib import Path

from pyswip import Prolog

from problog.core import  ProbLogObject
from problog.formula import LogicFormula
from problog.program import ExtendedPrologFactory
from problog.parser import PrologParser
from problog.logic import unquote
parser = PrologParser(ExtendedPrologFactory())

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


def process_proof(proof):
    expanded = []
    to_expand = [proof]
    while len(to_expand) > 0:
        l = to_expand[0]
        del (to_expand[0])
        if type(l) is list:
            to_expand = l + to_expand
        else:
            expanded.append(l)
    proof = []
    for t in expanded:
        neg = False
        if t.name.value == 'neg':
            neg = True
            t = t.args[0]
        p, f, i = t.args
        f = parse(f)
        i = int(i)
        proof.append((p, f, i, neg))
    return proof


class TranslatedProgram(ProbLogObject):

    def __init__(self, db):
        self.clauses = []
        self.db = db
        self.ad_heads = defaultdict(list)
        self.i = 0
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
        self.clauses.append((handle_prob(node.probability), handle_functor(node.functor, node.args), self.i, ''))

    def add_clause(self, node):
        self.i += 1
        prob = node.probability if node.group is None else None
        self.clauses.append(
            (handle_prob(prob), handle_functor(node.functor, node.args), self.i, self.to_str(self.db.get_node(node.child))))

    def add_choice(self, node):
        self.i += 1
        self.ad_heads[node.group].append((handle_prob(node.probability), handle_functor(node.functor, node.args), self.i))

    def get_lines(self):
        lines = ['ad([p({},{},{})],[{}])'.format(*c) for c in self.clauses]
        for ad in self.ad_heads:
            lines.append('ad([' + ','.join('p({},{},{})'.format(*head) for head in self.ad_heads[ad]) + '],[])')
        return lines

    def __str__(self):
        return '\n'.join(l + '.' for l in self.get_lines())

    def get_proofs(self, query):
        prolog = Prolog()
        prolog.retractall('ad(_,_)')
        path = str(Path(__file__).parent / 'engine.pl')
        prolog.consult(path)
        for l in self.get_lines():
            prolog.assertz(l)
        result = list(prolog.query('prove({},Q,Proof)'.format(query)))
        proofs = []
        # query = parse(query)
        for r in result:
            # new_vars = {Var(v): Constant(r[v]) for v in r}
            # nq = query.apply_term(new_vars)
            nq = parse(str(r['Q']))
            proofs.append((nq, process_proof(r['Proof'])))
        return proofs

    def ground(self, query, target):

        proofs = self.get_proofs(query)
        proof_keys = defaultdict(list)
        query = parse(query)
        if len(proofs) == 0: #query is determinstically false, add trivial
            target.add_name(query,target.FALSE, label=target.LABEL_QUERY)
        for q, proof in proofs:
            if len(proof) == 0: #query is deterministically true
                target.add_name(q, target.TRUE, label=target.LABEL_QUERY)
            else:
                proof_atoms = []
                for p, a, i, n in proof:
                    p = None if p > 1.0 - 1e-8 else p
                    group = None
                    if a.functor == 'choice':
                        group = a.args[0], a.args[3:]
                    key = target.add_atom(i, p, name=a, group=group)
                    proof_atoms.append(-key if n else key)
                proof_keys[q].append(target.add_and(proof_atoms))
        for q in proof_keys:
            key = target.add_or(proof_keys[q])
            target.add_name(q, key, label=target.LABEL_QUERY)
        return target


def translate_clausedb(db):
    program = TranslatedProgram(db)

    for n in db.iter_nodes():
        ntype = type(n).__name__

        if ntype == 'fact':
            program.add_fact(n)
        elif ntype == 'clause':
            program.add_clause(n)
        elif ntype == 'choice':
            program.add_choice(n)
    return program
