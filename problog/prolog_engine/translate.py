from collections import defaultdict
from pathlib import Path

from pyswip import Prolog, Functor, Atom
from pyswip.prolog import PrologError
from problog.core import ProbLogObject
from problog.logic import unquote
from problog.parser import PrologParser
from problog.program import ExtendedPrologFactory
from problog.engine import UnknownClause

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


def process_result(result):
    nq = None
    proof = None
    for r in result:
        var_name = r.args[0].value
        if var_name == 'Q':
            nq = parse(r.args[1].value)
        elif var_name == 'Proof':
            proof = r.args[1]
    return nq, proof

def pyswip_to_str(obj):
    if type(obj) is Functor:
        name = obj.name.value
        if name == '\\\\==':
            name == '\\=='
        return '{}({})'.format(name, ','.join(pyswip_to_str(a) for a in obj.args))
    elif type(obj) is Atom:
        return str(obj)
    elif type(obj) is list:
        return '['+','.join(pyswip_to_str(a) for a in obj)+']'
    else:
        return str(obj)

def process_proof(proof):
    name = proof.name.value
    node_name = pyswip_to_str(proof.args[0])
    body = proof.args[1]
    return name, node_name, [process_proof(b) for b in body]


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
            (handle_prob(prob), handle_functor(node.functor, node.args), self.i,
             self.to_str(self.db.get_node(node.child))))

    def add_choice(self, node):
        self.i += 1
        self.ad_heads[node.group].append(
            (handle_prob(node.probability), handle_functor(node.functor, node.args), self.i))

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
        try:
            result = prolog.query('prove({},Q,Proof)'.format(query), normalize=False)
            proofs = []
            for r in result:
                nq, proof = process_result(r)
                proofs.append((nq, process_proof(proof)))
            return proofs
        except PrologError as e:
            if 'unknown_clause' in str(e):
                raise UnknownClause(str(e), 0)


    def build_formula(self, proof, target):
        _, name, body = proof
        name = parse(name)
        if proof[0] == 'and':
            if len(body) == 0:
                p = float(name.args[0])
                if p > 1.0 - 1e-8:
                    p = None
                id = str(name.args[2])
                group = None
                if name.args[1].functor == 'choice':
                    id = str(name.args[1])
                    group = name.args[1].args[0], name.args[1].args[3:]
                return target.add_atom(id, p, name=name, group=group)
            else:
                return target.add_and([self.build_formula(b, target) for b in body])#%, name=name)
        elif proof[0] == 'neg':
            return - target.add_and([self.build_formula(b, target) for b in body])#, name=name)

    def ground(self, query, target):

        proofs = self.get_proofs(query)
        proof_keys = defaultdict(list)
        query = parse(query)
        if len(proofs) == 0:  # query is determinstically false, add trivial
            target.add_name(query, target.FALSE, label=target.LABEL_QUERY)
        for q, proof in proofs:
            if len(proof) == 0:  # query is deterministically true
                target.add_name(q, target.TRUE, label=target.LABEL_QUERY)
            else:
                # proof_atoms = []
                # for p, a, i, n in proof:
                #     p = None if p > 1.0 - 1e-8 else p
                #     group = None
                #     if a.functor == 'choice':
                #         group = a.args[0], a.args[3:]
                #     key = target.add_atom(i, p, name=a, group=group)
                #     proof_atoms.append(-key if n else key)
                proof_keys[q].append(self.build_formula(proof, target))
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
