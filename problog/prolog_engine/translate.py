from problog.core import transform, ProbLogObject
from problog.clausedb import ClauseDB
from collections import defaultdict

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
            return '{}({})'.format(func, ','.join(handle_var(a) for a in args))
        else:
            if func == 'true':
                return 'true'
    else:
        return str(func)

class TranslatedProgram(ProbLogObject):

    def __init__(self, db):
        self.clauses = []
        self.db = db
        self.ad_heads = defaultdict(list)


    def to_str(self, node):
        ntype = type(node).__name__
        if ntype == 'conj':
            return ','.join(self.to_str(self.db.get_node(c)) for c in node.children)
        elif ntype == 'call':
            return handle_functor(node.functor, node.args)
        elif ntype == 'neg':
            return 'neg({})'.format(self.to_str(self.db.get_node(node.child)))
        return ntype+'_unhandled'

    def add_fact(self, node):
        self.clauses.append((handle_prob(node.probability), handle_functor(node.functor, node.args), ''))

    def add_clause(self, node):
        prob = node.probability if node.group is None else None
        self.clauses.append((handle_prob(prob), handle_functor(node.functor, node.args), self.to_str(self.db.get_node(node.child))))

    def add_choice(self, node):
        self.ad_heads[node.group].append((handle_prob(node.probability), handle_functor(node.functor, node.args)))

    def __str__(self):
        lines = ['ad([p({},{})],[{}]).'.format(*c) for c in self.clauses]
        for ad in self.ad_heads:
            lines.append('ad(['+','.join('p({},{})'.format(*head) for head in self.ad_heads[ad])+'],[]).')
        return '\n'.join(lines)

@transform(ClauseDB, TranslatedProgram)
def translate_clasusedb(db):

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