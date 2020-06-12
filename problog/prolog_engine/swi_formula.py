from collections import defaultdict
from problog.formula import LogicFormula
from functools import lru_cache
from problog.logic import Term

class Node(object):

    def __init__(self, children=None):
        self.children = [] if children is None else children

    def add_child(self, child):
        # if child != self:
        self.children.append(child)

    def simplify(self, _):
        return self

class Atom(Node):

    def __init__(self, children=None):
        super().__init__(children)

    def simplify(self, nodes):
        if len(self.children) == 1:
            child = self.children[0]
            if type(child) is Term:
                return nodes[child].simplify(nodes)
            else:
                return child.simplify(nodes)
        else:
            return Or(self.children)

class T(Node):
    def __init__(self):
        super().__init__(None)

class F(Node):
    def __init__(self):
        super().__init__(None)

class Or(Node):

    def __init__(self, children=None):
        super().__init__(children)

    def simplify(self, _):
        if len(self.children) == 0:
            return F()
        elif len(self.children) == 1:
            return self.children[0].simplify(nodes)
        return self


class And(Node):

    def __init__(self, children=None):
        super().__init__(children)

    def simplify(self, nodes):
        self.children = [c for c in self.children if type(c.simplify(nodes)) != T]
        if len(self.children) == 0:
            return T()
        elif len(self.children) == 1:
            return self.children[0].simplify(nodes)
        return self


class Not(Node):

    def __init__(self, children=None):
        super().__init__(children)


class Fact(Node):

    def __init__(self, fact):
        super().__init__(None)
        self.fact = fact


class SWI_Formula(object):

    def __init__(self):
        self.atoms = defaultdict(Atom)
        self.names = set()
    # def get_children(self, term):
    #     if term.functor == ',':
    #         children = []
    #         for c in term.args:
    #             children += self.get_children(c)
    #         return children
    #     # elif term.functor == 'neg':
    #     #     return self.get_children(term.args[0])
    #     else:
    #         return [term]

    def get_children(self, term):
        if term.functor == ',':
            return And([self.get_children(a) for a in term.args])
        elif term.functor == ';':
            return Or([self.get_children(a) for a in term.args])
        elif term.functor == 'neg':
            return Not([self.get_children(a) for a in term.args])
        else:
            if term.functor == 'true':
                return T()
            elif term.functor == 'fail':
                return F()
            return self.atoms[term]
    @lru_cache(maxsize=None)
    def add_proof(self, p):
        print('adding ',p)
        if p.functor == '::':  # If it's a fact
            n = self.atoms[p.args[2]]
            n.add_child(Fact(p))
        elif p.functor == ':-':  # If it's a clause

            n = self.atoms[p.args[0]]
            child_node = self.get_children(p.args[1])
            n.add_child(child_node)
            #     n2 = Node(Node.AND)
            #     for c in self.get_children(p.args[1]):
            #         if c.functor == 'neg':
            #
            #             c_n = Node(Node.NOT)
            #             c_n.children = self.nodes[self.get_children(c.args[0])[0]]
            #             n2.add_child(c_n)
            #         elif c.functor == ';':
            #             c_n = Node(Node.OR)
            #             else:
            #             n2.add_child(self.nodes[c])
            #     n.add_child(n2)
        else:
            raise ValueError('Unhandled node {}'.format(p))
        # return n

    # def simplify(self):
    #     modified = True
    #     while modified:
    #         modified = False
    #         for k in self.nodes:
    #             modified |= self.nodes[k].simplify()



    def to_formula(self, target):
        self.atoms = dict(self.atoms)
        try:
            queue = ProcessQueue(self.atoms)
            names = [(queue[self.atoms[key].simplify(self.atoms)], key, label) for key, label in self.names]
            while len(queue) > 0:
                node = queue.pop()
                print('node: ', node)
                print('children: ', node.children)
                if type(node) is And:
                    target.add_and([queue[c.simplify(self.atoms)] for c in node.children], queue[node])
                elif type(node) is Or:
                    target.add_or([queue[c.simplify(self.atoms)] for c in node.children], queue[node])
                elif type(node) is Fact:
                    target.add_atom(queue[node], node.fact.args[1])

            for key, name, label in names:
                target.add_name(name, key, label)
            return target
        except KeyError as e:
            from problog.engine import UnknownClause
            raise UnknownClause(str(e), 0)

    # def to_formula(self, target):
    #     # self.simplify()
    #     # for name, key, _ in target.get_names_with_label():
    #     #     1 (name, key)
    #     #     d[name] = key
    #     nodes, names = self.enumerate()
    #     for i, node in enumerate(nodes):
    #         node_type = node[0]
    #         if node_type == 'fact':
    #             target.add_atom(i, node[2].args[1])
    #         elif node_type == 'and':
    #             target.add_and(node[2], i)
    #         elif node_type == 'or':
    #             target.add_or(node[2], i)
    #         else:
    #             raise ValueError(node_type)
    #     for key, name, label in names:
    #         target.add_name(name, key, label)

class ProcessQueue(object):

    def __init__(self, atoms):
        # self.processed = []
        self.to_process = []
        self.indices = dict()
        self.i = 1
        self.atoms = atoms

    def __getitem__(self, item):
        if type(item) is T:
            return 0

        if type(item) is F:
            return None
        negated = False
        if type(item) is Not:
            negated = True
            item = item.children[0].simplify(self.atoms)
            # item = self.atoms[item.children[0]].simplify(self.atoms)
        try:
            i = self.indices[item]
        except KeyError:
            self.to_process.append(item)
            self.indices[item] = self.i
            i = self.i
            self.i += 1
        return -i if negated else i

    def __len__(self):
        return len(self.to_process)

    def pop(self):
        v = self.to_process.pop(0)
        # self.processed.append(v)
        return v