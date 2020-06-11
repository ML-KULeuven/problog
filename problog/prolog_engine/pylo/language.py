from abc import ABC
from typing import Sequence, Union


class Term(ABC):

    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._name == other._name
        else:
            return False

    def __hash__(self):
        return hash(f"{type(self)}:{self._name}")

    def __repr__(self):
        return str(self._name)


class Constant(Term):

    def __init__(self, name):
        if not isinstance(name, (int, float)):
            assert name.islower(), f"Constants should be name with lowercase {name}"
        super().__init__(name)


class Variable(Term):

    def __init__(self, name):
        assert name.isupper(), f"Variables should be name uppercase {name}"
        super().__init__(name)


class Functor:

    def __init__(self, name: str, arity: int):
        self._name = name
        self._arity = arity

    def get_name(self):
        return self._name

    def get_arity(self):
        return self._arity

    def __eq__(self, other):
        if isinstance(other, Functor):
            return self._name == other._name and self._arity == other._arity
        else:
            return False

    def __hash__(self):
        return hash(f"func:{self._name}/{self._arity}")

    def __repr__(self):
        return str(self._name)

    def __call__(self, *args):
        args_to_use = []
        global global_context
        for elem in args:
            if isinstance(elem, str) and elem.islower():
                args_to_use.append(global_context.get_constant(elem))
            elif isinstance(elem, str) and elem.isupper():
                args_to_use.append(global_context.get_variable(elem))
            elif isinstance(elem, (Constant, Variable, Structure, List, int, float)):
                args_to_use.append(elem)
            else:
                raise Exception(f"don't know how to convert {type(elem)} {elem} to term")

        return Structure(self, args_to_use)


class Structure(Term):

    def __init__(self, functor: Functor, args: Sequence[Union[Term, int, float]]):
        self._functor = functor
        self._args = args
        super().__init__(name=functor.get_name())

    def get_functor(self):
        return self._functor

    def get_arguments(self):
        return self._args

    def __eq__(self, other):
        if isinstance(other, Structure):
            if self._functor != other._functor or len(self._args) != len(other._args):
                return False
            else:
                return all([x == y for x, y in zip(self._args, other._args)])
        else:
            return False

    def __repr__(self):
        return f"{self._name}({','.join([str(x) for x in self._args])})"

    def __hash__(self):
        return hash(str(self))


list_func = Functor(".", 2)


class List(Structure):

    def __init__(self, elements: Sequence[Union[Term, int, float]]):
        argsToUse = []
        for elem in elements:
            if isinstance(elem, str) and elem[0].isupper():
                argsToUse.append(global_context.get_variable(elem))
            elif isinstance(elem, str) and elem[0].islower():
                argsToUse.append(global_context.get_constant(elem))
            elif isinstance(elem, (Constant, Variable, Structure, Predicate, List)):
                argsToUse.append(elem)
            elif isinstance(elem, (int, float)):
                argsToUse.append(elem)
            else:
                raise Exception(f"Predicate:Don't know how to convert {type(elem)} {elem} to term")
        super(List, self).__init__(list_func, argsToUse)

    def __repr__(self):
        return f"[{','.join([str(x) for x in self._args])}]"


class Predicate:

    def __init__(self, name: str, arity: int):
        self._name = name
        self._arity = arity

    def get_name(self):
        return self._name

    def get_arity(self):
        return self._arity

    def __eq__(self, other):
        if isinstance(other, Predicate):
            return self._name == other._name and self._arity == other._arity
        else:
            return False

    def __hash__(self):
        return hash(f"pred:{self._name}/{self._arity}")

    def __repr__(self):
        return str(self._name)

    def __call__(self, *args):
        argsToUse = []
        # global global_context
        for elem in args:
            if isinstance(elem, str) and elem[0].isupper():
                argsToUse.append(global_context.get_variable(elem))
            elif isinstance(elem, str) and elem[0].islower():
                print("CONSTANT ELEM:" + elem)
                argsToUse.append(global_context.get_constant(elem))
            elif isinstance(elem, (Constant, Literal, Variable, Structure, Predicate)):
                argsToUse.append(elem)
            elif isinstance(elem, (int, float)):
                argsToUse.append(elem)
            else:
                print(type(elem))
                print(elem)
                raise Exception(f"Predicate:Don't know how to convert {type(elem)} {elem} to term")

        return Literal(self, argsToUse)


class Literal:

    def __init__(self, predicate: Predicate, args: Sequence[Union[Term, int, float]]):
        self._predicate = predicate
        self._args = args

    def get_predicate(self):
        return self._predicate

    def get_arguments(self):
        return self._args

    def __and__(self, other: Union["Literal", "Negation"]):
        return Conj(self, other)

    def __le__(self, other: Union["Literal", "Negation", "Conj"]):
        if isinstance(other, (Literal, Negation)):
            return Clause(self, Conj(other))
        else:
            return Clause(self, other)

    def __eq__(self, other):
        if isinstance(other, Literal):
            return self._predicate == other._predicate and\
                    len(self._args) == len(other._args) and\
                    all([x == y for x, y in zip(self._args, other._args)])
        else:
            return False

    def __repr__(self):
        return f"{self._predicate}({','.join([str(x) for x in self._args])})"

    def __hash__(self):
        return hash(str(self))


class Negation:

    def __init__(self, literal: Literal):
        self._lit = literal

    def get_literal(self):
        return self._lit

    def __and__(self, other: Union["Literal", "Negation"]):
        return Conj(self, other)

    def __eq__(self, other):
        if isinstance(other, Negation):
            return self._lit == other._lit
        else:
            return False

    def __repr__(self):
        return f"\\+ {self._lit}"


class Conj:

    def __init__(self, *lits):
        self._lits = list(lits)

    def get_literals(self):
        return self._lits

    def __repr__(self):
        return ",".join([str(x) for x in self._lits])

    def __eq__(self, other):
        if isinstance(other, Conj):
            return all([x == y for x, y in zip(self._lits, other._lits)])
        else:
            return False

    def __hash__(self):
        return hash(str(self))

    def __and__(self, other: Union["Literal", "Negation", "Conj"]):
        if isinstance(other, (Literal, Negation)):
            self._lits += [other]
        elif isinstance(other, Conj):
            self._lits += other.get_literals()
        else:
            raise Exception(f"Don't know how to add object of type {type(other)} to Conj")


class Clause:

    def __init__(self, head: Literal, body: Conj):
        self._head = head
        self._body = body

    def get_head(self):
        return self._head

    def get_body(self):
        return self._body

    def __and__(self, other: Union["Literal", "Negation", "Conj"]):
        if isinstance(other, (Literal, Negation)):
            self._body & other
        elif isinstance(other, Conj):
            self._body = Conj(self._body.get_literals() + other.get_literals())
        else:
            raise Exception(f"Don't know how to add bject of type {type(other)} to Clause")

    def __repr__(self):
        return f"{self._head} :- {self._body}"


class Context:

    def __init__(self):
        self._consts = {}
        self._vars = {}
        self._predicates = {}
        self._functors = {}

    def get_constant(self, name: str):
        if name not in self._consts:
            self._consts[name] = Constant(name)

        return self._consts[name]

    def get_variable(self, name: str):
        if name not in self._vars:
            self._vars[name] = Variable(name)

        return self._vars[name]

    def get_functor(self, name: str, arity: int):
        if name not in self._functors:
            self._functors[name] = {}

        if arity not in self._functors[name]:
            self._functors[name][arity] = Functor(name, arity)

        return self._functors[name][arity]

    def get_predicate(self, name: str, arity: int):
        if name not in self._predicates:
            self._predicates[name] = {}

        if arity not in self._predicates[name]:
            self._predicates[name][arity] = Predicate(name, arity)

        return self._predicates[name][arity]

    def get_symbol(self, name, **kwargs):
        if name in self._consts:
            return self._consts[name]
        elif name in self._functors:
            if 'arity' in kwargs:
                return self._functors[name][kwargs['arity']]
            else:
                ks = list(self._functors[name].keys())
                if len(ks) > 1:
                    raise Exception(f"Cannot uniquely identify functor symbol {name}")
                return self._functors[name][ks[0]]
        elif name in self._predicates:
            if 'arity' in kwargs:
                return self._predicates[name][kwargs['arity']]
            else:
                ks = list(self._predicates[name].keys())
                if len(ks) > 1:
                    raise Exception(f"Cannot uniquely identify predicate symbol {name}")
                return self._predicates[name][ks[0]]
        else:
            raise Exception(f"Cannot identify symbol {name}")


global_context = Context()