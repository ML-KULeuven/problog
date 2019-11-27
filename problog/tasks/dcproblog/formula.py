from collections import defaultdict

from problog.formula import LogicFormula, LogicNNF
from problog.logic import Constant, Term, term2list
from .evaluator import FormulaEvaluatorHAL
from .logic import (
    pdfs,
    SymbolicConstant,
    ValueDimConstant,
    ValueExpr,
    comparison_functors,
)


class DiGraph(object):
    def __init__(self):
        self.s = ""
        self.value_links = {}


class LogicNNFHAL(LogicNNF):
    """A propositional formula in NNF form (i.e. only negation on facts)."""

    def __init__(self, auto_compact=True, **kwdargs):
        LogicNNF.__init__(self, auto_compact=True, **kwdargs)
        self._str2name = {}

    def _create_evaluator(self, semiring, weights, **kwargs):
        return FormulaEvaluatorHAL(self, semiring, weights)

    def evaluate(
        self, index=None, semiring=None, evidence=None, weights=None, **kwargs
    ):
        """Evaluate a set of nodes.

        :param index: node to evaluate (default: all queries)
        :param semiring: use the given semiring
        :param evidence: use the given evidence values (overrides formula)
        :param weights: use the given weights (overrides formula)
        :return: The result of the evaluation expressed as an external value of the semiring. \
         If index is ``None`` (all queries) then the result is a dictionary of name to value.
        """

        evaluator = self.get_evaluator(semiring, evidence, weights, **kwargs)
        # evaluator._fact_weights = {}
        if index is None:
            result = {}
            # Probability of query given evidence

            # interrupted = False
            for name, node, label in evaluator.formula.labeled():
                w = evaluator.evaluate(node)
                result[name] = w
        else:
            result = evaluator.evaluate(index)

        return result

    def extract_weights(self, semiring, weights=None):
        """Extracts the positive and negative weights for all atoms in the data structure.

        :param semiring: semiring that determines the interpretation of the weights
        :param weights: dictionary of { node name : weight } that overrides the builtin weights
        :returns: dictionary { key: (positive weight, negative weight) }
        :rtype: dict[int, tuple[any]]

        Atoms with weight set to neutral will get weight ``(semiring.one(), semiring.one())``.

        If the weights argument is given, it completely replaces the formula's weights.

        All constraints are applied to the weights.
        """

        if weights is None:
            weights = self.get_weights()
        else:
            oweights = dict(self.get_weights().items())
            oweights.update({self.get_node_by_name(n): v for n, v in weights.items()})
            weights = oweights

        result = {}
        for n, w in weights.items():
            if hasattr(self, "get_name"):
                name = self.get_name(n)
            else:
                name = n
            if w == self.WEIGHT_NEUTRAL and type(self.WEIGHT_NEUTRAL) == type(w):
                result[n] = semiring.one(), semiring.one()
            elif w == False:
                result[n] = semiring.false(name)
            elif w is None:
                result[n] = semiring.true(name)
            else:
                result[n] = (
                    semiring.pos_value(w, name, index=n),
                    semiring.neg_value(w, name, index=n),
                )

        for c in self.constraints():
            c.update_weights(result, semiring)

        return result


class LogicFormulaHAL(LogicFormula):
    def __init__(self, **kwargs):
        LogicFormula.__init__(self, **kwargs)
        self.density_nodes = {}
        self.density_indices = set()
        self.density_node_body = {}
        self.density_names = {}
        self.density_values = {}

        self.free_variables = set()
        self.density_queries = {}

    def is_density(self, index):
        if index in self.density_indices:
            return True
        else:
            return False

    def get_term(self, atom):
        if atom.name.functor == "choice":
            term = atom.name.args[2]
        else:
            term = atom.name
        return term

    def get_density_name(self, term, nid):
        if not term in self.density_names:
            self.density_names[term] = [nid]
            return (term, 0)
        elif nid in self.density_names[term]:
            return (term, self.density_names[term].index(nid))
        else:
            self.density_names[term].append(nid)
            return (term, len(self.density_names[term]) - 1)

    def bookkeep_density_node(self, atom, node_id):
        self.density_indices.add(node_id)
        term = self.get_term(atom)
        if term in self.density_nodes:
            self.density_nodes[term] += (node_id,)
        else:
            self.density_nodes[term] = (node_id,)

    def create_ast_representation(self, expression):
        # Create abstract syntax tree (AST) of algebraic expression
        if isinstance(expression, Constant):
            return self.create_ast_representation(expression.functor)
        elif isinstance(expression, (int, float)):
            return SymbolicConstant(expression, args=(), cvariables=set())
        elif isinstance(expression, SymbolicConstant):
            return expression
        elif isinstance(expression, ValueDimConstant):
            return expression
        elif isinstance(expression, bool):
            return SymbolicConstant(int(expression), args=(), cvariables=set())
        elif expression == None:
            return SymbolicConstant(None, args=(), cvariables=set())
        elif expression.functor == ".":
            expression = term2list(expression)
            symbolic_args = []
            for a in expression:
                symbolic_a = self.create_ast_representation(a)
                symbolic_args.append(symbolic_a)
            return SymbolicConstant("list", args=symbolic_args, cvariables=set())

        elif isinstance(expression, Constant):
            return self.create_ast_representation(expression.functor)
        elif isinstance(expression, Term):
            functor = expression.functor.strip("'")
            symbolic_args = []
            cvariables = set()
            for a in expression.args:
                symbolic_a = self.create_ast_representation(a)
                symbolic_args.append(symbolic_a)
                cvariables = cvariables.union(symbolic_a.cvariables)
            return SymbolicConstant(functor, args=symbolic_args, cvariables=cvariables)
        else:
            assert False

    def add_atom(
        self,
        identifier,
        probability,
        group=None,
        name=None,
        source=None,
        cr_extra=True,
        is_extra=False,
    ):
        if probability is None and not self.keep_all:
            return self.TRUE
        elif (
            isinstance(probability, SymbolicConstant)
            and probability.functor is None
            and not self.keep_all
        ):
            return self.TRUE
        elif probability is False and not self.keep_all:
            return self.FALSE
        elif (
            isinstance(probability, SymbolicConstant)
            and probability.functor is False
            and not self.keep_all
        ):
            return self.FALSE
        # elif probability != self.WEIGHT_NEUTRAL and self.semiring and \
        #         self.semiring.is_zero(self.semiring.value(probability)):
        #     return self.FALSE
        # elif probability != self.WEIGHT_NEUTRAL and self.semiring and \
        #         self.semiring.is_one(self.semiring.value(probability)):
        #     return self.TRUE
        else:
            symbolic_expr = self.create_ast_representation(probability)
            is_density = False
            if symbolic_expr.functor in pdfs and not isinstance(
                symbolic_expr, ValueExpr
            ):
                atom = self._create_atom(
                    identifier,
                    symbolic_expr,
                    group=group,
                    name=name,
                    source=source,
                    is_extra=is_extra,
                )
                is_density = True
            else:
                atom = self._create_atom(
                    identifier,
                    symbolic_expr,
                    group=group,
                    name=name,
                    source=source,
                    is_extra=is_extra,
                )

            length_before = len(self._nodes)
            node_id = self._add(atom, key=identifier)

            self.get_weights()[node_id] = symbolic_expr
            if name is not None:
                self.add_name(name, node_id, self.LABEL_NAMED)

            if len(self._nodes) != length_before:
                # The node was not reused?
                self._atomcount += 1
                # TODO if the next call return 0 or None, the node is still added?
                node_id = self._add_constraint_me(group, node_id, cr_extra=cr_extra)

            if is_density:
                self.bookkeep_density_node(atom, node_id)
            return node_id

    def functions_to_dot(self, not_as_node=True, nodeprops=None):
        """Write out in GraphViz (dot) format.

        :param not_as_node: represent negation as a node
        :param nodeprops: additional properties for nodes
        :return: string containing dot representation
        """
        s = self._sdd_functions_to_dot(not_as_node=True, nodeprops=None)
        return s

    def functions_to_dot(self, not_as_node=True, nodeprops=None):
        density_nodes = []

        if nodeprops is None:
            nodeprops = {}

        not_as_edge = not not_as_node

        # Keep track of mutually disjunctive nodes.
        clusters = defaultdict(list)

        queries = set(
            [(name, node) for name, node, label in self.get_names_with_label()]
        )
        for i, n, t in self:
            if n.name is not None:
                queries.add((n.name, i))

        # Keep a list of introduced not nodes to prevent duplicates.
        negative = set([])
        s = DiGraph()
        s.s += "digraph GP {\n"
        for index, node, nodetype in self:
            prop = nodeprops.get(index, "")
            if prop:
                prop = "," + prop
            if nodetype == "conj":
                s.s += '{index} [label="AND", shape="box", style="filled", fillcolor="white"{prop}];\n'.format(
                    index=index, prop=prop
                )
                for c in node.children:
                    opt = ""
                    if c < 0 and c not in negative and not_as_node:
                        s.s += '{c} [label="NOT"];\n'.format(c=c)
                        s.s += "{cp} -> {cn};\n".format(cp=c, cn=-c)
                        negative.add(c)

                    if c < 0 and not_as_edge:
                        opt = '[arrowhead="odotnormal"]'
                        c = -c
                    if c != 0:
                        s.s += "{index} -> {c}{opt};\n".format(
                            index=index, c=c, opt=opt
                        )
            elif nodetype == "disj":
                s.s += '{index} [label="OR", shape="diamond", style="filled", fillcolor="white"{prop}];\n '.format(
                    index=index, prop=prop
                )
                for c in node.children:
                    opt = ""
                    if c < 0 and c not in negative and not_as_node:
                        s.s += '{c} [label="NOT"];\n'.format(c=c)
                        s.s += "{cp} -> {cn};\n".format(cp=c, cn=-c)
                        negative.add(c)
                    if c < 0 and not_as_edge:
                        opt = '[arrowhead="odotnormal"]'
                        c = -c
                    if c != 0:
                        s.s += "{index} -> {c}{opt};\n".format(
                            index=index, c=c, opt=opt
                        )
            elif nodetype == "atom":

                if node.probability == self.WEIGHT_NEUTRAL:
                    pass
                elif node.group is None:
                    s.s += '{index} [label="{probability}", shape="ellipse", style="filled", fillcolor="white"{prop}];\n'.format(
                        index=index, probability=node.probability, prop=prop
                    )
                else:
                    clusters[node.group].append(
                        '{index} [ shape="ellipse", label="{probability}", '
                        'style="filled", fillcolor="white" ];\n'.format(
                            index=index, probability=node.probability
                        )
                    )

                links, s = self.function_to_dot(s, node.probability)
                for l in links:
                    s.s += '{index} -> {density_node} [style="dotted"];\n'.format(
                        index=index, density_node=l
                    )
            else:
                raise TypeError("Unexpected node type: '{}'".format(nodetype))

        c = 0
        for cluster, text in clusters.items():
            if len(text) > 1:
                s.s += 'subgraph cluster_{c} {{ style="dotted"; color="red"; \n\t{join_text}\n }}\n'.format(
                    c=c, join_text="\n\t".join(text)
                )
            else:
                s.s += text[0]
            c += 1

        q = 0
        for name, index in set(queries):
            if name.is_negated():
                pos_name = -name
                name = Term(pos_name.functor + "_aux", *pos_name.args)
            opt = ""
            if index is None:
                index = "false"
                if not_as_node:
                    s.s += '{} [label="NOT"];\n'.format(index)
                    s.s += "{} -> {};\n" % (index, 0)
                elif not_as_edge:
                    opt = ', arrowhead="odotnormal"'
                if 0 not in negative:
                    s.s += '{} [label="true"];\n'.format(0)
                    negative.add(0)
            elif index < 0:  # and not index in negative :
                if not_as_node:
                    s.s += '{} [label="NOT"];\n'.format(index)
                    s.s += "{pindex} -> {nindex};\n".format(pindex=index, nindex=-index)
                    negative.add(index)
                elif not_as_edge:
                    index = -index
                    opt = ', arrowhead="odotnormal"'
            elif index == 0 and index not in negative:
                s.s += '{} [label="true"];\n'.format(index)
                negative.add(0)
            s.s += 'q_{q} [ label="{name}", shape="plaintext" ];\n'.format(
                q=q, name=name
            )
            s.s += 'q_{q} -> {index} [style="dotted" {opt}];\n'.format(
                q=q, index=index, opt=opt
            )
            q += 1
        return s.s + "}"

    def function_to_dot(self, s, expression):
        if isinstance(expression, (int, float)):
            return (), s
        elif isinstance(expression, SymbolicConstant) and isinstance(
            expression.functor, (int, float)
        ):
            return (), s
        elif isinstance(expression, SymbolicConstant) and expression.functor == "/":
            return (), s
        elif isinstance(expression, SymbolicConstant) and expression.functor == "list":
            links = ()
            for a in expression.args:
                l, s = self.function_to_dot(s, a)
                links += l

            return links, s
        elif isinstance(expression, ValueExpr) and str(expression.functor) == "real":
            return (), s
        elif isinstance(expression, ValueExpr):
            if expression in s.value_links:
                return (s.value_links[expression],), s
            else:
                links = ()
                for a in expression.args:
                    l, s = self.function_to_dot(s, a)
                    links += l

                # node_name = Algebra.name2str(expression.name)
                node_name = hash(expression.name)
                str_density_args = ",".join(list(map(str, expression.args)))
                str_density_functor = str(expression.functor)
                str_density = "{}({})".format(str_density_functor, str_density_args)
                graph = '{name} [label="{function}", shape="ellipse", style="filled", fillcolor="white"];\n'.format(
                    name=node_name, function=str_density
                )
                s.s += graph
                for l in links:
                    s.s += '{index} -> {density_node} [style="dotted"];\n'.format(
                        index=node_name, density_node=l
                    )

                s.value_links[expression] = node_name
                return (node_name,), s
        elif expression.functor in comparison_functors:
            lhs_links, s = self.function_to_dot(s, expression.args[0])
            rhs_links, s = self.function_to_dot(s, expression.args[1])
            links = lhs_links + rhs_links
            return links, s
        elif isinstance(expression, ValueDimConstant):
            density_value = self.density_values[expression.functor[:-1]]
            links, s = self.function_to_dot(
                s, self.density_values[expression.functor[:-1]]
            )
            # for l in links:
            #     s.s += '{index} -> {density_node} [style="dotted"];\n'.format(index=hash(str_expression), density_node=l)
            return links, s
        elif isinstance(expression, SymbolicConstant):
            links = ()
            for a in expression.args:
                l, s = self.function_to_dot(s, a)
                links += l
            str_expression_args = ",".join(list(map(str, expression.args)))
            str_expression_functor = str(expression.functor)
            str_expression = "{}({})".format(
                str_expression_functor, str_expression_args
            )
            graph = '{name} [label="{expression}", shape="ellipse", style="filled", fillcolor="white"];\n'.format(
                name=hash(str_expression), expression=str_expression
            )
            s.s += graph
            for l in links:
                s.s += '{index} -> {density_node} [style="dotted"];\n'.format(
                    index=hash(str_expression), density_node=l
                )
            return (hash(str_expression),), s
