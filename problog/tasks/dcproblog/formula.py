from collections import defaultdict

from problog.formula import LogicFormula, atom
from problog.logic import Constant, Term, term2list

from .logic import Distribution, LogicVectorConstant, SymbolicConstant, RandomVariableConstant, RandomVariableComponentConstant, comparison_functors
from .algebra.algebra import Algebra


class DiGraph(object):
    def __init__(self):
        self.s = ""
        self.value_links = {}

class LogicFormulaHAL(LogicFormula):
    LABEL_OBSERVATION = "observation"
    LABEL_DQUERY = "density query"

    def __init__(self, density_values={}, density_names={}, free_variables={}, **kwargs):
        LogicFormula.__init__(self, **kwargs)
        self.density_names = density_names
        self.density_values = density_values



    def add_observation(self, name, key, value, keep_name=False):
        """Add an observation name.

        Same as ``add_name(name, key, self.OBSERVATION)``.

        :param name: name of the query
        :param key: key of the query node
        :param value: value of the observation
        """
        if value is None:
            self.add_name(name, key, self.LABEL_OBSERVATION, keep_name=keep_name)


    def clear_observation(self):
        """Remove all evidence."""
        self._names[self.LABEL_OBSERVATION] = {}


    def observation_all(self):
        """Get a list of all observation (including undetermined).

        :return: list of tuples (name, key, value) where value can be 1
        """
        observation = [x + (1,) for x in self.get_names(self.LABEL_OBSERVATION)]

        return observation

    def observation(self):
        """Get a list of all determined observations.

        :return: list of tuples (name, key) for obsveration
        """
        observation = self.get_names(self.LABEL_OBSERVATION)
        return list(observation)

    def add_dquery(self, name, key, keep_name=False):
        """Add a query name.

        Same as ``add_name(name, key, self.LABEL_DQUERY)``.

        :param name: name of the query
        :param key: key of the query node
        """
        self.add_name(name, key, self.LABEL_DQUERY, keep_name=keep_name)


    def clear_queries(self):
        """Remove all evidence."""
        self._names[self.LABEL_DQUERY] = {}


    def dqueries(self):
        """Get a list of all queries.

        :return: ``get_names(LABEL_DQUERY)``
        """
        return self.get_names(self.LABEL_DQUERY)



    def labeled(self):
        """Get a list of all query-like labels.

        :return:
        """
        result = []
        for name, node, label in self.get_names_with_label():
            if label not in (self.LABEL_NAMED, self.LABEL_EVIDENCE_POS, self.LABEL_EVIDENCE_NEG, self.LABEL_EVIDENCE_MAYBE, self.LABEL_OBSERVATION):
                result.append((name, node, label))
        return result

    def get_density_name(self, term, nid):
        if not term in self.density_names:
            self.density_names[term] = [nid]
            return (term,0)
        elif nid in self.density_names[term]:
            return (term, self.density_names[term].index(nid))
        else:
            self.density_names[term].append(nid)
            return (term, len(self.density_names[term])-1)

    def create_ast_representation(self, expression):
        #Create abstract syntax tree (AST) of algebraic expression
        if isinstance(expression, Constant):
            return self.create_ast_representation(expression.functor)
        elif isinstance(expression, (int,float)):
            return SymbolicConstant(expression, args=(), cvariables=set())
        elif isinstance(expression, bool):
            return SymbolicConstant(int(expression), args=(), cvariables=set())
        elif isinstance(expression, RandomVariableComponentConstant):
            return expression
        elif isinstance(expression, LogicVectorConstant):
            return expression
        elif isinstance(expression, SymbolicConstant):
            return expression
        elif expression==None:
            return SymbolicConstant(None,args=(),cvariables=set())
        elif expression.functor==".":
            expression = term2list(expression)
            symbolic_args = []
            for a in expression:
                symbolic_a = self.create_ast_representation(a)
                symbolic_args.append(symbolic_a)
            return SymbolicConstant("list", args=symbolic_args, cvariables=set())
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


    def add_atom(self, identifier, probability, group=None, name=None, source=None, cr_extra=True, is_extra=False):
        if probability is None and not self.keep_all:
            return self.TRUE
        elif isinstance(probability, SymbolicConstant) and probability.functor is None and not self.keep_all:
            return self.TRUE
        elif probability is False and not self.keep_all:
            return self.FALSE
        elif isinstance(probability, SymbolicConstant) and probability.functor is False and not self.keep_all:
            return self.FALSE
        elif probability != self.WEIGHT_NEUTRAL and self.semiring and \
                self.semiring.is_zero(self.semiring.value(probability)):
            return self.FALSE
        elif probability != self.WEIGHT_NEUTRAL and self.semiring and \
                self.semiring.is_one(self.semiring.value(probability)):
            return self.TRUE
        else:
            symbolic_expr = self.create_ast_representation(probability)
            atom = self._create_atom(identifier, symbolic_expr, group=group, name=name, source=source, is_extra=is_extra)

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

            return node_id


    def __str__(self):
        s = '\n'.join('%s: %s' % (i, n) for i, n, t in self)
        f = True
        for q in self.queries():
            if f:
                f = False
                s += '\nQueries : '
            s += '\n* ' +  "{} : {} [{}]".format(q[0],q[1], self.LABEL_QUERY)
        f = True
        for q in self.dqueries():
            if f:
                f = False
                s += '\nDensity Queries : '
            s += '\n* ' +  "{} : {} ".format(q[0][0],q[0][1])
        f = True
        for q in self.evidence():
            if f:
                f = False
                s += '\nEvidence : '
            s += '\n* %s : %s' % q
        f = True
        for o in self.observation():
            if f:
                f = False
                s += '\nObservation : '
            s += '\n* ' +  "{}={} : {}".format(o[0][0],o[0][1],o[1])
        f = True
        for c in self.constraints():
            if c.is_nontrivial():
                if f:
                    f = False
                    s += '\nConstraints : '
                s += '\n* ' + str(c)


        return s + '\n'


    def functions_to_dot(self, not_as_node=True, nodeprops=None):
        """Write out in GraphViz (dot) format.

        :param not_as_node: represent negation as a node
        :param nodeprops: additional properties for nodes
        :return: string containing dot representation
        """
        s = self._sdd_functions_to_dot(not_as_node=True, nodeprops=None)
        return s

    def functions_to_dot(self, not_as_node=True, nodeprops=None):

        if nodeprops is None:
            nodeprops = {}

        not_as_edge = not not_as_node

        # Keep track of mutually disjunctive nodes.
        clusters = defaultdict(list)

        queries = set([(name, node) for name, node, label in self.get_names_with_label()])
        for i, n, t in self:
            if n.name is not None:
                queries.add((n.name, i))

        # Keep a list of introduced not nodes to prevent duplicates.
        negative = set([])
        s = DiGraph()
        s.s += 'digraph GP {\n'
        for index, node, nodetype in self:
            prop = nodeprops.get(index, '')
            if prop:
                prop = ',' + prop
            if nodetype == 'conj':
                s.s += '{index} [label="AND", shape="box", style="filled", fillcolor="white"{prop}];\n'.format(index=index, prop=prop)
                for c in node.children:
                    opt = ''
                    if c < 0 and c not in negative and not_as_node:
                        s.s += '{c} [label="NOT"];\n'.format(c=c)
                        s.s += '{cp} -> {cn};\n'.format(cp=c, cn=-c)
                        negative.add(c)

                    if c < 0 and not_as_edge:
                        opt = '[arrowhead="odotnormal"]'
                        c = -c
                    if c != 0:
                        s.s += '{index} -> {c}{opt};\n'.format(index=index, c=c, opt=opt)
            elif nodetype == 'disj':
                s.s += '{index} [label="OR", shape="diamond", style="filled", fillcolor="white"{prop}];\n '.format(index=index, prop=prop)
                for c in node.children:
                    opt = ''
                    if c < 0 and c not in negative and not_as_node:
                        s.s += '{c} [label="NOT"];\n'.format(c=c)
                        s.s += '{cp} -> {cn};\n'.format(cp=c, cn=-c)
                        negative.add(c)
                    if c < 0 and not_as_edge:
                        opt = '[arrowhead="odotnormal"]'
                        c = -c
                    if c != 0:
                        s.s += '{index} -> {c}{opt};\n'.format(index=index, c=c, opt=opt)
            elif nodetype == 'atom':

                if node.probability == self.WEIGHT_NEUTRAL:
                    pass
                elif node.group is None:
                    s.s += '{index} [label="{probability}", shape="ellipse", style="filled", fillcolor="white"{prop}];\n'\
                        .format(index=index, probability=node.probability, prop=prop)
                else:
                    clusters[node.group].append('{index} [ shape="ellipse", label="{probability}", '
                        'style="filled", fillcolor="white" ];\n'.format(index=index, probability=node.probability))

                links, s = self.function_to_dot(s, node.probability)
                for l in links:
                    s.s += '{index} -> {density_node} [style="dotted"];\n'.format(index=index, density_node=l)
            else:
                raise TypeError("Unexpected node type: '{}'".format(nodetype))

        c = 0
        for cluster, text in clusters.items():
            if len(text) > 1:
                s.s += 'subgraph cluster_{c} {{ style="dotted"; color="red"; \n\t{join_text}\n }}\n'.format(c=c, join_text='\n\t'.join(text))
            else:
                s.s += text[0]
            c += 1

        q = 0
        for name, index in set(queries):
            if name.is_negated():
                pos_name = -name
                name = Term(pos_name.functor + '_aux', *pos_name.args)
            opt = ''
            if index is None:
                index = 'false'
                if not_as_node:
                    s.s += '{} [label="NOT"];\n'.format(index)
                    s.s += '{} -> {};\n' % (index, 0)
                elif not_as_edge:
                    opt = ', arrowhead="odotnormal"'
                if 0 not in negative:
                    s.s += '{} [label="true"];\n'.format(0)
                    negative.add(0)
            elif index < 0:  # and not index in negative :
                if not_as_node:
                    s.s += '{} [label="NOT"];\n'.format(index)
                    s.s += '{pindex} -> {nindex};\n'.format(pindex=index, nindex=-index)
                    negative.add(index)
                elif not_as_edge:
                    index = -index
                    opt = ', arrowhead="odotnormal"'
            elif index == 0 and index not in negative:
                s.s += '{} [label="true"];\n'.format(index)
                negative.add(0)
            s.s += 'q_{q} [ label="{name}", shape="plaintext" ];\n'.format(q=q, name=name)
            s.s += 'q_{q} -> {index} [style="dotted" {opt}];\n'.format(q=q, index=index, opt=opt)
            q += 1
        return s.s + '}'


    def function_to_dot(self, s, expression):
        if isinstance(expression, (int, float)):
            return (), s
        elif isinstance(expression,SymbolicConstant) and isinstance(expression.functor, (int, float)):
            return (), s
        elif isinstance(expression, SymbolicConstant) and expression.functor=="/":
            return (), s
        elif isinstance(expression, SymbolicConstant) and expression.functor=="list":
            links = ()
            for a in expression.args:
                l, s = self.function_to_dot(s, a)
                links += l

            return links, s
        elif isinstance(expression, SymbolicConstant) and str(expression.functor)=="real":
            return (), s
        elif isinstance(expression, SymbolicConstant):
            if expression in s.value_links:
                return (s.value_links[expression],), s
            else:
                links = ()
                for a in expression.args:
                    l, s = self.function_to_dot(s, a)
                    links += l

                # node_name = Algebra.name2str(expression.name)
                node_name = hash(str(expression))
                str_density_args = ",".join(list(map(str,expression.args)))
                str_density_functor = str(expression.functor)
                str_density = "{}({})".format(str_density_functor, str_density_args)
                graph = '{name} [label="{function}", shape="ellipse", style="filled", fillcolor="white"];\n'\
                    .format(name=node_name, function=str_density)
                s.s += graph
                for l in links:
                    s.s += '{index} -> {density_node} [style="dotted"];\n'.format(index=node_name, density_node=l)

                s.value_links[expression] = node_name
                return (node_name,), s
        elif expression.functor in comparison_functors:
            lhs_links, s = self.function_to_dot(s, expression.args[0])
            rhs_links, s = self.function_to_dot(s, expression.args[1])
            links = lhs_links+rhs_links
            return links, s
        elif isinstance(expression, ValueDimConstant):
            density_value = self.density_values[expression.functor[:-1]]
            links, s = self.function_to_dot(s, self.density_values[expression.functor[:-1]])
            # for l in links:
            #     s.s += '{index} -> {density_node} [style="dotted"];\n'.format(index=hash(str_expression), density_node=l)
            return links, s
        elif isinstance(expression, SymbolicConstant):
            links = ()
            for a in expression.args:
                l, s = self.function_to_dot(s, a)
                links += l
            str_expression_args = ",".join(list(map(str,expression.args)))
            str_expression_functor = str(expression.functor)
            str_expression = "{}({})".format(str_expression_functor, str_expression_args)
            graph = '{name} [label="{expression}", shape="ellipse", style="filled", fillcolor="white"];\n'\
                .format(name=hash(str_expression), expression=str_expression)
            s.s += graph
            for l in links:
                s.s += '{index} -> {density_node} [style="dotted"];\n'.format(index=hash(str_expression), density_node=l)
            return (hash(str_expression),), s
