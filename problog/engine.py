"""
problog.engine - Grounding engine
---------------------------------

Grounding engine to transform a ProbLog program into a propositional formula.

..
    Part of the ProbLog distribution.

    Copyright 2015 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
from __future__ import print_function

import logging
import os

from collections import defaultdict, namedtuple

from .program import LogicProgram
from .logic import *
from .formula import LogicFormula
from .engine_unify import *

from .core import transform
from .errors import GroundingError
from .util import Timer, OrderedSet


@transform(LogicProgram, LogicFormula)
def ground(model, target=None, queries=None, evidence=None, propagate_evidence=False,
           labels=None, engine=None, **kwdargs):
    """Ground a given model.

    :param model: logic program to ground
    :type model: LogicProgram
    :param target: formula in which to store ground program
    :type target: LogicFormula
    :param queries: list of queries to override the default
    :param evidence: list of evidence atoms to override the default
    :return: the ground program
    :rtype: LogicFormula
    """
    if engine is None:
        engine = DefaultEngine(**kwdargs)
    return engine.ground_all(model, target, queries=queries, evidence=evidence,
                                               propagate_evidence=propagate_evidence, labels=labels)


class GenericEngine(object):  # pragma: no cover
    """Generic interface to a grounding engine."""

    def prepare(self, db):
        """Prepare the given database for querying.
        Calling this method is optional.

       :param db: logic program
       :returns: logic program in optimized format where builtins are initialized and directives \
       have been evaluated
        """
        raise NotImplementedError('GenericEngine.prepare is an abstract method.')

    def query(self, db, term):
        """Evaluate a query without generating a ground program.

       :param db: logic program
       :param term: term to query; variables should be represented as None
       :returns: list of tuples of argument for which the query succeeds.
        """
        raise NotImplementedError('GenericEngine.query is an abstract method.')

    def ground(self, db, term, target=None, label=None):
        """Ground a given query term and store the result in the given ground program.

       :param db: logic program
       :param term: term to ground; variables should be represented as None
       :param target: target logic formula to store grounding in (a new one is created if none is \
       given)
       :param label: optional label (query, evidence, ...)
       :returns: logic formula (target if given)
        """
        raise NotImplementedError('GenericEngine.ground is an abstract method.')

    def ground_all(self, db, target=None, queries=None, evidence=None):
        """Ground all queries and evidence found in the the given database.

       :param db: logic program
       :param target: logic formula to ground into
       :param queries: list of queries to evaluate instead of the ones in the logic program
       :param evidence: list of evidence to evaluate instead of the ones in the logic program
       :returns: ground program
        """
        raise NotImplementedError('GenericEngine.ground_all is an abstract method.')


class ClauseDBEngine(GenericEngine):
    """Parent class for all Python ClauseDB-based engines."""

    UNKNOWN_ERROR = 0
    UNKNOWN_FAIL = 1

    def __init__(self, builtins=True, **kwdargs):
        self.__builtin_index = {}
        self.__builtins = []
        self.__externals = {}

        self._unique_number = 0
        self.unknown = self.UNKNOWN_ERROR

        if builtins:
            self.load_builtins()

        self.functions = {}
        self.args = kwdargs.get('args')

    def load_builtins(self):
        """Load default builtins."""
        raise NotImplementedError("ClauseDBEngine.loadBuiltIns is an abstract function.")

    def get_builtin(self, index):
        """Get builtin's evaluation function based on its identifier.
       :param index: index of the builtin
       :return: function that evaluates the builtin
        """
        real_index = -(index + 1)
        return self.__builtins[real_index]

    def add_builtin(self, predicate, arity, function):
        """Add a builtin.

        :param predicate: name of builtin predicate
        :param arity: arity of builtin predicate
        :param function: function to execute builtin
        """
        sig = '%s/%s' % (predicate, arity)
        self.__builtin_index[sig] = -(len(self.__builtins) + 1)
        self.__builtins.append(function)

    def get_builtins(self):
        """Get the list of builtins."""
        return self.__builtin_index

    def prepare(self, db):
        """Convert given logic program to suitable format for this engine.
        Calling this method is optional, but it allows to perform multiple operations on the same \
        database.
        This also executes any directives in the input model.

        :param db: logic program to prepare for evaluation
        :type db: LogicProgram
        :return: logic program in a suitable format for this engine
        :rtype: ClauseDB
        """
        result = ClauseDB.createFrom(db, builtins=self.get_builtins())
        result.engine = self
        self._process_directives(result)
        return result

    def call(self, query, database, target, transform=None, **kwdargs):
        raise NotImplementedError("ClauseDBEngine.call is an abstract function.")

    def execute(self, node_id, database=None, context=None, target=None, **kwdargs):
        raise NotImplementedError("ClauseDBEngine.execute is an abstract function.")

    def get_non_cache_functor(self):
        """Get a unique functor that is excluded from caching.

        :return: unique functor that is excluded from caching
        :rtype: basestring
        """
        self._unique_number += 1
        return '_nocache_%s' % self._unique_number

    def _process_directives(self, db):
        """Process directives present in the database."""
        term = Term('_directive')
        directive_node = db.find(term)
        if directive_node is None:
            return True    # no directives
        directives = db.get_node(directive_node).children

        gp = LogicFormula()
        while directives:
            current = directives.pop(0)
            self.execute(current, database=db, context=self.create_context((), define=None),
                         target=gp)
        return True

    # noinspection PyUnusedLocal
    def create_context(self, content, define=None):
        """Create a variable context."""
        return content

    def _fix_context(self, context):
        return tuple(context)

    def _clone_context(self, context):
        return list(context)

    def query(self, db, term, **kwdargs):
        """

       :param db:
       :param term:
       :param kwdargs:
       :return:
        """
        gp = LogicFormula()
        if term.is_negated():
            term = -term
            negative = True
        else:
            negative = False
        gp, result = self._ground(db, term, gp, **kwdargs)
        if negative:
            if not result:
                return [term]
            else:
                return []
        else:
            return [x for x, y in result]

    def ground(self, db, term, target=None, label=None, **kwdargs):
        """Ground a query on the given database.

       :param db: logic program
       :type db: LogicProgram
       :param term: query term
       :type term: Term
       :param gp: output data structure (for incremental grounding)
       :type gp: LogicFormula
       :param label: type of query (e.g. ``query``, ``evidence`` or ``-evidence``)
       :type label: str
       :param kwdargs: additional arguments
       :return: ground program containing the query
       :rtype: LogicFormula
        """
        if term.is_negated():
            negated = True
            term = -term
        elif term.functor in ('not', '\+') and term.arity == 1:
            negated = True
            term = term.args[0]
        else:
            negated = False

        target, results = self._ground(db, term, target, silent_fail=False, **kwdargs)
        for args, node_id in results:
            term_store = term.with_args(*args)
            if negated:
                target.add_name(-term_store, target.negate(node_id), label)
            else:
                target.add_name(term_store, node_id, label)
        if not results:
            if negated:
                target.add_name(-term, target.TRUE, label)
            else:
                target.add_name(term, target.FALSE, label)

        return target

    def ground_step(self, db, term, gp=None, silent_fail=True, assume_prepared=False, **kwdargs):
        """

        :param db:
        :type db: LogicProgram
        :param term:
        :param gp:
        :param silent_fail:
        :param assume_prepared:
        :param kwdargs:
        :return:
        """
        # Convert logic program if needed.
        if not assume_prepared:
            db = self.prepare(db)
        # Create a new target datastructure if none was given.
        if gp is None:
            gp = LogicFormula()
        # Find the define node for the given query term.
        clause_node = db.find(term)
        # If term not defined: fail query (no error)    # TODO add error to make it consistent?
        if clause_node is None:
            # Could be builtin?
            clause_node = db.get_builtin(term.signature)
        if clause_node is None:
            if silent_fail or self.unknown == self.UNKNOWN_FAIL:
                return []
            else:
                raise UnknownClause(term.signature, location=db.lineno(term.location))

        try:
            term = term.apply(_ReplaceVar())  # replace Var(_) by integers

            context = self.create_context(term.args)
            context, xxx = substitute_call_args(context, context, 0)
            actions = self.execute_init(clause_node, database=db, target=gp, context=context,
                                        **kwdargs)
        except UnknownClauseInternal:
            if silent_fail or self.unknown == self.UNKNOWN_FAIL:
                return []
            else:
                raise UnknownClause(term.signature, location=db.lineno(term.location))
        return actions

    def _ground(self, db, term, gp=None, silent_fail=True, assume_prepared=False, **kwdargs):
        """

        :param db:
        :type db: LogicProgram
        :param term:
        :param gp:
        :param silent_fail:
        :param assume_prepared:
        :param kwdargs:
        :return:
        """
        # Convert logic program if needed.
        if not assume_prepared:
            db = self.prepare(db)
        # Create a new target datastructure if none was given.
        if gp is None:
            gp = LogicFormula()
        # Find the define node for the given query term.
        clause_node = db.find(term)
        # If term not defined: fail query (no error)    # TODO add error to make it consistent?
        if clause_node is None:
            # Could be builtin?
            clause_node = db.get_builtin(term.signature)
        if clause_node is None:
            if silent_fail or self.unknown == self.UNKNOWN_FAIL:
                return gp, []
            else:
                raise UnknownClause(term.signature, location=db.lineno(term.location))

        try:
            term = term.apply(_ReplaceVar())  # replace Var(_) by integers

            context = self.create_context(term.args)
            context, xxx = substitute_call_args(context, context, 0)
            if self.debugger:
                location = db.lineno(term.location)
                self.debugger.call_create(clause_node, term.functor, context, None, location)
            results = self.execute(clause_node, database=db, target=gp, context=context, **kwdargs)
        except UnknownClauseInternal:
            if silent_fail or self.unknown == self.UNKNOWN_FAIL:
                return gp, []
            else:
                raise UnknownClause(term.signature, location=db.lineno(term.location))
        return gp, results

    def ground_evidence(self, db, target, evidence, propagate_evidence=False):
        logger = logging.getLogger('problog')
        # Ground evidence
        for query in evidence:
            if len(query) == 1:  # evidence/1
                if query[0].is_negated():
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(db, -query[0], target, label=target.LABEL_EVIDENCE_NEG, is_root=True)
                    logger.debug("Ground program size: %s", len(target))
                else:
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(db, query[0], target, label=target.LABEL_EVIDENCE_POS, is_root=True)
                    logger.debug("Ground program size: %s", len(target))
            else:  # evidence/2
                if str(query[1]) == 'true' or query[1] == True:
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(db, query[0], target, label=target.LABEL_EVIDENCE_POS, is_root=True)
                    logger.debug("Ground program size: %s", len(target))
                elif str(query[1]) == 'false' or query[1] == False:
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(db, query[0], target, label=target.LABEL_EVIDENCE_NEG, is_root=True)
                    logger.debug("Ground program size: %s", len(target))
                else:
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(db, query[0], target, label=target.LABEL_EVIDENCE_MAYBE, is_root=True)
                    logger.debug("Ground program size: %s", len(target))
        if propagate_evidence:
            with Timer('Propagating evidence'):
                target.lookup_evidence = {}
                ev_nodes = [node for name, node in target.evidence() if node != 0 and node is not None]
                target.propagate(ev_nodes, target.lookup_evidence)

    def ground_queries(self, db, target, queries):
        logger = logging.getLogger('problog')
        for label, query in queries:
            logger.debug("Grounding query '%s'", query)
            target = self.ground(db, query, target, label=label)
            logger.debug("Ground program size: %s", len(target))

    def ground_all(self, db, target=None, queries=None, evidence=None, propagate_evidence=False, labels=None):
        if labels is None:
            labels = []
        # Initialize target if not given.
        if target is None:
            target = LogicFormula()

        db = self.prepare(db)
        logger = logging.getLogger('problog')
        with Timer('Grounding'):
            # Load queries: use argument if available, otherwise load from database.
            if queries is None:
                queries = [q[0] for q in self.query(db, Term('query', None))]
            for query in queries:
                if not isinstance(query, Term):
                    raise GroundingError('Invalid query')   # TODO can we add a location?
            # Load evidence: use argument if available, otherwise load from database.
            if evidence is None:
                evidence = self.query(db, Term('evidence', None, None))
                evidence += self.query(db, Term('evidence', None))

            queries = [(target.LABEL_QUERY, q) for q in queries]
            for label, arity in labels:
                queries += [(label, q[0]) for q in self.query(db, Term(label, *([None] * arity)))]

            for ev in evidence:
                if not isinstance(ev[0], Term):
                    raise GroundingError('Invalid evidence')   # TODO can we add a location?
            # Ground queries
            if propagate_evidence:
                self.ground_evidence(db, target, evidence, propagate_evidence=propagate_evidence)
                self.ground_queries(db, target, queries)
                if hasattr(target, 'lookup_evidence'):
                    logger.debug('Propagated evidence: %s' % list(target.lookup_evidence))
            else:
                self.ground_queries(db, target, queries)
                self.ground_evidence(db, target, evidence)
        return target

    def add_external_calls(self, externals):
        self.__externals.update(externals)

    def get_external_call(self, func_name):
        if self.__externals is None or func_name not in self.__externals:
            return None
        return self.__externals[func_name]


class _ReplaceVar(object):

    def __init__(self):
        self.translate = {}

    def __getitem__(self, name):
        if type(name) == str:
            if name in self.translate:
                return self.translate[name]
            else:
                v = -len(self.translate) - 1
                self.translate[name] = v
                return v
        else:
            return name


class UnknownClauseInternal(Exception):
    """Undefined clause in call used internally."""
    pass


class NonGroundProbabilisticClause(GroundingError):
    """Encountered a non-ground probabilistic clause."""

    def __init__(self, location):
        GroundingError.__init__(self, 'Encountered a non-ground probabilistic clause', location)


class UnknownClause(GroundingError):
    """Undefined clause in call."""

    def __init__(self, signature, location):
        GroundingError.__init__(self, "No clauses found for '%s'" % signature, location)
        self.signature = signature


from .engine_stack import StackBasedEngine as DefaultEngine


def intersection(l1, l2):
    i = 0
    j = 0
    n1 = len(l1)
    n2 = len(l2)
    r = []
    a = r.append
    while i < n1 and j < n2:
        if l1[i] == l2[j]:
            a(l1[i])
            i += 1
            j += 1
        elif l1[i] < l2[j]:
            i += 1
        else:
            j += 1
    return r


class ClauseIndex(list):

    def __init__(self, parent, arity):
        list.__init__(self)
        self.__parent = parent
        self.__basetype = OrderedSet
        self.__index = [defaultdict(self.__basetype) for _ in range(0, arity)]
        self.__optimized = False
        self.__erased = set()

    def find(self, arguments):
        results = None
        for i, arg in enumerate(arguments):
            if not is_ground(arg):
                pass  # Variable => no restrictions
            else:
                curr = self.__index[i].get(arg)
                none = self.__index[i].get(None, self.__basetype())
                if curr is None:
                    curr = none
                else:
                    curr |= none

                if results is None:  # First argument with restriction
                    results = curr
                else:
                    results = results & curr       # for some reason &= doesn't work here
            if results is not None and not results:
                return []
        if results is None:
            if self.__erased:
                return OrderedSet(self) - self.__erased
            else:
                return self
        else:
            if self.__erased:
                return results - self.__erased
            else:
                return results

    def _add(self, key, item):
        for i, k in enumerate(key):
            self.__index[i][k].add(item)

    def append(self, item):
        list.append(self, item)
        key = []
        args = self.__parent.get_node(item).args
        for arg in args:
            if is_ground(arg):
                key.append(arg)
            else:
                key.append(None)
        self._add(key, item)

    def erase(self, items):
        self.__erased |= set(items)


class ClauseDB(LogicProgram):
    """Compiled logic program.

    A logic program is compiled into a table of instructions.
    The types of instructions are:

    define( functor, arity, defs )
        Pointer to all definitions of functor/arity.
        Definitions can be: ``fact``, ``clause`` or ``adc``.

    clause( functor, arguments, bodynode, varcount )
        Single clause. Functor is the head functor, Arguments are the head arguments. Body node is a pointer to the node representing the body. Var count is the number of variables in head and body.

    fact( functor, arguments, probability )
        Single fact.

    adc( functor, arguments, bodynode, varcount, parent )
        Single annotated disjunction choice. Fields have same meaning as with ``clause``, parent_node points to the parent ``ad`` node.

    ad( childnodes )
        Annotated disjunction group. Child nodes point to the ``adc`` nodes of the clause.

    call( functor, arguments, defnode )
        Body literal with call to clause or builtin. Arguments contains the call arguments, definition node is the pointer to the definition node of the given functor/arity.

    conj( childnodes )
        Logical and. Currently, only 2 children are supported.

    disj( childnodes )
        Logical or. Currently, only 2 children are supported.

    neg( childnode )
        Logical not.

    """
    _define = namedtuple('define', ('functor', 'arity', 'children', 'location'))
    _clause = namedtuple('clause', ('functor', 'args', 'probability', 'child',
                                    'varcount', 'locvars', 'group', 'location'))
    _fact = namedtuple('fact', ('functor', 'args', 'probability', 'location'))
    _call = namedtuple('call', ('functor', 'args', 'defnode', 'location', 'op_priority', 'op_spec'))
    _disj = namedtuple('disj', ('children', 'location'))
    _conj = namedtuple('conj', ('children', 'location'))
    _neg = namedtuple('neg', ('child', 'location'))
    _choice = namedtuple('choice', ('functor', 'args', 'probability', 'locvars', 'group', 'choice', 'location'))
    _extern = namedtuple('extern', ('functor', 'arity', 'function',))

    FUNCTOR_CHOICE = 'choice'
    FUNCTOR_BODY = 'body'

    def __init__(self, builtins=None, parent=None):
        LogicProgram.__init__(self)
        self.__nodes = []   # list of nodes
        self.__heads = {}   # head.sig => node index

        self.__builtins = builtins

        self.data = {}
        self.engine = None

        self.__parent = parent
        self.__node_redirect = {}

        if parent is None:
            self.__offset = 0
        else:
            if hasattr(parent, 'line_info'):
                self.line_info = parent.line_info
            if hasattr(parent, 'source_files'):
                self.source_files = parent.source_files[:]
            self.__offset = len(parent)

        self.dont_cache = set()

    def __len__(self):
        return len(self.__nodes) + self.__offset

    def extend(self):
        return ClauseDB(parent=self)

    def set_data(self, key, value):
        self.data[key] = value

    def update_data(self, key, value):
        if self.has_data(key):
            if type(value) == list:
                self.data[key] += value
            elif type(value) == dict:
                self.data[key].update(value)
            else:
                raise TypeError('Can\'t update data of type \'%s\'' % type(value))
        else:
            self.data[key] = value

    def has_data(self, key):
        return key in self.data

    def get_data(self, key, default=None):
        return self.data.get(key, default)

    def get_builtin(self, signature):
        if self.__builtins is None:
            if self.__parent is not None:
                return self.__parent.get_builtin(signature)
            else:
                return None
        else:
            return self.__builtins.get(signature)

    def get_reserved_names(self):
        return {self.FUNCTOR_CHOICE, self.FUNCTOR_BODY}

    def is_reserved_name(self, name):
        return name is self.get_reserved_names()

    def _create_index(self, arity):
        # return []
        return ClauseIndex(self, arity)

    def _add_and_node(self, op1, op2, location=None):
        """Add an *and* node."""
        return self._append_node(self._conj((op1, op2), location))

    def _add_not_node(self, op1, location=None):
        """Add a *not* node."""
        return self._append_node(self._neg(op1, location))

    def _add_or_node(self, op1, op2, location=None):
        """Add an *or* node."""
        return self._append_node(self._disj((op1, op2), location))

    def _add_define_node(self, head, childnode):
        define_index = self._add_head(head)
        define_node = self.get_node(define_index)
        if not define_node:
            clauses = self._create_index(head.arity)
            self._set_node(define_index, self._define(head.functor, head.arity, clauses, head.location))
        else:
            clauses = define_node.children
        clauses.append(childnode)
        return childnode

    def _add_choice_node(self, choice, functor, args, probability, locvars, group, location=None):
        choice_node = self._append_node(self._choice(functor, args, probability, locvars, group, choice, location))
        return choice_node

    def _add_clause_node(self, head, body, varcount, locvars, group=None):
        clause_node = self._append_node(self._clause(
            head.functor, head.args, head.probability, body, varcount, locvars, group, head.location))
        return self._add_define_node(head, clause_node)

    def _add_call_node(self, term):
        """Add a *call* node."""
        if term.signature in ('query/1', 'evidence/1', 'evidence/2'):
            raise AccessError("Can\'t call %s directly." % term.signature)

        defnode = self._add_head(term, create=False)
        return self._append_node(self._call(term.functor, term.args, defnode, term.location,
                                            term.op_priority, term.op_spec))

    def get_node(self, index):
        """Get the instruction node at the given index.

        :param index: index of the node to retrieve
        :type index: :class:`int`
        :returns: requested node
        :rtype: :class:`tuple`
        :raises IndexError: the given index does not point to a node
        """
        index = self.__node_redirect.get(index, index)

        if index < self.__offset:
            return self.__parent.get_node(index)
        else:
            return self.__nodes[index - self.__offset]

    def _set_node(self, index, node):
        if index < self.__offset:
            raise IndexError('Can\'t update node in parent.')
        else:
            self.__nodes[index - self.__offset] = node

    def _append_node(self, node=()):
        index = len(self)
        self.__nodes.append(node)
        return index

    def _get_head(self, head):
        node = self.__heads.get(head.signature)
        if node is None and self.__parent:
            node = self.__parent._get_head(head)
        return node

    def _set_head(self, head, index):
        self.__heads[head.signature] = index

    def _add_head(self, head, create=True):
        if self.is_reserved_name(head.functor):
            raise AccessError("'%s' is a reserved name" % head.functor)
        node = self.get_builtin(head.signature)
        if node is not None:
            if create:
                raise AccessError("Can not overwrite built-in '%s'." % head.signature)
            else:
                return node

        node = self._get_head(head)
        if node is None:
            if create:
                node = self._append_node(self._define(head.functor, head.arity, self._create_index(head.arity),
                                                      head.location))
            else:
                node = self._append_node()
            self._set_head(head, node)
        elif create and node < self.__offset:
            existing = self.get_node(node)
            # node exists in parent
            clauses = self._create_index(head.arity)
            if existing:
                for c in existing.children:
                    clauses.append(c)
            old_node = node
            node = self._append_node(self._define(head.functor, head.arity, clauses,
                                                  head.location))
            self.__node_redirect[old_node] = node
            self._set_head(head, node)

        return node

    def find(self, head):
        """Find the ``define`` node corresponding to the given head.

        :param head: clause head to match
        :type head: :class:`.basic.Term`
        :returns: location of the clause node in the database, \
                     returns ``None`` if no such node exists
        :rtype: :class:`int` or ``None``
        """
        return self._get_head(head)

    def __repr__(self):
        s = ''
        for i, n in enumerate(self.__nodes):
            i += self.__offset
            s += '%s: %s\n' % (i, n)
        s += str(self.__heads)
        return s

    def add_clause(self, clause):
        """Add a clause to the database.

       :param clause: Clause to add
       :type clause: Clause
       :returns: location of the definition node in the database
       :rtype: int
        """
        return self._compile(clause)

    def add_fact(self, term):
        """Add a fact to the database.
       :param term: fact to add
       :type term: Term
       :return: position of the definition node in the database
       :rtype: int
        """

        # Count the number of variables in the fact
        variables = _AutoDict()
        term.apply(variables)
        # If the fact has variables, threat is as a clause.
        if len(variables) == 0:
            fact_node = self._append_node(self._fact(term.functor, term.args,
                                                     term.probability, term.location))
            return self._add_define_node(term, fact_node)
        else:
            return self.add_clause(Clause(term, Term('true')))

    def add_extern(self, predicate, arity, function):
        head = Term(predicate, *[None] * arity)
        node_id = self._get_head(head)
        if node_id is None:
            node_id = self._append_node(self._extern(predicate, arity, function))
            self._set_head(head, node_id)
        else:
            node = self.get_node(node_id)
            if node == ():
                self._set_node(node_id, self._extern(predicate, arity, function))
            else:
                raise AccessError("External function overrides already defined predicate '%s'"
                                  % head.signature)

    def get_local_scope(self, signature):
        if signature in ('findall/3', 'all/3', 'all_or_none/3'):
            return 0, 1
        else:
            return []

    def _compile(self, struct, variables=None):
        """Compile the given structure and add it to the database.

        :param struct: structure to compile
        :type struct: Term
        :param variables: mapping between variable names and variable index
        :type variables: _AutoDict
        :return: position of the compiled structure in the database
        :rtype: int
        """
        if variables is None:
            variables = _AutoDict()

        if isinstance(struct, And):
            op1 = self._compile(struct.op1, variables)
            op2 = self._compile(struct.op2, variables)
            return self._add_and_node(op1, op2)
        elif isinstance(struct, Or):
            op1 = self._compile(struct.op1, variables)
            op2 = self._compile(struct.op2, variables)
            return self._add_or_node(op1, op2)
        elif isinstance(struct, Not):
            variables.enter_local()
            child = self._compile(struct.child, variables)
            variables.exit_local()
            return self._add_not_node(child, location=struct.location)
        elif isinstance(struct, Term) and struct.signature == 'not/1':
            child = self._compile(struct.args[0], variables)
            return self._add_not_node(child, location=struct.location)
        elif isinstance(struct, AnnotatedDisjunction):
            # Determine number of variables in the head
            new_heads = [head.apply(variables) for head in struct.heads]

            # Group id
            group = len(self.__nodes)

            # Create the body clause
            body_node = self._compile(struct.body, variables)
            body_count = len(variables)
            # Body arguments
            body_args = tuple(range(0, len(variables)))
            body_functor = self.FUNCTOR_BODY + '_' + str(len(self))
            if len(new_heads) > 1:
                heads_list = Term('multi')  # list2term(new_heads)
            else:
                heads_list = new_heads[0]
            body_head = Term(body_functor, Constant(group), heads_list, *body_args)
            self._add_clause_node(body_head, body_node, len(variables), variables.local_variables)
            clause_body = self._add_head(body_head)
            for choice, head in enumerate(new_heads):
                # For each head: add choice node
                choice_functor = Term(self.FUNCTOR_CHOICE,
                                      Constant(group), Constant(choice), head.with_probability())
                choice_node = self._add_choice_node(choice, choice_functor, body_args,
                                                    head.probability, variables.local_variables,
                                                    group, head.location)
                choice_call = self._append_node(self._call(choice_functor, body_args, choice_node,
                                                           head.location, None, None))
                body_call = self._append_node(self._call(body_functor, body_head.args, clause_body,
                                                         head.location, None, None))
                choice_body = self._add_and_node(body_call, choice_call)
                self._add_clause_node(head, choice_body, body_count, {}, group=group)
            return None
        elif isinstance(struct, Clause):
            if struct.head.probability is not None:
                return self._compile(AnnotatedDisjunction([struct.head], struct.body))
            else:
                new_head = struct.head.apply(variables)
                body_node = self._compile(struct.body, variables)
                return self._add_clause_node(new_head, body_node, len(variables),
                                             variables.local_variables)
        elif isinstance(struct, Var):
            return self._add_call_node(Term('call', struct.apply(variables),
                                            location=struct.location))
        elif isinstance(struct, Term):
            local_scope = self.get_local_scope(struct.signature)
            if local_scope:
                # Special case for findall: any variables added by the first
                #  two arguments of findall are 'local' variables.
                args = []
                for i, a in enumerate(struct.args):
                    if not isinstance(a, Term):
                        # For nested findalls: 'a' can be a raw variable pointer
                        # Temporarily wrap it in a Term, so we can call 'apply' on it.
                        a = Term('_', a)
                    if i in local_scope:
                        variables.enter_local()
                        new_arg = a.apply(variables)
                        variables.exit_local()
                    else:
                        new_arg = a.apply(variables)
                    if a.functor == '_':
                        # If the argument was temporarily wrapped: unwrap it.
                        new_arg = new_arg.args[0]
                    args.append(new_arg)
                return self._add_call_node(struct(*args))
            else:
                return self._add_call_node(struct.apply(variables))
        else:
            raise ValueError("Unknown structure type: '%s'" % struct)

    def _create_vars(self, term):
        if type(term) == int:
            return Var('V_' + str(term))
        else:
            args = [self._create_vars(arg) for arg in term.args]
            return term.with_args(*args)

    def _extract(self, node_id):
        node = self.get_node(node_id)
        if not node:
            raise ValueError("Unexpected empty node.")

        nodetype = type(node).__name__
        if nodetype == 'fact':
            return Term(node.functor, *node.args, p=node.probability)
        elif nodetype == 'call':
            func = node.functor
            args = node.args
            if isinstance(func, Term):
                return self._create_vars(func(*(func.args + args)))
            else:
                return self._create_vars(Term(func, *args,
                                              priority=node.op_priority, opspec=node.op_spec))
        elif nodetype == 'conj':
            a, b = node.children
            return And(self._extract(a), self._extract(b))
        elif nodetype == 'disj':
            a, b = node.children
            return Or(self._extract(a), self._extract(b))
        elif nodetype == 'neg':
            return Not('\+', self._extract(node.child))
        else:
            raise ValueError("Unknown node type: '%s'" % nodetype)

    def to_clause(self, index):
        node = self.get_node(index)
        nodetype = type(node).__name__
        if nodetype == 'fact':
            return Term(node.functor, *node.args, p=node.probability)
        elif nodetype == 'clause':
            head = self._create_vars(Term(node.functor, *node.args, p=node.probability))
            return Clause(head, self._extract(node.child))

    def __iter__(self):
        clause_groups = defaultdict(list)
        for index, node in enumerate(self.__nodes):
            index += self.__offset
            if not node:
                continue
            nodetype = type(node).__name__
            if nodetype == 'fact':
                yield Term(node.functor, *node.args, p=node.probability)
            elif nodetype == 'clause':
                if node.group is None:
                    head = self._create_vars(Term(node.functor, *node.args, p=node.probability))
                    yield Clause(head, self._extract(node.child))
                else:
                    clause_groups[node.group].append(index)
        for group in clause_groups.values():
            heads = []
            body = None
            for index in group:
                node = self.get_node(index)
                heads.append(self._create_vars(Term(node.functor, *node.args, p=node.probability)))
                if body is None:
                    body_node = self.get_node(node.child)
                    body_node = self.get_node(body_node.children[0])
                    body = self._create_vars(Term(body_node.functor, *body_node.args))
            yield AnnotatedDisjunction(heads, body)

    def iter_raw(self):
        """Iterate over clauses of model as represented in the database i.e. with choice facts and
         without annotated disjunctions.
        """

        clause_groups = defaultdict(list)
        for index, node in enumerate(self.__nodes):
            index += self.__offset
            if not node:
                continue
            nodetype = type(node).__name__
            if nodetype == 'fact':
                yield Term(node.functor, *node.args, p=node.probability)
            elif nodetype == 'clause':
                if node.group is None:
                    head = self._create_vars(Term(node.functor, *node.args, p=node.probability))
                    yield Clause(head, self._extract(node.child))
                else:
                    head = self._create_vars(Term(node.functor, *node.args))
                    yield Clause(head, self._extract(node.child))
            elif nodetype == 'choice':
                group = node.functor.args[0]
                c = node.functor(*(node.functor.args + node.args))
                clause_groups[group].append(c)
                yield c.with_probability(node.probability)

        for group in clause_groups.values():
            if len(group) > 1:
                yield Term('mutual_exclusive', list2term(group))

    def resolve_filename(self, filename):
        root = self.source_root
        if hasattr(filename, 'location') and filename.location:
            source_root = self.source_files[filename.location[0]]
            if source_root:
                root = os.path.dirname(source_root)

        atomstr = str(filename)
        if atomstr[0] == atomstr[-1] == "'":
            atomstr = atomstr[1:-1]
        filename = os.path.join(root, atomstr)
        return filename

    def create_function(self, functor, arity):
        """Create a Python function that can be used to query a specific predicate on this database.

        :param functor: functor of the predicate
        :param arity: arity of the predicate (the function will take arity - 1 arguments
        :return: a Python callable
        """
        return PrologFunction(self, functor, arity)

    def iter_nodes(self):
        # TODO make this work for extended database
        return iter(self.__nodes)



class PrologFunction(object):

    def __init__(self, database, functor, arity):
        self.database = database
        self.functor = functor
        self.arity = arity

    def __call__(self, *args):
        args = args[:self.arity - 1]
        query_term = Term(self.functor, *(args + (None,)))
        result = self.database.engine.query(self.database, query_term)
        assert len(result) == 1
        return result[0][-1]



class AccessError(GroundingError):
    pass


class _AutoDict(dict):

    def __init__(self):
        dict.__init__(self)
        self.__record = set()
        self.__anon = 0
        self.__localmode = False
        self.local_variables = set()

    def enter_local(self):
        self.__localmode = True

    def exit_local(self):
        self.__localmode = False

    def __getitem__(self, key):
        if key == '_' and self.__localmode:
            key = '_#%s' % len(self.local_variables)

        if key == '_' or key is None:

            value = len(self)
            self.__anon += 1
            return value
        else:
            value = self.get(key)
            if value is None:
                value = len(self)
                self[key] = value
                if self.__localmode:
                    self.local_variables.add(value)
            elif not self.__localmode and value in self.local_variables:
                # Variable initially defined in local scope is reused outside local scope.
                # This means it's not local anymore.
                self.local_variables.remove(value)
            self.__record.add(value)
            return value

    def __len__(self):
        return dict.__len__(self) + self.__anon

    def usedVars(self):
        result = set(self.__record)
        self.__record.clear()
        return result

    def define(self, key):
        if key not in self:
            value = len(self)
            self[key] = value
