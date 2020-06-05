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

from collections import defaultdict

from .program import LogicProgram
from .logic import *
from .formula import LogicFormula
from .engine_unify import *

from .core import transform
from .errors import GroundingError, NonGroundQuery
from .util import Timer

from subprocess import CalledProcessError


@transform(LogicProgram, LogicFormula)
def ground(model, target=None, grounder=None, **kwdargs):
    """Ground a given model.

    :param model: logic program to ground
    :type model: LogicProgram
    :return: the ground program
    :rtype: LogicFormula
    """
    if grounder in ("yap", "yap_debug"):
        from .ground_yap import ground_yap

        return ground_yap(model, target, **kwdargs)
    else:
        return ground_default(model, target, **kwdargs)


@transform(LogicProgram, LogicFormula)
def ground_default(
    model,
    target=None,
    queries=None,
    evidence=None,
    propagate_evidence=False,
    labels=None,
    engine=None,
    **kwdargs
):
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
    return engine.ground_all(
        model,
        target,
        queries=queries,
        evidence=evidence,
        propagate_evidence=propagate_evidence,
        labels=labels,
    )


class GenericEngine(object):  # pragma: no cover
    """Generic interface to a grounding engine."""

    def prepare(self, db):
        """Prepare the given database for querying.
        Calling this method is optional.

       :param db: logic program
       :returns: logic program in optimized format where builtins are initialized and directives \
       have been evaluated
        """
        raise NotImplementedError("GenericEngine.prepare is an abstract method.")

    def query(self, db, term):
        """Evaluate a query without generating a ground program.

       :param db: logic program
       :param term: term to query; variables should be represented as None
       :returns: list of tuples of argument for which the query succeeds.
        """
        raise NotImplementedError("GenericEngine.query is an abstract method.")

    def ground(self, db, term, target=None, label=None):
        """Ground a given query term and store the result in the given ground program.

       :param db: logic program
       :param term: term to ground; variables should be represented as None
       :param target: target logic formula to store grounding in (a new one is created if none is \
       given)
       :param label: optional label (query, evidence, ...)
       :returns: logic formula (target if given)
        """
        raise NotImplementedError("GenericEngine.ground is an abstract method.")

    def ground_all(self, db, target=None, queries=None, evidence=None):
        """Ground all queries and evidence found in the the given database.

       :param db: logic program
       :param target: logic formula to ground into
       :param queries: list of queries to evaluate instead of the ones in the logic program
       :param evidence: list of evidence to evaluate instead of the ones in the logic program
       :returns: ground program
        """
        raise NotImplementedError("GenericEngine.ground_all is an abstract method.")


class ClauseDBEngine(GenericEngine):
    """Parent class for all Python ClauseDB-based engines."""

    UNKNOWN_ERROR = 0
    UNKNOWN_FAIL = 1

    def __init__(self, builtins=True, **kwdargs):
        self.__builtin_index = {}
        self.__builtins = []
        self.__externals = {}

        self._unique_number = 0
        self.unknown = kwdargs.get("unknown", self.UNKNOWN_ERROR)

        if builtins:
            self.load_builtins()

        self.functions = {}
        self.args = kwdargs.get("args")

    def load_builtins(self):
        """Load default builtins."""
        raise NotImplementedError(
            "ClauseDBEngine.loadBuiltIns is an abstract function."
        )

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
        sig = "%s/%s" % (predicate, arity)
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
        return "_nocache_%s" % self._unique_number

    def _process_directives(self, db):
        """Process directives present in the database."""
        term = Term("_directive")
        directive_node = db.find(term)
        if directive_node is not None:
            directives = db.get_node(directive_node).children

            gp = LogicFormula()
            while directives:
                current = directives.pop(0)
                self.execute(
                    current,
                    database=db,
                    context=self.create_context((), define=None),
                    target=gp,
                )
        return True

    # noinspection PyUnusedLocal
    def create_context(self, content, define=None, parent=None):
        """Create a variable context."""
        return content

    def _fix_context(self, context):
        return tuple(context)

    def _clone_context(self, context):
        return list(context)

    def query(self, db, term, backend=None, **kwdargs):
        """

        :param db:
        :param term:
        :param kwdargs:
        :return:
        """

        if backend in ("swipl", "yap"):
            from .util import mktempfile, subprocess_check_output

            tmpfn = mktempfile(".pl")
            with open(tmpfn, "w") as tmpf:
                print(db.to_prolog(), file=tmpf)

            from problog.logic import term2str

            termstr = term2str(term)
            cmd = [
                "swipl",
                "-l",
                tmpfn,
                "-g",
                "%s, writeln(%s), fail; halt" % (termstr, termstr),
            ]

            try:
                output = subprocess_check_output(cmd)
            except CalledProcessError as err:
                in_error = True
                error_message = []
                for line in err.output.split("\n"):
                    if line.startswith("Warning:"):
                        in_error = False
                    elif line.startswith("ERROR:"):
                        in_error = True
                    if in_error:
                        error_message.append(line)
                error_message = "SWI-Prolog returned some errors:\n" + "\n".join(
                    error_message
                )
                raise GroundingError(error_message)

            return [
                Term.from_string(line).args
                for line in output.split("\n")
                if line.strip()
            ]
        else:
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
        elif term.functor in ("not", "\+") and term.arity == 1:
            negated = True
            term = term.args[0]
        else:
            negated = False

        target, results = self._ground(db, term, target, silent_fail=False, **kwdargs)

        args_node = defaultdict(list)
        for args, node_id in results:
            if not is_ground(*args) and target.is_probabilistic(node_id):
                raise NonGroundQuery(term, db.lineno(term.location))
            args = tuple(args)
            args_node[args].append(node_id)
        for args, node_ids in args_node.items():
            if len(node_ids) > 1:
                node_id = target.add_or(node_ids)
            else:
                node_id = node_ids[0]
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

    def ground_step(
        self, db, term, gp=None, silent_fail=True, assume_prepared=False, **kwdargs
    ):
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
            actions = self.execute_init(
                clause_node, database=db, target=gp, context=context, **kwdargs
            )
        except UnknownClauseInternal:
            if silent_fail or self.unknown == self.UNKNOWN_FAIL:
                return []
            else:
                raise UnknownClause(term.signature, location=db.lineno(term.location))
        return actions

    def _ground(
        self, db, term, gp=None, silent_fail=True, assume_prepared=False, **kwdargs
    ):
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
                self.debugger.call_create(
                    clause_node, term.functor, context, None, location
                )
            results = self.execute(
                clause_node, database=db, target=gp, context=context, **kwdargs
            )
        except UnknownClauseInternal:
            if silent_fail or self.unknown == self.UNKNOWN_FAIL:
                return gp, []
            else:
                raise UnknownClause(term.signature, location=db.lineno(term.location))
        return gp, results

    def ground_evidence(self, db, target, evidence, propagate_evidence=False):
        logger = logging.getLogger("problog")
        # Ground evidence
        for query in evidence:
            if len(query) == 1:  # evidence/1
                if query[0].is_negated():
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(
                        db,
                        -query[0],
                        target,
                        label=target.LABEL_EVIDENCE_NEG,
                        is_root=True,
                    )
                    logger.debug("Ground program size: %s", len(target))
                else:
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(
                        db,
                        query[0],
                        target,
                        label=target.LABEL_EVIDENCE_POS,
                        is_root=True,
                    )
                    logger.debug("Ground program size: %s", len(target))
            else:  # evidence/2
                if str(query[1]) == "true" or query[1] == True:
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(
                        db,
                        query[0],
                        target,
                        label=target.LABEL_EVIDENCE_POS,
                        is_root=True,
                    )
                    logger.debug("Ground program size: %s", len(target))
                elif str(query[1]) == "false" or query[1] == False:
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(
                        db,
                        query[0],
                        target,
                        label=target.LABEL_EVIDENCE_NEG,
                        is_root=True,
                    )
                    logger.debug("Ground program size: %s", len(target))
                else:
                    logger.debug("Grounding evidence '%s'", query[0])
                    target = self.ground(
                        db,
                        query[0],
                        target,
                        label=target.LABEL_EVIDENCE_MAYBE,
                        is_root=True,
                    )
                    logger.debug("Ground program size: %s", len(target))
        if propagate_evidence:
            with Timer("Propagating evidence"):
                target.lookup_evidence = {}
                ev_nodes = [
                    node
                    for name, node in target.evidence()
                    if node != 0 and node is not None
                ]
                target.propagate(ev_nodes, target.lookup_evidence)

    def ground_queries(self, db, target, queries):
        logger = logging.getLogger("problog")
        for label, query in queries:
            logger.debug("Grounding query '%s'", query)
            target = self.ground(db, query, target, label=label)
            logger.debug("Ground program size: %s", len(target))

    def ground_all(
        self,
        db,
        target=None,
        queries=None,
        evidence=None,
        propagate_evidence=False,
        labels=None,
    ):
        if labels is None:
            labels = []
        # Initialize target if not given.
        if target is None:
            target = LogicFormula()

        db = self.prepare(db)

        logger = logging.getLogger("problog")
        with Timer("Grounding"):
            # Load queries: use argument if available, otherwise load from database.
            if queries is None:
                queries = [q[0] for q in self.query(db, Term("query", None))]
            for query in queries:
                if not isinstance(query, Term):
                    raise GroundingError("Invalid query")  # TODO can we add a location?
            # Load evidence: use argument if available, otherwise load from database.
            if evidence is None:
                evidence = self.query(db, Term("evidence", None, None))
                evidence += self.query(db, Term("evidence", None))

            queries = [(target.LABEL_QUERY, q) for q in queries]
            for label, arity in labels:
                queries += [
                    (label, q[0])
                    for q in self.query(db, Term(label, *([None] * arity)))
                ]

            for ev in evidence:
                if not isinstance(ev[0], Term):
                    raise GroundingError(
                        "Invalid evidence"
                    )  # TODO can we add a location?
            # Ground queries
            if propagate_evidence:
                self.ground_evidence(
                    db, target, evidence, propagate_evidence=propagate_evidence
                )
                self.ground_queries(db, target, queries)
                if hasattr(target, "lookup_evidence"):
                    logger.debug(
                        "Propagated evidence: %s" % list(target.lookup_evidence)
                    )
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
        GroundingError.__init__(
            self, "Encountered a non-ground probabilistic clause", location
        )


class UnknownClause(GroundingError):
    """Undefined clause in call."""

    def __init__(self, signature, location):
        GroundingError.__init__(self, "No clauses found for '%s'" % signature, location)
        self.signature = signature


from .engine_stack import StackBasedEngine as DefaultEngine

from .clausedb import ClauseDB
