from __future__ import print_function

from .util import subprocess_check_output, mktempfile, Timer
from logging import getLogger
from .logic import AnnotatedDisjunction, list2term, Term, Clause, Or, Constant
import sys
import os
from collections import defaultdict, deque
from subprocess import CalledProcessError
from .errors import GroundingError
from .engine import UnknownClause, NonGroundProbabilisticClause


def ground_yap(model, target=None, queries=None, evidence=None, propagate_evidence=False,
               labels=None, engine=None, debug=False, **kwdargs):
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

    with Timer('Grounding (YAP)'):

        if debug:
            fn_model = '/tmp/model.pl'
            fn_ground = '/tmp/model.ground'
            fn_evidence = '/tmp/model.evidence'
            fn_query = '/tmp/model.query'

        else:
            fn_model = mktempfile('.pl')
            fn_ground = mktempfile('.ground')
            fn_evidence = mktempfile('.evidence')
            fn_query = mktempfile('.query')

        with open(fn_model, 'w') as f:
            f.write('\n'.join(map(statement_to_yap, model)) + '\n')

        yap_ground = os.path.join(os.path.dirname(__file__), 'yap', 'ground_compact.pl')

        cmd = ['yap', '-L', yap_ground, '--', fn_model, fn_ground, fn_evidence, fn_query]

        try:
            output = subprocess_check_output(cmd)

        except CalledProcessError as err:
            errmsg = err.output.strip()
            if errmsg.startswith('undefined'):
                raise UnknownClause(errmsg.split()[1], None)
            elif errmsg.startswith('non-ground'):
                raise GroundingError('Non-ground clause detected: %s' % errmsg.split()[1], None)
            else:
                raise err

        with open(fn_query) as f:
            queries = f.readlines()

        with open(fn_evidence) as f:
            evidence = f.readlines()

        with open(fn_ground) as f:
            return read_grounding(f, target, queries, evidence)


def read_grounding(lines, target, queries, evidence):
    """

    :param lines:
    :param target:
    :type target: LogicFormula
    :return:
    """

    id_counts = defaultdict(int)
    parsed_lines = deque()
    names = {}

    # Iteration 1: read in the file and determine which line numbers occur multiple times
    line_nums = []
    for i, line in enumerate(lines):
        line = line.strip()
        line, name = line.split('|')
        name = name.strip()
        name = Term.from_string(name)

        line_num, line_type, line_content = line.split(None, 2)
        line_num = int(line_num)
        names[name] = line_num

        # Count how many times each line occurs
        id_counts[line_num] += 1

        line_nums.append(line_num)
        if line_type == 'FACT':
            parsed_lines.append((line_num, 'FACT', [i, float(line_content)], name))
        elif line_type == 'AND':
            parsed_lines.append((line_num, 'AND', [int(x) for x in line_content.split()], name))
        else:
            raise Exception('Unexpected type: %s' % line_type)

    num2index = {}

    while parsed_lines:
        line_num, line_type, line_content, line_name = parsed_lines.popleft()

        if line_num in num2index:
            # There is already a node for this line number => add this as a disjunct
            or_node = num2index[line_num]
        elif id_counts[line_num] > 1:
            # It's an or-node
            or_node = target.add_or((), placeholder=True)
            num2index[line_num] = or_node
        else:
            or_node = None

        if line_type == 'FACT':
            node_id = target.add_atom(line_content[0], probability=line_content[1], name=line_name)
        else:  # AND
            found_all = True
            children = []
            for child in line_content:
                negated = child < 0
                child = abs(child)
                child_key = num2index.get(child)
                if child_key is None:
                    if id_counts[child] > 1:
                        # Child is and or-node that doesn't exist yet => we can make a placeholder
                        child_key = target.add_or((), placeholder=True)
                        num2index[child] = child_key
                    else:
                        found_all = False
                        child_key = None
                        break
                if child_key is not None:
                    if negated:
                        children.append(-child_key)
                    else:
                        children.append(child_key)
            if found_all:
                node_id = target.add_and(children, name=line_name)
            else:
                parsed_lines.append((line_num, line_type, line_content, line_name))
                node_id = None

        if node_id is not None:
            if or_node is None:
                num2index[line_num] = node_id
            else:
                target.add_disjunct(or_node, node_id)

    for q in queries:
        q = Term.from_string(q)
        if q not in names:
            key = None
        else:
            key = num2index[names[q]]
        target.add_query(q, key)

    for q in evidence:
        q, v = q.rsplit(None, 1)
        q = Term.from_string(q)
        if q not in names:
            key = None
        else:
            key = num2index[names[q]]
        target.add_evidence(q, key, v == 't')

    return target


def statement_to_yap(statement):

    if isinstance(statement, Clause) and statement.head.functor == '_directive':
        if statement.body.functor in ('consult', 'use_module'):
            return ''
        else:
            return ':- %s.' % statement.body

    if isinstance(statement, AnnotatedDisjunction):
        heads = statement.heads

        # heads = [Term('', h.with_probability]

        head = Term('problog_ad', list2term(heads), statement.body)

        return '%s.' % head
    else:
        return '%s.' % statement




