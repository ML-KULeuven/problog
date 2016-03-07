"""
Maximum A-Posteriori inference for ProbLog (MAP)
"""

from __future__ import print_function


from .. import get_evaluatable
from ..program import PrologFile
from ..formula import LogicFormula
from ..util import Timer, format_dictionary
from .dtproblog import search_exhaustive, search_local
from ..constraint import TrueConstraint

import math


def main(argv):
    args = argparser().parse_args(argv)

    search = None

    # certain queries are given, they refer to facts
    #  we want to find assignments for these
    #  such that the overall probability is maximized
    # query facts => decisions with utility log p

    # Load the model
    model = PrologFile(args.filename)

    # Ground the model
    ground_program = LogicFormula.create_from(model)

    # Get the conditional probabilities of the query facts
    query_probs = get_evaluatable().create_from(ground_program).evaluate()

    # Extract queries and replace them with decisions
    decisions = []
    utilities = []
    decision_nodes = set()
    for qn, qi in ground_program.queries():
        node = ground_program.get_node(qi)
        prob = query_probs[qn]
        if type(node).__name__ != 'atom':
            raise Exception("Queries should be facts: '%s'" % qn)
        decisions.append((qi, qn))
        decision_nodes.add(qi)
        probability = prob
        utilities.append((qn, math.log(float(probability)) - math.log(1 - float(probability))))
    utilities = dict(utilities)

    # Add constraints that enforce the evidence
    for qn, qi in ground_program.evidence():
        ground_program.add_constraint(TrueConstraint(qi))
    ground_program.clear_evidence()

    # Process constraints (only on decisions, the others are encoded in the model)
    constraints = []
    for c in ground_program.constraints():
        if set(c.get_nodes()) & decision_nodes:
            constraints.append(c)

    # Compile the program (again)
    knowledge = get_evaluatable().create_from(ground_program)

    # Use dt-problog search to find the solution
    if search == 'local':
        result = search_local(knowledge, decisions, utilities, constraints)
    else:
        result = search_exhaustive(knowledge, decisions, utilities, constraints)

    # Print the result
    decisions, score, stats = result
    for x, v in decisions.items():
        if v:
            print (x)
        else:
            print (-x)


def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    return parser
