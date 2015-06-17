#! /usr/bin/env python
"""Implementation of MPE (most probable explanation) as a constrained optimization problem.

    Requires fzn-gecode to be installed.
"""


from __future__ import print_function

import math
import subprocess
import sys

from problog.program import PrologFile
from problog.formula import LogicFormula
from problog.evaluator import SemiringLogProbability


def extract_relevant(gp):

    relevant = [False] * (len(gp)+1)

    explore_next = [abs(n) for q, n in gp.queries()] + [abs(n) for q, n in gp.evidence()]
    while explore_next:
        explore_now = explore_next
        explore_next = []
        for x in explore_now:
            if not relevant[x]:
                relevant[x] = True
                node = gp.getNode(x)
                if type(node).__name__ != 'atom':
                    explore_next += list(map(abs, node.children))
    return relevant

def groundprogram2flatzinc(gp):
    """

    :param gp:
    :type: LogicFormula
    :return:
    """
    variables = []
    constraints = []

    weights = gp.extractWeights(SemiringLogProbability())

    has_constraint = False
    for c in gp.constraints():
        has_constraint = True
        break

    if has_constraint:
        relevant = [True] * (len(gp)+1)
    else:
        relevant = extract_relevant(gp)

    print ('Relevant program size:', sum(relevant), file=sys.stderr)

    atom_vars = []
    atoms = []
    probs = []
    probs_offset = 0.0
    neg_nodes = set()
    for i, n, t in gp:
        if not relevant[i]:
            continue
        varname = 'b_lit_%s_pos' % i

        if t == 'conj':
            variables.append('var bool: %s :: var_is_introduced;' % varname)
            childvars = []
            for c in n.children:
                if c > 0:
                    childvars.append('b_lit_%s_pos' % c)
                else:
                    if c not in neg_nodes:
                        neg_nodes.add(c)
                        variables.append('var bool: b_lit_%s_neg :: var_is_introduced;' % -c)
                        constraints.append('constraint bool_not(b_lit_%s_pos,b_lit_%s_neg) :: defines_var(b_lit_%s_neg);' % (-c, -c, -c))
                    childvars.append('b_lit_%s_neg' % -c)
            childvars = ','.join(childvars)
            constraints.append('constraint array_bool_and([%s],%s) :: defines_var(%s);' % (childvars, varname, varname))
        elif t == 'disj':
            variables.append('var bool: %s :: var_is_introduced;' % varname)
            childvars = []
            for c in n.children:
                if c > 0:
                    childvars.append('b_lit_%s_pos' % c)
                else:
                    if c not in neg_nodes:
                        neg_nodes.add(c)
                        variables.append('var bool: b_lit_%s_neg :: var_is_introduced;' % -c)
                        constraints.append('constraint bool_not(b_lit_%s_pos,b_lit_%s_neg) :: defines_var(b_lit_%s_neg);' % (-c,-c,-c))
                    childvars.append('b_lit_%s_neg' % -c)
            childvars = ','.join(childvars)
            constraints.append('constraint array_bool_or([%s],%s) :: defines_var(%s);' % (childvars, varname, varname))
        else:
            variables.append('var bool: %s :: output_var;' % varname)
            var_int = 'i_lit_%s_pos' % i
            var_float_pos = 'f_lit_%s_pos' % i
            # var_float_neg = 'f_lit_%s_neg' % i
            variables.append('var int: %s :: var_is_introduced;' % var_int)
            variables.append('var float: %s :: var_is_introduced;' % var_float_pos)
            # variables.append('var float: %s :: var_is_introduced;' % var_float_neg)
            constraints.append('constraint bool2int(%s,%s) :: defines_var(%s);' % (varname, var_int, var_int))
            constraints.append('constraint int2float(%s,%s) :: defines_var(%s);' % (var_int, var_float_pos, var_float_pos))
            # constraints.append('constraint float_plus(%s,-1.0,%s) :: defines_var(%s);' % (var_float_pos, var_float_neg, var_float_neg))

            pp, pn = weights[i]
            atom_vars.append(varname)
            atoms.append(var_float_pos)
            # atoms.append(var_float_neg)
            probs.append(str(pp-pn))
            # probs.append(str(-pn))
            probs_offset -= pn

    optvar = 'score'
    variables.append('var float: score :: output_var;')
    constraints.append('constraint float_lin_eq([%s],[%s],%s) :: defines_var(%s);' % (','.join(probs + ['-1.0']), ','.join(atoms + [optvar]), probs_offset, optvar))

    for q, n in gp.queries():
        constraints.append('constraint bool_eq(b_lit_%s_pos,true);' % n)

    for q, n in gp.evidence():
        if n > 0:
            constraints.append('constraint bool_eq(b_lit_%s_pos,true);' % n)
        else:
            constraints.append('constraint bool_eq(b_lit_%s_pos,false);' % -n)

    for c in gp.constraints():
        for r in c.encodeCNF():
            r_pos = []
            r_neg = []
            for x in r:
                if x > 0:
                    r_pos.append('b_lit_%s_pos' % x)
                else:
                    r_neg.append('b_lit_%s_pos' % -x)
            constraints.append('constraint bool_clause([%s],[%s]);' % (','.join(r_pos),','.join(r_neg)))

    solve = 'solve :: bool_search([%s], occurrence, indomain_max, complete) maximize %s;' % (','.join(atom_vars), optvar)
    return '\n'.join(variables) + '\n' + '\n'.join(constraints) + '\n' + solve

def groundprogram2lp(gp):

    # b = a_1 /\ a_2 /\ ... /\ a_n
    #
    #   => 0 <= a_1 + a_2 + ... + a_n - nb < n
    #
    # b = a_1 \/ a_2 \/ ... \/ a_n
    #
    #   -b = -a_1 /\ -a2 /\ ... /\ -a_n
    #
    #   => 0 <= -a_1 - a_2 ... - a_n + nb < n

    variables = []
    constraints = []

    weights = gp.extractWeights(SemiringLogProbability())

    has_constraint = False
    for c in gp.constraints():
        has_constraint = True
        break

    if has_constraint:
        relevant = [True] * (len(gp)+1)
    else:
        relevant = extract_relevant(gp)

    print ('Relevant program size:', sum(relevant), file=sys.stderr)

    atom_vars = []
    bounds = []
    atoms = []
    probs = []
    probs_offset = 0.0
    neg_nodes = set()
    for i, n, t in gp:
        if not relevant[i]:
            continue
        varname = 'x%s' % i     # Variable name corresponding to this node
        bounds.append('0 <= x%s <= 1' % i)
        if t == 'conj':
            formula = ''
            low_bound = 0
            high_bound = len(n.children)
            for c in n.children:
                if c < 0:
                    low_bound -= 1
                    high_bound -= 1
                    formula += '-x%s' % -c
                else:
                    if formula:
                        formula += '+'
                    formula += 'x%s' % c
            constraints.append(formula + ' >= ' + str(low_bound))
            constraints.append(formula + ' < ' + str(high_bound))
        elif t == 'disj':
            formula = ''
            low_bound = 0
            high_bound = len(n.children)
            for c in n.children:
                if c < 0:
                    low_bound += 1
                    high_bound += 1
                    if formula:
                        formula += '+'
                    formula += 'x%s' % -c
                else:
                    formula += '-x%s' % c
            constraints.append(formula + ' >= ' + str(low_bound))
            constraints.append(formula + ' < ' + str(high_bound))
        else:
            pp, pn = weights[i]
            atom_vars.append(varname)
            atoms.append(varname)
            # atoms.append(var_float_neg)
            probs.append(str(pp-pn))
            # probs.append(str(-pn))
            probs_offset -= pn

    obj = []
    for at, pr in zip(probs, atoms):
        obj.append('%s %s' % (at, pr))
    obj = ' + '.join(obj)

    for q, n in gp.queries():
        constraints.append('x%s > 0' % n)

    for q, n in gp.evidence():
        if n > 0:
            constraints.append('x%s > 0;' % n)
        else:
            constraints.append('x%s = 0;' % n)

    # TODO constraints
    # for c in gp.constraints():
    #     for r in c.encodeCNF():
    #         r_pos = []
    #         r_neg = []
    #         for x in r:
    #             if x > 0:
    #                 r_pos.append('b_lit_%s_pos' % x)
    #             else:
    #                 r_neg.append('b_lit_%s_pos' % -x)
    #         constraints.append('constraint bool_clause([%s],[%s]);' % (','.join(r_pos),','.join(r_neg)))

    result = """
maximize
  obj: %s
subject to
  %s
bounds
  %s
binary
  %s
end
""" % (obj, '\n  '.join(constraints), '\n  '.join(bounds), ' '.join(atom_vars))

    return result, probs_offset

def call_scip(lp, solver):
    filename = '/tmp/mpe.lp'
    with open(filename, 'w') as f:
        f.write(lp)

    output = subprocess.check_output(solver.split(' ') + ['-f', filename])

    obj_value = None
    facts = set()

    in_the_zone = False
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('objective value'):
            in_the_zone = True
            obj_value = float(line.split()[2])
        elif in_the_zone:
            if not line:
                return obj_value, facts
            else:
                facts.add(int(line.split()[0][1:]))

def call_flatzinc_solver(fzn, solver, **kwdargs):

    filename = '/tmp/mpe.fzn'
    with open(filename, 'w') as f:
        f.write(fzn)
    output = subprocess.check_output(solver.split(' ') + [filename])

    score = None
    facts = {}
    for line in output.split('\n'):
        line = line.strip()
        if line == '=====UNSATISFIABLE=====':
            return 0.0, facts
        elif line[0] == '-':
            return score, facts
        else:
            var, val = line.split(' = ')
            if var == 'score':
                score = math.exp(float(val[:-1]))
            else:
                fact = int(var[6:-4])
                val = True if val == 'true;' else False
                facts[fact] = val


def main(inputfile, **kwdargs):
    pl = PrologFile(inputfile)
    gp = LogicFormula.createFrom(pl, label_all=True)

    lp, score_offset = groundprogram2lp(gp)

    score, facts = call_scip(lp, **kwdargs)
    score -= score_offset

    for name, node in gp.getNames():
        if node in facts:
            facts.remove(node)
            print (name)
    for f in facts:
        print ('node_%s' % f)

    print ('Probability:', math.exp(score))



    # flatzinc = groundprogram2flatzinc(gp)
    # score, facts = call_solver(flatzinc, **kwdargs)
    # for name, node in gp.getNames():
    #     val = facts.get(node)
    #     if val == True:
    #         del facts[node]
    #         print (name)
    #     elif val == False:
    #         del facts[node]
    #         print ('\+' + str(name))
    # for f, v in facts.items():
    #     if v:
    #         print ('node_%s' % f)
    #     else:
    #         print ('\+node_%s' % f)
    #
    # print ('Probability:', score)


def argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('--solver', type=str, default='scip', help="location of the backend solver")
    return parser


if __name__ == '__main__':
    main(**vars(argparser().parse_args()))