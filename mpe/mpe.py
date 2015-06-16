#! /usr/bin/env python
"""Implementation of MPE (most probable explanation) as a constrained optimization problem.

    Requires fzn-gecode to be installed.
"""


from __future__ import print_function

import math
import subprocess

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
    relevant = extract_relevant(gp)

    print ('Relevant program size:', sum(relevant))

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


def call_solver(fzn, solver):

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

    flatzinc = groundprogram2flatzinc(gp)
    score, facts = call_solver(flatzinc, **kwdargs)
    for name, node in gp.getNames():
        val = facts.get(node)
        if val == True:
            del facts[node]
            print (name)
        elif val == False:
            del facts[node]
            print ('\+' + str(name))
    for f, v in facts.items():
        if v:
            print ('node_%s' % f)
        else:
            print ('\+node_%s' % f)

    print ('Probability:', score)


def argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('--solver', type=str, default='fzn-gecode', help="FlatZinc solver")
    return parser


if __name__ == '__main__':
    main(**vars(argparser().parse_args()))