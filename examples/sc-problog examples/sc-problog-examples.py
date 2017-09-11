#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
*******************************************************************************
File name:          run_tests.py
Author:             Anna Latour
e-mail:             a.l.d.latour@liacs.leidenuniv.nl
Date created:       3/7/2017
Date last modified: 25/8/2017
Python Version:     3.4

Short description:  Script for testing installation of SC-ProbLog.
Prerequisites:      - Obtain academic license and install Gurobi:
                      http://www.gurobi.com/
                    - install Gecode (NOTE: Gecode support will be available asap)
                    - install prerelease version of ProbLog

*******************************************************************************
Currently supported OPTIMISATION types:
    maxSumProb: maximise sum of expectations of queries
    minSumProb: minimise sum of expectations of queries
    maxTheory:  maximise number of decision variables that are True
    minTheory:  minimise number of decision variables that are True

    only problems that are described by the related files below
    only in combination with certain constraint types

*******************************************************************************
Currently (more or less) supported CONSTRAINT types:
    ubSumProb: sum of expectations of a number of queries <= threshold
    lbSumProb: sum of expectations of a number of queries >= threshold
    ubTheory:  number of decision variables that are True <= threshold
    lbTheory:  number of decision variables that are True >= threshold

    only problems that are described by the related files below
    only in combination with certain optimisation types

*******************************************************************************
Related files:
    - search/sdd2Gurobi.py
    - search/sdd2Gecode.py
    - tests/timeout.py

    - tests/viral_marketing_small_example.pl
    - tests/spine_community_16.pl
    - tests/spine_community_27.pl

*******************************************************************************
Gurobi parameters:
    PreSolve off (0), conservative (1), aggressive (2), or
    chosen heuristically by Gurobi (-1).

*******************************************************************************
SDD parameters:
    no minimisation
    default minimisation
    SMP minimisation

*******************************************************************************
Output (optional):
    saved model in .mps file
    saved model in .lp file
    saved formula in .png file
    all in test/results/ subdirectory

*******************************************************************************
Usage example:
    python run_tests.py
"""

from subprocess import check_call
from timeit import default_timer as timer

import os

from problog import get_evaluatable
from problog.program import PrologFile
from problog.logic import Term
from problog.formula import LogicDAG

import sdd
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'../problog/lib/mip/'))
from mipify_sdd import sdd2mip_builder

from timeout import timeout, TimeoutError

maxtime = 3600          # Timeout time

save_formula = True
save_model = True
save_solution = True
resultdir = 'results/'

if (save_formula or save_model or save_solution):

    if not os.path.exists(resultdir):
        os.makedirs(resultdir)


###############################################################################
#                                                                             #
#                              Helper functions                               #
#                                                                             #
###############################################################################


def handler(signum, frame):
    print("Process timed out!")
    raise Exception("Process timed out!")


def count_vars(dag):
    """ Return number of decision variables and number of probabilistic
    variables in the DAG """
    decvars = 0
    probvars = 0
    for _, n, t in dag:
        if t == 'atom':
            if n.probability == Term('?'):
                decvars += 1
            else:
                probvars += 1
    return decvars, probvars


###############################################################################
#                                                                             #
#                              Timing functions                               #
#                                                                             #
###############################################################################


def time_action(function, *args):
    time = maxtime
    result = False
    try:
        start = timer()
        result = function(*args)
        time = timer() - start
    except TimeoutError:
        print('TIMEOUT: Call to ' + function.__name__ + ' timed out\n')
        return False, -1
    except Exception as e:
        print('Exception:', e)
        return False, -1
    return result, time


@timeout(maxtime)
def get_dag(program):
    return LogicDAG.create_from(program)


@timeout(maxtime)
def get_program_sdd(dag, autogc):
    return get_evaluatable('sdd').create_from(dag, sdd_auto_gc=autogc)


@timeout(maxtime)
def get_sdd_formula(program_sdd):
    return program_sdd.to_formula()


@timeout(maxtime)
def get_program_bdd(dag):
    return get_evaluatable('bdd').create_from(dag)


@timeout(maxtime)
def build_model(cp_builder,     # choose from [SDD2Gurobi, SDD2Gecode]        # NOTE: SDD2Gecode not yet supported
                formula,        # formula object obtained from SDD
                optqueries,     # list of positive queries, i.e. (Term, root, 'query') tuples with root in the SDD formula     
                cstqueries,     # list of negative queries, i.e. (Term, root, 'query') tuples with root in the SDD formula     
                threshold,      # int or float
                opttype,        # choose from [maxSumProb, maxTheory, minSumProb, minTheory]
                csttype,        # choose from [ubSumProb, ubTheory, lbSumProb, lbTheory]
                presolve):      # only needed for SDD2Gurobi; choose from [0 (off), 1 (conservative), 2 (aggressive), -1 (heuristically determined by Gurobi)]
    return cp_builder(formula,
                      optqueries=optqueries,
                      cstqueries=cstqueries,
                      threshold=threshold,
                      opttype=opttype,
                      csttype=csttype,
                      presolve=presolve)


@timeout(maxtime)
def solve(cp):
    cp.solve()
    return cp


@timeout(300)
def save_sdd_formula(filename, formula):
    ff = open(resultdir + filename + '.dot', 'w')
    ff.write(sdd_formula.to_dot())
    ff.close()
    check_call(['dot', '-Tpng', resultdir + filename + '.dot', '-o', resultdir + filename + '.png'])


def save_model_to_mps(model, name):
    mps_file = resultdir + name + '.mps'
    model.model.write(mps_file)
    print('Wrote model to', mps_file)
    

def save_model_to_lp(model, name):
    lp_file = resultdir + name + '.lp'
    model.model.write(lp_file)
    print('Wrote model to', lp_file)

    
def print_and_save_results(results, filename):
    ff = open(resultdir + filename, 'a')
    
    methods = list(results.keys())
    settings = list(results[methods[0]].keys())
        
    print('\n' + 80 * '=')
    print('=' + 78 * ' ' + '=')
    print('={:^78}='.format('Total solution times'))
    print('=' + 78 * ' ' + '=')
    print(80 * '=' + '\n')
    
    ff.write('\n' + 80 * '=' + '\n')
    ff.write('=' + 78 * ' ' + '=' + '\n')
    ff.write('={:^78}='.format('Total solution times')  + '\n')
    ff.write('=' + 78 * ' ' + '=' + '\n')
    ff.write(80 * '=' + '\n\n')
    
    header = 'obj         cst        k     objval     '     # TODO: add check value
    for method in methods:
        header = header + '{:17s}'.format(method)
    
    print(header)
    print('-' * len(header))     
    
    ff.write(header + '\n')
    ff.write('-' * len(header) + '\n')     
    
    for (obj, cst) in settings:
        for threshold in sorted(results[methods[0]][(obj, cst)].keys()):
            obj_val = results[methods[0]][(obj, cst)][threshold]['obj_val']
            newline = ''
            
            if obj_val is None:
                newline = '{:12s}{:11s}{:4.1f} no sol   '.format(
                       obj, cst, threshold)
            else:
                newline = '{:12s}{:11s}{:4.1f}{:10.4f}'.format(
                       obj, cst, threshold, results[methods[0]][(obj, cst)][threshold]['obj_val'])
            for method in methods:
                newline = newline + '{:9.4f}s'.format(results[method][(obj, cst)][threshold]['total_time'])
            print(newline)
            ff.write(newline + '\n')
            
        print('.' * len(header))
        ff.write('.' * len(header) + '\n')
    ff.close()

###############################################################################
#                                                                             #
#                                  Run Tests                                  #
#                                                                             #
###############################################################################

###############################################################################
#                                                                             #
#                 First test: Small Viral Marketing example                   #
#                                                                             #
###############################################################################

dag = False
program_sdd = False
sdd_formula = False
setting2qs_and_ths = {('maxSumProb', 'ubSumProb'): {'pos_query_ids' : [0, 1],
                                                    'neg_query_ids' : [2, 3],
                                                    'thresholds' : [0.2, 0.6, 1, 1.4, 1.8],
                                                    'description': 'SETTING 1:  (maxSumProb, ubSumProb)\n' + 
                                                                   'OBJECTIVE:  maximise sum of expectations that persons a and b buy the product\n' + 
                                                                   'CONSTRAINT: the sum of expectations that person c and d buy the product is limited by an upper bound'},
                      ('maxSumProb', 'ubTheory'): {'pos_query_ids' : [0, 1, 2, 3],
                                                   'neg_query_ids' : [],
                                                   'thresholds' : [1, 2, 3],
                                                   'description': 'SETTING 2:  (maxSumProb, ubTheory)\n' + 
                                                                  'OBJECTIVE:  maximise sum of expectations that each perso buys the product\n' + 
                                                                  'CONSTRAINT: target at most k persons directly'},
                      ('maxTheory', 'ubSumProb'): {'pos_query_ids' : [],
                                                   'neg_query_ids' : [0, 1, 2, 3],
                                                   'thresholds' : [0.4, 1.2, 2, 2.8, 3.6],
                                                   'description': 'SETTING 3:  (maxTheory, ubSumProb)\n' + 
                                                                  'OBJECTIVE:  maximise the number of persons targeted directly\n' + 
                                                                  'CONSTRAINT: the sum of expectations that each person buys the product is limited by an upper bound'},
                      ('minSumProb', 'lbSumProb'): {'pos_query_ids' : [0, 1],
                                                    'neg_query_ids' : [2, 3],
                                                    'thresholds' : [0.2, 0.6, 1, 1.4, 1.8],
                                                    'description': 'SETTING 4:  (minSumProb, lbSumProb)\n' + 
                                                                   'OBJECTIVE:  minimise sum of expectations that persons a and b buy the product\n' + 
                                                                   'CONSTRAINT: the sum of expectations that person c and d buy the product is limited by a lower bound'},
                      ('minSumProb', 'lbTheory'): {'pos_query_ids' : [0, 1, 2, 3],
                                                   'neg_query_ids' : [],
                                                   'thresholds' : [1, 2, 3],
                                                   'description': 'SETTING 5:  (minSumProb, lbTheory)\n' + 
                                                                  'OBJECTIVE:  minimise sum of expectations that each person buys the product\n' + 
                                                                  'CONSTRAINT: target at least k persons directly'},
                      ('minTheory', 'lbSumProb'): {'pos_query_ids' : [],
                                                   'neg_query_ids' : [0, 1, 2, 3],
                                                   'thresholds' : [0.4, 1.2, 2, 2.8, 3.6],
                                                   'description': 'SETTING 6:  (minTheory, lbSumProb)\n' + 
                                                                  'OBJECTIVE:  minimise the number of persons targeted directly\n' + 
                                                                  'CONSTRAINT: the sum of expectations that each person buys the product is limited by a lower bound'}
                      }

print('*' * 80)
print('*' + ' ' * 78 + '*')
print('*' + ' ' * 26 + 'Small Viral Marketing test' + ' ' * 26 + '*')
print('*' + ' ' * 78 + '*')
print('*' * 80 + '\n')

prologfile = 'example problems/viral_marketing_small_example.pl'

print('Parse ProbLog program in' + prologfile + '...' )
program = PrologFile(prologfile)

print('\nGround the program...' )
dag, dag_time = time_action(get_dag, program)
if dag:
    print('Program grounded in {:4.2f}s.'.format(dag_time))
    print('DAG size = {:d}'.format(len(dag)))
    
    decvarcount, stochvarcount = count_vars(dag)
    print('Number of relevant decision variables: {:d}'.format(decvarcount))
    print('Number of relevant stochastic variables: {:d}'.format(stochvarcount))
    
else:
    print('Error while grounding, moving on to next example')

results = dict()

# Loop over methods
for builder, autogc, name in [(sdd2mip_builder, False, 'Gurobi-no-mini'),
                              (sdd2mip_builder, True, 'Gurobi-smp-mini')]:

    print('\n' + '/' * 80)
    print('\nStart experiment for method ' + name)
    print('\n' + '/' * 80)
    
    
    results[name] = dict()
    
    if not dag:
        print('No DAG available, move on to next example')
        break
    
    print('\nCompile program to SDD...')
    program_sdd, sdd_time = time_action(get_program_sdd, 
                                        dag, autogc)
    print('SDD compiled in {:4.2f}s'.format(sdd_time))
    
    if program_sdd:
        sddm = program_sdd.get_manager()
        sddm_size = sdd.sdd_manager_size(sddm.get_manager())
        print('SDD manager size = {:d}'.format(sddm_size))
        
        print('\nConvert SDD to formula...')
        sdd_formula, formula_time = time_action(get_sdd_formula, program_sdd)
    else:
        print('Error while compiling SDD, moving on to next example')
    
    if sdd_formula and save_formula:
        print('\nSave SDD formula...')
        filename = 'sddformula-viral-marketing-small-example-{n}'.format(n=name)
        print(filename)
        try:
            save_sdd_formula(filename, sdd_formula)
            print('Saved SDD formula to:\n' + 
                  '\t' + filename + '.dot, and\n' +
                  '\t' + filename + '.png')
        except TimeoutError:
            print('Saving SDD formula timed out.')
        
    if sdd_formula:
        print('Start experiment for different settings')
        
        for ((obj, cst), d) in setting2qs_and_ths.items():
            
            results[name][(obj, cst)] = dict()
            
            print('\n' + '-' * 80)
            print(d['description'])
            print('-' * 80)
            
            print('\nGet opt and cst queries...')
            queries = sdd_formula.labeled()
            optqueries = [queries[i] for i in d['pos_query_ids']]
            cstqueries = [queries[i] for i in d['neg_query_ids']]
        
            print('\nBuild MIP for different values of threshold k and solve:')
        
            # Loop over thresholds
            for threshold in d['thresholds']:
                model = False
                new_model = False
                print('\n' + '-' * 80)
                print('\nBuild model for k = {k}...'.format(k=threshold))
                model, _ = time_action(build_model,
                                       sdd2mip_builder, sdd_formula,
                                       optqueries, cstqueries,
                                       threshold,
                                       obj, cst,
                                       -1)
                     
                if not model:
                    print('Error while building model, moving on to next example')
                    break
                
                if save_model:
                    print('\nSave model...')
                    filename = 'model-viral-marketing-small-example-{o}-{c}-{k}-{n}'.format(o=obj, c=cst, k=threshold, n=name)
                    save_model_to_mps(model, filename)
                    save_model_to_lp(model, filename)
                
                numConstrs = model.numConstrs
                numQConstrs = model.numQConstrs
                print('Model building took {:.2f}s'.format(model.build_time))
                build_time = model.build_time
                print('Model has {nl} linear constraints and {nq} quadratic constraints.'.format(
                        nl=numConstrs, nq=numQConstrs))
                
                print('\nSolve the model...')
                model, _ = time_action(solve, model)
                
                if not model:
                    print('Error while solving model, moving on to next threshold')
                    continue
                
                print('Model solving took {:4.2f}s'.format(model.opt_time))

                if (model.obj_val is not None):
                    print('Objective value: {:4.2f}'.format(model.obj_val))
                    print('Theory size: {:d}'.format(model.nTrue))
                else:
                    print('No solution for threshold {k}'.format(k=threshold))
    
                results[name][(obj, cst)][threshold] = {
                        'obj_val': model.obj_val,
                        'nTrue': model.nTrue,
                        'total_time': dag_time + sdd_time +  formula_time + 
                                      build_time + model.opt_time}
                
                if save_solution:
                    model.save_solution(resultdir + 'solution-viral-marketing-small-example-{o}-{c}-{k}-{n}.txt'.format(o=obj, c=cst, k = threshold, n=name))
 
print_and_save_results(results, 'results-viral-marketing-small-example.txt')

"""
###############################################################################
#                                                                             #
#                       Second test: SPINE community 16                       #
#                                                                             #
###############################################################################

dag = False
program_sdd = False
sdd_formula = False
setting2qs_and_ths = {('maxSumProb', 'ubTheory'): {'pos_query_ids' : range(23),
                                                   'neg_query_ids' : [],
                                                   'thresholds' : [5, 10, 15, 20, 30],
                                                   'description': 'SETTING 1:  (maxSumProb, ubTheory)\n' + 
                                                                  'OBJECTIVE:  maximise sum of expectations for paths between the pairs\n' + 
                                                                  'CONSTRAINT: choose at most k edges in your theory'},
                      ('minTheory', 'lbSumProb'): {'pos_query_ids' : [],
                                                   'neg_query_ids' : range(23),
                                                   'thresholds' : [2.3, 6.9, 11.5, 16.1, 20.7],
                                                   'description': 'SETTING 2:  (minTheory, lbSumProb)\n' + 
                                                                  'OBJECTIVE:  minimise the number number of edges in your theory\n' + 
                                                                  'CONSTRAINT: the sum of expectations of paths between the pairs should be at least equal to some threshold'}
                      }

print('*' * 80)
print('*' + ' ' * 78 + '*')
print('*' + '{:^78s}'.format('SPINE community 16 test') + '*')
print('*' + ' ' * 78 + '*')
print('*' * 80 + '\n')

prologfile = 'tests/spine_community_16.pl'

print('Parse ProbLog program in' + prologfile + '...' )
program = PrologFile(prologfile)

print('\nGround the program...' )
dag, dag_time = time_action(get_dag, program)
if dag:
    print('Program grounded in {:4.2f}s.'.format(dag_time))
    print('DAG size = {:d}'.format(len(dag)))
    
    decvarcount, stochvarcount = count_vars(dag)
    print('Number of relevant decision variables: {:d}'.format(decvarcount))
    print('Number of relevant stochastic variables: {:d}'.format(stochvarcount))
    
else:
    print('Error while grounding, moving on to next example')

results = dict()

# Loop over methods
for builder, autogc, name in [(sdd2mip_builder, False, 'Gurobi-no-mini')]: #,
                              #(sdd2mip_builder, True, 'Gurobi-smp-mini')]:

    print('\n' + '/' * 80)
    print('\nStart experiment for method ' + name)
    print('\n' + '/' * 80)
    
    
    results[name] = dict()
    
    if not dag:
        print('No DAG available, move on to next example')
        break
    
    print('\nCompile program to SDD...')
    program_sdd, sdd_time = time_action(get_program_sdd, 
                                        dag, autogc)
    print('SDD compiled in {:4.2f}s'.format(sdd_time))
    
    if program_sdd:
        sddm = program_sdd.get_manager()
        sddm_size = sdd.sdd_manager_size(sddm.get_manager())
        print('SDD manager size = {:d}'.format(sddm_size))
        
        print('\nConvert SDD to formula...')
        sdd_formula, formula_time = time_action(get_sdd_formula, program_sdd)
    else:
        print('Error while compiling SDD, moving on to next example')
    
    if sdd_formula and save_formula:
        print('\nSave SDD formula...')
        filename = 'sddformula-spine-community-27-{n}'.format(n=name)
        try:
            save_sdd_formula(filename, sdd_formula)
            print('Saved SDD formula to:\n' + 
                  '\t' + resultdir + filename + '.dot, and\n' +
                  '\t' + resultdir + filename + '.png')
        except TimeoutError:
            print('Saving SDD formula timed out.')
        
    if sdd_formula:
        print('Start experiment for different settings')
        
        for ((obj, cst), d) in setting2qs_and_ths.items():
            
            results[name][(obj, cst)] = dict()
            
            print('\n' + '-' * 80)
            print(d['description'])
            print('-' * 80)
            
            print('\nGet opt and cst queries...')
            queries = sdd_formula.labeled()
            optqueries = [queries[i] for i in d['pos_query_ids']]
            cstqueries = [queries[i] for i in d['neg_query_ids']]
        
            print('\nBuild MIP for different values of threshold k and solve:')
        
            # Loop over thresholds
            for threshold in d['thresholds']:
                model = False
                new_model = False
                print('\n' + '-' * 80)
                print('\nBuild model for k = {k}...'.format(k=threshold))
                model, _ = time_action(build_model,
                                       sdd2mip_builder,
                                       sdd_formula,
                                       optqueries, cstqueries,
                                       threshold,
                                       obj, cst,
                                       -1) # TODO: integrate presolve in instructions in to p comments 
                     
                if not model:
                    print('Error while building model, moving on to next example')
                    break
                
                numConstrs = model.numConstrs
                numQConstrs = model.numQConstrs
                print('Model building took {:.2f}s'.format(model.build_time))
                build_time = model.build_time
                print('Model has {nl} linear constraints and {nq} quadratic constraints.'.format(
                        nl=numConstrs, nq=numQConstrs))
                
                print('\nSolve the model...')
                model, _ = time_action(solve, model)
                
                if not model:
                    print('Error while solving model, moving on to next threshold')
                    continue
                
                print('Model solving took {:4.2f}s'.format(model.opt_time))
                
                if (model.obj_val is not None):
                    print('Objective value: {:4.2f}'.format(model.obj_val))
                    print('Theory size: {:d}'.format(model.nTrue))
                else:
                    print('No solution for threshold {k}'.format(k=threshold))
    
                results[name][(obj, cst)][threshold] = {
                        'obj_val': model.obj_val,
                        'nTrue': model.nTrue,
                        'total_time': dag_time + sdd_time +  formula_time + 
                                      build_time + model.opt_time}
                
                if save_solution:
                    model.save_solution(resultdir + 'solution-spine-community-16-{o}-{c}-{k}-{n}-'.format(o=obj, c=cst, k = threshold, n=name))
 
print_and_save_results(results, 'results-spine-community-16.txt')

###############################################################################
#                                                                             #
#                        Third test: SPINE community 27                       #
#                                                                             #
###############################################################################

dag = False
program_sdd = False
sdd_formula = False
setting2qs_and_ths = {('maxSumProb', 'ubSumProb'): {'pos_query_ids' : range(13),
                                                    'neg_query_ids' : range(13, 26),
                                                    'thresholds' : [13 * m for m in [.1, .3, .5, .7, .9]],
                                                    'description': 'SETTING 1:  (maxSumProb, ubSumProb)\n' + 
                                                                   'OBJECTIVE:  maximise sum of expectations that persons a and b buy the product\n' + 
                                                                   'CONSTRAINT: the sum of expectations that person c and d buy the product is limited by an upper bound'},
                      ('maxSumProb', 'ubTheory'): {'pos_query_ids' : range(13),
                                                   'neg_query_ids' : [],
                                                   'thresholds' : [5, 15, 25, 35, 50, 70, 72],
                                                   'description': 'SETTING 2:  (maxSumProb, ubTheory)\n' + 
                                                                  'OBJECTIVE:  maximise sum of expectations that each person buys the product\n' + 
                                                                  'CONSTRAINT: target at most k persons directly'},
                      ('maxTheory', 'ubSumProb'): {'pos_query_ids' : [],
                                                   'neg_query_ids' : range(13, 26),
                                                   'thresholds' : [13 * m for m in [.1, .3, .5, .7, .9]],
                                                   'description': 'SETTING 3:  (maxTheory, ubSumProb)\n' + 
                                                                  'OBJECTIVE:  maximise the number of persons targeted directly\n' + 
                                                                  'CONSTRAINT: the sum of expectations that each person buys the product is limited by an upper bound'},
                      ('minTheory', 'lbSumProb'): {'pos_query_ids' : [],
                                                   'neg_query_ids' : range(13, 26),
                                                   'thresholds' : [13 * m for m in [.1, .3, .5, .7, .9]],
                                                   'description': 'SETTING 4:  (minTheory, lbSumProb)\n' + 
                                                                  'OBJECTIVE:  minimise the number of persons targeted directly\n' + 
                                                                  'CONSTRAINT: the sum of expectations that each person buys the product is limited by a lower bound'}
                      }

print('*' * 80)
print('*' + ' ' * 78 + '*')
print('*' + '{:^78s}'.format('SPINE community 27 test') + '*')
print('*' + ' ' * 78 + '*')
print('*' * 80 + '\n')

prologfile = 'tests/spine_community_27.pl'

print('Parse ProbLog program in' + prologfile + '...' )
program = PrologFile(prologfile)

print('\nGround the program...' )
dag, dag_time = time_action(get_dag, program)
if dag:
    print('Program grounded in {:4.2f}s.'.format(dag_time))
    print('DAG size = {:d}'.format(len(dag)))
    
    decvarcount, stochvarcount = count_vars(dag)
    print('Number of relevant decision variables: {:d}'.format(decvarcount))
    print('Number of relevant stochastic variables: {:d}'.format(stochvarcount))
    
else:
    print('Error while grounding, moving on to next example')

results = dict()

# Loop over methods
for builder, autogc, name in [(sdd2mip_builder, False, 'Gurobi-no-mini')]: #,
                              #(sdd2mip_builder, True, 'Gurobi-smp-mini')]:

    print('\n' + '/' * 80)
    print('\nStart experiment for method ' + name)
    print('\n' + '/' * 80)
    
    
    results[name] = dict()
    
    if not dag:
        print('No DAG available, move on to next example')
        break
    
    print('\nCompile program to SDD...')
    program_sdd, sdd_time = time_action(get_program_sdd, 
                                        dag, autogc)
    print('SDD compiled in {:4.2f}s'.format(sdd_time))
    
    if program_sdd:
        sddm = program_sdd.get_manager()
        sddm_size = sdd.sdd_manager_size(sddm.get_manager())
        print('SDD manager size = {:d}'.format(sddm_size))
        
        print('\nConvert SDD to formula...')
        sdd_formula, formula_time = time_action(get_sdd_formula, program_sdd)
    else:
        print('Error while compiling SDD, moving on to next example')
    
    if sdd_formula and save_formula:
        print('\nSave SDD formula...')
        filename = 'sddformula-spine-community-16-{n}'.format(n=name)
        try:
            save_sdd_formula(filename, sdd_formula)
            print('Saved SDD formula to:\n' + 
                  '\t' + resultdir + filename + '.dot, and\n' +
                  '\t' + resultdir + filename + '.png')
        except TimeoutError:
            print('Saving SDD formula timed out.')
        
    if sdd_formula:
        print('Start experiment for different settings')
        
        for ((obj, cst), d) in setting2qs_and_ths.items():
            
            results[name][(obj, cst)] = dict()
            
            print('\n' + '-' * 80)
            print(d['description'])
            print('-' * 80)
            
            print('\nGet opt and cst queries...')
            queries = sdd_formula.labeled()
            optqueries = [queries[i] for i in d['pos_query_ids']]
            cstqueries = [queries[i] for i in d['neg_query_ids']]
        
            print('\nBuild MIP for different values of threshold k and solve:')
        
            # Loop over thresholds
            for threshold in d['thresholds']:
                model = False
                new_model = False
                print('\n' + '-' * 80)
                print('\nBuild model for k = {k}...'.format(k=threshold))
                model, _ = time_action(build_model,
                                       sdd2mip_builder, sdd_formula,
                                       optqueries, cstqueries,
                                       threshold,
                                       obj, cst,
                                       -1) # TODO: integrate presolve in instructions in to p comments 
                     
                if not model:
                    print('Error while building model, moving on to next example')
                    break
                
                numConstrs = model.numConstrs
                numQConstrs = model.numQConstrs
                print('Model building took {:.2f}s'.format(model.build_time))
                build_time = model.build_time
                print('Model has {nl} linear constraints and {nq} quadratic constraints.'.format(
                        nl=numConstrs, nq=numQConstrs))
                
                print('\nSolve the model...')
                model, _ = time_action(solve, model)
                
                if not model:
                    print('Error while solving model, moving on to next threshold')
                    continue
                
                print('Model solving took {:4.2f}s'.format(model.opt_time))
                
                if (model.obj_val is not None):
                    print('Objective value: {:4.2f}'.format(model.obj_val))
                    print('Theory size: {:d}'.format(model.nTrue))
                else:
                    print('No solution for threshold {k}'.format(k=threshold))
    
                results[name][(obj, cst)][threshold] = {
                        'obj_val': model.obj_val,
                        'nTrue': model.nTrue,
                        'total_time': dag_time + sdd_time +  formula_time + 
                                      build_time + model.opt_time}
                
                if save_solution:
                    model.save_solution(resultdir + 'solution-spine-community-27-{o}-{c}-{k}-{n}-'.format(o=obj, c=cst, k = threshold, n=name))
 
print_and_save_results(results, 'results-spine-community-27.txt')
"""
