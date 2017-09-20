#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mipify_sdd.py - Converting SDDs to MIPs
--------------------------------------------------

Interface to for converting formulas to MIPs

..
    Part of the ProbLog distribution.

    Copyright 2015 KU Leuven, DTAI Research Group (ProbLog)
    Copyright 2017 KU Leuven, DTAI Research Group;
    UC Louvain, ICTEAM; and Leiden University, LIACS (SC-ProbLog)

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

--------------------------------------------------
File name:          mipify_sdd.py
Author:             Behrouz Babaki, Anna Latour
Date created:       27/3/2017
Date last modified: 20/9/2017
Python Version:     3.4
Short description:  Class for transforming SDD into MIP, and having it solved
                    by Gurobi
Prerequisite:       Obtain academic license and install Gurobi:
                    http://www.gurobi.com/
"""

from gurobipy import Model, LinExpr, GRB
from problog.logic import Term, term2str
from timeit import default_timer as timer
import warnings

# Custom Exceptions
SolutionError = Exception("Gurobi has not found an optimal solution yet")

"""
MIP builder class

*******************************************************************************
Methods:
    __init__
    _case1              dec AND dec
    _case2              dec AND prob
    _case3              dec AND mix
    _case4              prob AND prob
    _case5              prob AND mix
    _case6              mix AND mix (should never occur)
    get_prod            Add conjunction of two variables to model
    get_sum             Add disjunction of arbitrary number of variables
                        to model
    get_var             Returns Gurobi variable
    save_solution       Writes solution to specified file
    set_opt_and_csts    For adding objective function and constraints to Gurobi
    set_parameters      For setting Gurobi parameters
    solve               Solve self.model
"""


class sdd2mip_builder(object):
    def __init__(self,
                 sdd_formula,
                 optqueries=[],
                 cstqueries=[],
                 threshold=0.0,
                 opttype='maxProb',
                 csttype='ubProb',
                 timeout=None,
                 verbosity=0,
                 presolve=0):
        """ Initialise Gurobi MIP / LP.

        Arguments:
        problem     <str> with problem type ('mes', 'bio', 'fas', ...)
        sdd_formula formula object obtained from sdd
        optqueries  <list> of (Term, root, 'query') tuples with root in the sdd
        cstqueries  idem
        threshold   <float> in (0, 1) for bounding constraints
        opttype     <str> with optimisation type ('maxProb', 'maxSet', ...)
        csttype     <str> with constraint type ('ubProb', 'lbProb', ...)
        """

        self.build_time = 0
        self.opt_time = 0
        start = timer()
        self.timeout = timeout

        self.sdd = sdd_formula
        self.optqueries = optqueries    # list of (Term(name), root, 'query') tuples
        self.cstqueries = cstqueries    # list of (Term(name), root, 'query') tuples

        self.opttype = opttype
        self.csttype = csttype
        self.threshold = threshold

        self.model = Model('sdd')       # string serves merely as name

        self.node_types = dict()        # maps the SDD ids of internal nodes
                                        # to their type (conj or disj)
        self.vars = dict()              # maps node id's of the SDD to
                                        # Gurobi variables
        self.var_types = dict()         # maps Gurobi variables to their type:
                                        # dec or mixed
        self.dec_vars = dict()          # maps SDD term names (str) to
                                        # Gurobi variables

        self.or_count = 0               # Number of disjunctions, for
                                        # Gurobi variable naming purposes
        self.and_count = 0              # Number of conjunctions, for
                                        # Gurobi variable naming purposes

        self.set_parameters(timeout, verbosity, presolve)

        # Create Gurobi variables for each decision variable and probabilistic
        for node, term, ttype in self.sdd:
            if ttype == 'atom':
                if term.probability == Term('?'):
                    v = self.model.addVar(lb=0.0, ub=1.0,
                                          vtype=GRB.BINARY,
                                          name=str(term.name))
                    self.vars[node] = v
                    self.dec_vars[term2str(term.name)] = v
                else:
                    self.vars[node] = float(term.probability)
            elif ttype == 'conj':
                self.node_types[node] = 'conj'
            elif ttype == 'disj':
                self.node_types[node] = 'disj'
            self.model.update()

        # Specify optimisation criterion and constraint
        self.set_opt_and_csts()

        self.model.update()
        self.build_time = timer() - start
        self.numConstrs = self.model.NumConstrs
        self.numQConstrs = self.model.NumQConstrs

    def set_opt_and_csts(self):
        """ Translate optimisation and constraint types together with the
        corresponding queries to Objective functions and Constraints in Gurobi.
        """
        
        assert (self.opttype, self.csttype) in [('maxSumProb', 'ubSumProb'),
                                                ('maxSumProb', 'ubTheory'),
                                                ('maxTheory', 'ubSumProb'),
                                                ('minSumProb', 'lbSumProb'),
                                                ('minSumProb', 'lbTheory'),
                                                ('minTheory', 'lbSumProb')], \
               "Invalid or nonsensical combination of optimisation type and constraint type"

        # First add optimisation criterion
        if self.opttype in ['maxSumProb', 'minSumProb']:
            assert self.optqueries, \
                'Specify optimisation queries for maxSumProb'
            opt_vars = [self.get_var(r) for (_, r, _) in self.optqueries]
            expr = LinExpr([(1, var) for var in opt_vars])
            if self.opttype == 'maxSumProb':
                self.model.setObjective(expr, GRB.MAXIMIZE)
            else:
                self.model.setObjective(expr, GRB.MINIMIZE)

        elif self.opttype in ['maxTheory', 'minTheory']:
            assert not self.optqueries, \
                'No optimisation queries required for maxTheory'
            expr = LinExpr([(1, var) for var in self.dec_vars.values()])
            if self.opttype == 'maxTheory':
                self.model.setObjective(expr, GRB.MAXIMIZE)
            else:
                self.model.setObjective(expr, GRB.MINIMIZE)
        else:
            raise ValueError("Choose optimisation type from "
                             "['maxSumProb', 'maxTheory', 'minSumProb', 'minTheory']")

        # Then add constraints
        if self.csttype in ['ubSumProb', 'lbSumProb']:
            assert len(self.cstqueries) >= 1, \
                'ubSumProb must have at least one constraint'
            cst_vars = [self.get_var(r) for (_, r, _) in self.cstqueries]
            expr = LinExpr([(1, var) for var in cst_vars])
            if self.csttype == 'ubSumProb':
                self._cst_var = self.model.addConstr(expr <= self.threshold)
            else:
                self._cst_var = self.model.addConstr(expr >= self.threshold)

        elif self.csttype in ['ubTheory', 'lbTheory']:
            expr = LinExpr([(1, var) for var in self.dec_vars.values()])
            if self.csttype == 'ubTheory':
                self._cst_var = self.model.addConstr(expr <= self.threshold)
            else:
                self._cst_var = self.model.addConstr(expr >= self.threshold)


    def set_parameters(self, timeout, verbosity, presolve):
        self.model.Params.OutputFlag = verbosity
        if self.timeout:
            self.model.Params.TimeLimit = timeout

        self.model.Params.Presolve = int(presolve)
        self.model.update()

    def solve(self):
        self.model.optimize()

        self.opt_time = self.model.Runtime
        self.obj_val = None
        self.status = self.model.status
        if self.status == GRB.OPTIMAL:
            self.obj_val = self.model.ObjVal

            self.solution = {name: None for name in self.dec_vars.keys()}
            for (name, var) in self.dec_vars.items():
                self.solution[name] = var.X
            self.nTrue = len([name for name in self.solution if bool(self.solution[name]) == True])
            self.cst_val = self._cst_var.RHS
        else:
            self.solution = {name: False for name in self.dec_vars.keys()}
            self.nTrue = -1
            self.cst_val = self._cst_var.RHS

    def get_var(self, node):
        if node in self.vars:
            return self.vars[node]

        children = self.sdd.get_node(node).children

        if self.node_types[node] == 'conj':
            v = self.get_prod(children)
        else:
            v = self.get_sum(children)
        self.vars[node] = v
        return v

    def _case1(self, dec1, dec2, decch1, decch2):
        """
        :param dec1:    decision variable
        :param dec2:    decision variable
        :param decch1:  is dec1 negated (False) or not (True)
        :param decch2:  is dec2 negated (False) or not (True)
        """
        z = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY,
                              name='and{ac}'.format(ac=self.and_count))
        self.and_count += 1
        self.model.update()

        if decch1 and decch2:
            self.model.addConstr(z <= dec1)
            self.model.addConstr(z <= dec2)
            self.model.addConstr(z >= dec1 + dec2 - 1.0)
        elif decch1 and not decch2:
            self.model.addConstr(z <= dec1)
            self.model.addConstr(z <= 1.0 - dec2)
            self.model.addConstr(z >= dec1 + (1.0 - dec2) - 1.0)
        elif not decch1 and decch2:
            self.model.addConstr(z <= 1.0 - dec1)
            self.model.addConstr(z <= dec2)
            self.model.addConstr(z >= (1.0 - dec1) + dec2 - 1.0)
        elif not decch1 and not decch2:
            self.model.addConstr(z <= 1.0 - dec1)
            self.model.addConstr(z <= 1.0 - dec2)
            self.model.addConstr(z >= (1.0 - dec1) + (1.0 - dec2) - 1.0)

        self.model.update()
        return z

    def _case2(self, decvar, probvar, decch, probch):
        """
        :param decvar:   decision variable
        :param probvar:  probability (float)
        :param decch:    is decvar negated (False) or not (True)
        :param probch:   is probvar negated (False) or not (True)
        """
        z = self.model.addVar(lb=0.0, ub=probvar, vtype=GRB.CONTINUOUS,
                              name='and{ac}'.format(ac=self.and_count))
        self.and_count += 1
        self.model.update()

        if decch and probch:
            self.model.addConstr(z == decvar * probvar)
        elif decch and not probch:
            self.model.addConstr(z == decvar * (1.0 - probvar))
            z.ub = (1 - probvar)
        elif not decch and probch:
            self.model.addConstr(z == (1.0 - decvar) * probvar)
        elif not decch and not probch:
            self.model.addConstr(z == (1.0 - decvar) * (1 - probvar))
            z.ub = (1 - probvar)
        self.model.update()
        return z

    def _case3(self, decvar, mixvar, decch, mixch):
        """
        :param decvar:  decision variable
        :param mixvar:  decision variable or internal variable
        :param decch:   is decvar negated (False) or not (True)
        :param mixch:   is mixvar negated (False) or not (True)

        linearisation method for case (decch and mixch) from:
        http://orinanobworld.blogspot.be/2010/10/binary-variables-and-quadratic-terms.html
        """

        lbm = mixvar.lb
        ubm = mixvar.ub
        z = self.model.addVar(lb=0.0, ub=ubm, vtype=GRB.CONTINUOUS,
                              name='and{ac}'.format(ac=self.and_count))
        self.and_count += 1
        self.model.update()

        if decch and mixch:
            # correct 29 mrt 2017
            self.model.addConstr(z <= decvar * ubm)                 # z <= ubm if decvar, z <= 0 if not decvar
            self.model.addConstr(z >= decvar * lbm)                 # z >= lbm if decvar, z >= 0 if not decvar
            self.model.addConstr(z <= mixvar - lbm * (1 - decvar))  # z <= mixvar if decvar, z <= mixvar - lbm if not decvar
            self.model.addConstr(z >= mixvar - ubm * (1 - decvar))  # z >= mixvar if decvar, z >= mixvar - ubm if not decvar
        elif decch and not mixch:
            # correct 29 mrt 2017
            self.model.addConstr(z <= decvar * (1 - lbm))                       # z <= 1 - lbm if decvar, z <= 0 if not decvar
            self.model.addConstr(z >= decvar * (1 - ubm))                       # z >= 1 - ubm if decvar, z >= 0 if not decvar
            self.model.addConstr(z <= (1 - mixvar) - (1 - ubm) * (1 - decvar))  # z <= (1-mixvar) if decvar, z <= (1-mixvar) - (1-lbm) if not decvar
            self.model.addConstr(z >= (1 - mixvar) - (1 - lbm) * (1 - decvar))
            z.ub = (1 - lbm)
        elif not decch and mixch:
            # correct 29 mrt 2017
            self.model.addConstr(z <= (1 - decvar) * ubm)
            self.model.addConstr(z >= (1 - decvar) * lbm)
            self.model.addConstr(z <= mixvar - lbm * decvar)
            self.model.addConstr(z >= mixvar - ubm * decvar)
        elif not decch and not mixch:
            # correct 29 mrt 2017
            self.model.addConstr(z <= (1 - decvar) * (1 - lbm))
            self.model.addConstr(z >= (1 - decvar) * (1 - ubm))
            self.model.addConstr(z <= (1 - mixvar) - (1 - ubm) * decvar)
            self.model.addConstr(z >= (1 - mixvar) - (1 - lbm) * decvar)
            z.ub = (1 - lbm)

        self.model.update()
        return z

    def _case4(self, probvar1, probvar2, probch1, probch2):
        """
        :param probvar1:  probability (float)
        :param probvar2:  probability (float)
        :param probch1:   is probvar1 negated (False) or not (True)
        :param probch2:   is probvar2 negated (False) or not (True)
        """

        if probch1 and probch2:
            return probvar1 * probvar2
        elif probch1 and not probch2:
            return probvar1 * (1 - probvar2)
        elif not probch1 and probch2:
            return (1 - probvar1) * probvar2
        elif not probch1 and not probch2:
            return (1 - probvar1) * (1 - probvar2)

    def _case5(self, probvar, mixvar, probch, mixch):
        """
        :param probvar: probability (float)
        :param mixvar:  internal node
        :param probch:  is probvar negated (False) or not (True)
        :param mixch:   is mixvar negated (False) or not (True)
        """

        lbm = mixvar.lb
        ubm = mixvar.ub
        z = self.model.addVar(lb=lbm*probvar, ub=ubm*probvar,
                              vtype=GRB.CONTINUOUS,
                              name='and{ac}'.format(ac=self.and_count))
        self.and_count += 1
        self.model.update()

        if probch and mixch:
            self.model.addConstr(z == probvar * mixvar)
        elif probch and not mixch:
            self.model.addConstr(z == probvar * (1 - mixvar))
            z.lb = probvar * (1 - ubm)
            z.ub = probvar * (1 - lbm)
        elif not probch and mixch:
            self.model.addConstr(z == (1 - probvar) * mixvar)
            z.lb = (1 - probvar) * lbm
            z.ub = (1 - probvar) * ubm
        elif not probch and not mixch:
            self.model.addConstr(z == (1 - probvar) * (1 - mixvar))
            z.lb = (1 - probvar) * (1 - ubm)
            z.ub = (1 - probvar) * (1 - lbm)

        self.model.update()
        return z

    def _case6(self, v1, v2, ch1, ch2):
        """
        :param v1:  mixed node
        :param v2:  mixed node
        :param ch1: is v1 negated (False) or not (True)
        :param ch2: is v2 negated (False) or not (True)
        """

        warnings.warn('Two mixed nodes!', Warning)
#        print('v1 = {n1}, v2 = {n2}'.format(n1=v1.VarName, n2=v2.VarName))

        lb1 = v1.lb
        ub1 = v1.ub
        lb2 = v2.lb
        ub2 = v2.ub
        z = self.model.addVar(lb=lb1*lb2, ub=ub1*ub2, vtype=GRB.CONTINUOUS,
                              name='and{ac}'.format(ac=self.and_count))
        self.and_count += 1
        self.model.update()

        if ch1 and ch2:
            self.model.addConstr(z == v1 * v2)
        elif ch1 and not ch2:
            self.model.addConstr(z == v1 * (1 - v2))
            z.lb = lb1 * (1 - ub2)
            z.ub = ub1 * (1 - lb2)
        elif not ch1 and ch2:
            self.model.addConstr(z == (1 - v1) * v2)
            z.lb = (1 - ub1) * lb2
            z.ub = (1 - lb1) * ub2
        elif not ch1 and not ch2:
            self.model.addConstr(z == (1 - v1) * (1 - v2))
            z.lb = (1 - ub1) * (1 - ub2)
            z.ub = (1 - lb1) * (1 - lb2)

        self.model.update()
        return z

    def get_prod(self, children):
        [v1, v2] = [self.get_var(abs(i)) for i in children]
        [ch1, ch2] = [i > 0 for i in children]

        """
        cases:
        1)  dec AND dec
        2)  dec AND prob
        3)  dec AND mix
        4)  prob AND prob
        5)  prob AND mix
        6)  mix AND mix (should never hapen)

        And then each of the children might be negated or not.
        """

        if isinstance(v1, float):                       # v1 is probabilistic
            if isinstance(v2, float):                   # v2 is probabilistic
                return self._case4(v1, v2, ch1, ch2)
            elif v2.vtype == GRB.BINARY:                # v2 is decision
                return self._case2(v2, v1, ch2, ch1)
            else:                                       # v2 is mixed
                return self._case5(v1, v2, ch1, ch2)

        elif v1.vtype == GRB.BINARY:                    # v1 is decision
            if isinstance(v2, float):                   # v2 is probabilistic
                return self._case2(v1, v2, ch1, ch2)
            elif v2.vtype == GRB.BINARY:                # v2 is decision
                return self._case1(v1, v2, ch1, ch2)
            else:                                       # v2 is mixed
                return self._case3(v1, v2, ch1, ch2)

        else:                                           # v1 is mixed
            if isinstance(v2, float):                   # v2 is probabilistic
                return self._case5(v2, v1, ch2, ch1)
            elif v2.vtype == GRB.BINARY:                # v2 is decision
                return self._case3(v2, v1, ch2, ch1)
            else:                                       # v2 is internal
                return self._case6(v1, v2, ch1, ch2)    # THIS SHOULD NEVER OCCUR

    def get_sum2(self,v1,v2,ch1,ch2):
        # constants need to be pre-processed, otherwise we will create variables for a part of the v-tree that contains only
        # probabilities, which will create a product between variables.
        if isinstance(v1,float) and isinstance(v2,float):
            if ch1 and ch2:
                return v1 + v2
            elif ch1 and not ch2:
                return v1 + (1 - v2)
            elif not ch1 and ch2:
                return (1 - v1) + v2
            else:
                return (1 - v1) + (1 - v2)
        if ( isinstance(v1,float) or v1.vtype == GRB.CONTINUOUS ) and  ( isinstance(v2,float) or v2.vtype == GRB.CONTINUOUS ):
            # both variables are probabilities, we can sum up
            z = self.model.addVar(lb=0, ub=1.0, vtype=GRB.CONTINUOUS,name='or{ac}'.format(ac=self.or_count))
            self.or_count += 1
            self.model.update()
            if ch1 and ch2:
                self.model.addConstr(z == v1 + v2)
            elif ch1 and not ch2:
                self.model.addConstr(z == v1 + (1 - v2))
            elif not ch1 and ch2:
                self.model.addConstr(z == (1 - v1) + v2)
            else:
                self.model.addConstr(z == (1 - v1) + (1 - v2))
        else:
            if v1.vtype == GRB.BINARY and v2.vtype == GRB.BINARY:
                z = self.model.addVar(lb=0, ub=1.0, vtype=GRB.BINARY, name='or{ac}'.format(ac=self.or_count))
            else:
                z = self.model.addVar(lb=0, ub=1.0, vtype=GRB.CONTINUOUS, name='or{ac}'.format(ac=self.or_count))
            self.or_count += 1
            self.model.update()
            if ch1:
                t1 = v1
            else:
                t1 = 1 - v1
            if ch2:
                t2 = v2
            else:
                t2 = 1 - v2
            self.model.addConstr(z >= t1)
            self.model.addConstr(z >= t2)
            self.model.addConstr(z <= t1+t2)
        return z


    def get_sum(self, children):
        """
        children   list of ints
        v1         <class 'gurobi.Var'>
        ch1        <type 'bool'>
        """
        if len(children) == 2:
            [v1, v2] = [self.get_var(abs(i)) for i in children]
            [ch1, ch2] = [i > 0 for i in children]
            return self.get_sum2 ( v1, v2, ch1, ch2 )
        elif len(children) > 2:
            v1 = self.get_var(abs(children[0]))
            ch1 = (children[0] > 0)
            v2 = self.get_sum(children[1:])
            ch2 = True
            return self.get_sum2 ( v1, v2, ch2, ch2 )

    def save_solution(self, filename):
        # Check if optimal value has indeed been found
        sf = open(filename, 'a')
        if self.model.Status == GRB.OPTIMAL:                
            sf.write('varname                       assignment\n')
            sf.write('----------------------------------------\n')
            for (name, val) in sorted(self.solution.items(), reverse=True):
                sf.write('{:<30s}{:<s}\n'.format(name, str(bool(val))))
            sf.write('\n# True = {nt}\n\n'.format(nt=self.nTrue))
        else:
            sf.write('ERROR: Gurobi did not find an optimal solution yet\n\n')
            warnings.warn('ERROR: Gurobi did not find an optimal solution yet',
                          UserWarning)
        sf.close()
