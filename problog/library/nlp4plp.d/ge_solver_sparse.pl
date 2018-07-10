% Copyright 2017 KU Leuven, DTAI Research Group
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.


% given a list of complex/2 terms, produce an equivalent one in triangular form
% a complex/2 term is a sparse representation of a linear equation system, where the first argument is a list of (coefficient,varname) pairs with non-zero coefficients and the second is a constant
:- use_module(library(lists)).

% top level: given list of complex-terms, return all equalities found by one ge pass
eqs_from_ge(ConstraintsIn,EqsOut) :-
	ge(ConstraintsIn,PropagatedEqs,ConstraintsOut),
	ge_split_rest(ConstraintsOut,OtherEqs,_),
	complex2eq(OtherEqs,OEqs),
	append(PropagatedEqs,OEqs,EqsOut).

complex2eq([],[]).
complex2eq([complex([(C,V)],T)|R],[eq(V,T)|RR]) :-
	C =:= 1,
	complex2eq(R,RR).

/*
% (too high-level) interface from multiset_constraints
fixpoint_prop_solve(PartialSol0,Constraints0,PartialSol1,Constraints1) :-
	add_eq2complex(PartialSol0,Constraints0,InList),
	ge_fixpoint(InList,PartialSol1,Constraints1).
*/

add_eq2complex([],L,L).
add_eq2complex([eq(I,V)|L],Old,[complex([(1,I)],V)|LL]) :-
	add_eq2complex(L,Old,LL).

%%% top level for ge: to get fixpoint, call ge once and continue if necessary
ge_fixpoint(InList,Eqs,Rest) :- %debugprint(in,comes,InList),
	ge(InList,Eqs1,[],Rest1),
	ge_fp(Rest1,Eqs1,Eqs,Rest).

% if splitting the remaining complex list splits off no further equalities, we are done
ge_fp(Rest,Eqs,Eqs,Rest) :-
	ge_split_rest(Rest,[],Rest).
% if it does split off some equalities, we run another round on the (re-ordered) rest and add the new equalities from that round to the ones we already had
ge_fp(Rest,Eqs,EqsFinal,RestFP) :-
	ge_split_rest(Rest,EqsN,RestN),
	EqsN \= [],
	append(EqsN,RestN,InList),
	ge_fixpoint(InList,EqsFP,RestFP),
	append(Eqs,EqsFP,EqsFinal).

%% splitting out eq(-generating) complexes
%% (note: cannot have complex([],_) because incoming list is result of ge, i.e., each complex contains at least the variable it eliminated from the rest)
% empty list - done
ge_split_rest([],[],[]).
% if first complex is eq, it goes to first list
ge_split_rest([complex([(C,Var)],T)|Cs],[complex([(C,Var)],T)|Eqs],Rest) :-
	ge_split_rest(Cs,Eqs,Rest).
% if first complex is zero-inducing, it goes to first list
ge_split_rest([complex(List,Total)|Cs],[complex(List,Total)|Eqs],Rest) :-
	Total =:= 0,
	length(List,N),
	N > 1,
	zero_inducing(List),
	ge_split_rest(Cs,Eqs,Rest).
% otherwise, first complex goes to second list
ge_split_rest([C|Cs],Eqs,[C|Rest]) :-
	general_case(C),
	ge_split_rest(Cs,Eqs,Rest).

%%% top level predicate for single iteration
ge(InList,Eqs,Rest) :-
	ge(InList,Eqs,[],Rest).

% done
ge([],[],R,R).
% drop trivial constraint (empty sum equals 0)
ge([complex([],T)|ComplexNext],Eqs,RIn,ROut) :-
%	T =:= 0,debugprint(drop),
	abs(T) =< 10**(-10),%debugprint(drop,T,ComplexNext),
	ge(ComplexNext,Eqs,RIn,ROut).
% catch inconsistency (empty sum with non-zero value)
ge([complex([],T)|ComplexNext],Eqs,RIn,ROut) :- 
	%	T =\= 0,debugprint(T),
	abs(T) > 10**(-10),
	error('no solution -- empty sum equals ',T).
% if first complex is eq, propagate into accumulator of processed complexes and rest of list (those not yet processed)
ge([complex([(C,Var)],T)|ComplexIn],[eq(Var,NT)|Eqs],RIn,ROut) :- 
	NT is T/C,%debugprint(Var,=,NT),
	ge_solve_single(ComplexIn,Var,complex([(1,Var)],NT),ComplexNext),%debugprint(c,ComplexNext),
	ge_solve_single(RIn,Var,complex([(1,Var)],NT),RNext),%debugprint(r,RNext),
	ge(ComplexNext,Eqs,RNext,ROut).
% if first complex has at least two variables and is zero-inducing, replace it by its induced eq-constraints
ge([complex(List,Total)|ComplexIn],Eqs,RIn,ROut) :-
	Total =:= 0,
	length(List,N),
	N > 1,
	zero_inducing(List,Zeros),
	append(Zeros,ComplexIn,Next),%debugprint(eqi,Next),
	ge(Next,Eqs,RIn,ROut).
% else, propagate normalized version into rest only & add to accumulator of processed complexes
ge([In1|ComplexIn],Eqs,RIn,ROut) :-
	general_case(In1),
	ge_normalize(In1,Normalized1,Var),%debugprint(norm,Normalized1),
	ge_solve_single(ComplexIn,Var,Normalized1,ComplexNext),%debugprint(ComplexNext),
	ge(ComplexNext,Eqs,[Normalized1|RIn],ROut).

% check whether all coefficients have same sign (and produce corresponding equality constraints)
zero_inducing([(C,V)|In],[complex([(1,V)],0)|Out]) :-
	C > 0,
	zero_inducing_plus(In,Out).
zero_inducing([(C,V)|In],[complex([(1,V)],0)|Out]) :-
	C < 0,
	zero_inducing_minus(In,Out).

zero_inducing_plus([],[]).
zero_inducing_plus([(C,V)|In],[complex([(1,V)],0)|Out]) :-
	C > 0,
	zero_inducing_plus(In,Out).

zero_inducing_minus([],[]).
zero_inducing_minus([(C,V)|In],[complex([(1,V)],0)|Out]) :-
	C < 0,
	zero_inducing_minus(In,Out).

zero_inducing(L) :-
	zero_inducing(L,_).

% general case uses more than 1 var and is not zero-inducing (or doesn't sum to 0, in which case it also doesn't induce zeros)
general_case(complex(List,Total)) :-
	length(List,N),
	N > 1,
	Total =\= 0.
general_case(complex(List,Total)) :-
	Total =:= 0,
	length(List,N),
	N > 1,
	\+ zero_inducing(List).

%%% top level predicate ge_solve(+InList,-OutList)
% done
ge_solve([],[]).
% drop trivial constraint (empty sum equals 0)
ge_solve([complex([],T)|ComplexNext],ComplexOut) :-
	T =:= 0,
	ge_solve(ComplexNext,ComplexOut).
% catch inconsistency (empty sum with non-zero value)
ge_solve([complex([],T)|ComplexNext],ComplexOut) :-
	T =\= 0,
	error('no solution -- empty sum equals ',T).
% normalize next constraint & eliminate its var from rest
ge_solve([In1|ComplexIn],[Normalized1|ComplexOut]) :-
	In1 \= complex([],_),				      
	ge_normalize(In1,Normalized1,Var),%debugprint('==========',Normalized1,'=========='),
	ge_solve_single(ComplexIn,Var,Normalized1,ComplexNext),
	ge_solve(ComplexNext,ComplexOut).

% ge_normalize(+Statement,-Statement2,-Var)
% normalize Statement with respect to the first variable with coefficient 1 or -1 (if any), also return that
% used to just normalize for the first variable, but that can cause numerical trouble for some input sequences (e.g., h109)
ge_normalize(complex(Pairs,Total),Complex,Var) :-
	shuffle_for_norm(Pairs,[],Shuffled),
	ge_norm(complex(Shuffled,Total),Complex,Var).
ge_norm(complex([(C,V)|CVs],Tot),complex([(1,V)|CVs],Tot),V) :-
	C =:= 1.
ge_norm(complex([(C,V)|CVs],Tot),complex([(1,V)|DCVs],Total),V) :-
	C =\= 1,
	divide_list(CVs,C,DCVs),
	Total is Tot/C.

% try to get a 1 or -1 as the head of the list
% none found - return accumulator
shuffle_for_norm([],Others,Others).
% this coefficient fits - put the earlier ones to the end of the list and return
shuffle_for_norm([(C,V)|Rest],Others,Shuffled) :-
	abs(C) =:= 1,
	append([(C,V)|Rest],Others,Shuffled).
% this coefficient doesn't fit - move it to accumulator and keep going
shuffle_for_norm([(C,V)|Rest],Others,Shuffled) :-
	abs(C) =\= 1,
	shuffle_for_norm(Rest,[(C,V)|Others],Shuffled).

divide_list([],_,[]).
divide_list([(K,V)|L],C,[(D,V)|LL]) :-
	D is K/C,
	divide_list(L,C,LL).

%% single step of GE
% ge_solve_single(+ListOfStatements, +VariableToEliminate, +CorrespondingStatement, -UpdatedList)
% all done
ge_solve_single([],_,_,[]).
% if var is in complex, need to subtract 
ge_solve_single([complex(OL,OT)|ComplexIn],Var,complex(NL,NT),[complex(UL,UT)|ComplexNext]) :-
	member((C,Var),OL),
	ge_update(OL,C,NL,UL),
	UT is OT - C*NT,%debugprint(complex(UL,UT),complex(OL,OT)),
	ge_solve_single(ComplexIn,Var,complex(NL,NT),ComplexNext).
% if var is not in complex, keep complex and continue
ge_solve_single([complex(OL,OT)|ComplexIn],Var,Normalized,[complex(OL,OT)|ComplexNext]) :-
	\+ is_var_in(OL,Var),
	ge_solve_single(ComplexIn,Var,Normalized,ComplexNext).

is_var_in([(_,V)|_],V).
is_var_in([(_,V)|L],W) :-
	is_var_in(L,W).

is_var_in(V,L1,L2) :-
	is_var_in(L1,V).
is_var_in(V,L1,L2) :-
	is_var_in(L2,V).

% update old coefficient list by subtracting Factor*Norm(alized coefficient list)
ge_update(Old,Factor,Norm,New) :-
	findall(X,is_var_in(X,Old,Norm),Vars),
	sort(Vars,UVars),
	ge_update_coeff(UVars,Old,Factor,Norm,New).

ge_update_coeff([],_,_,_,[]).
ge_update_coeff([V|Vs],Old,Factor,Norm,[(CU,V)|New]) :-
	is_coeff(V,CO,Old),
	is_coeff(V,CN,Norm),
	CU is CO - Factor*CN,
	CU =\= 0,
	ge_update_coeff(Vs,Old,Factor,Norm,New).
ge_update_coeff([V|Vs],Old,Factor,Norm,New) :-
	is_coeff(V,CO,Old),
	is_coeff(V,CN,Norm),
	CU is CO - Factor*CN,
	CU =:= 0,
	ge_update_coeff(Vs,Old,Factor,Norm,New).

is_coeff(V,C,List) :-
	member((C,V),List).
is_coeff(V,0,List) :-
	\+ is_var_in(List,V).




