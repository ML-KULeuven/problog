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


%%%%%%%%%
% this is the top level file directly interpreting NLP output for probability problems
%
% it combines the ideas in meta_prob_reducing and meta_prob_constrained, i.e.,
% - collapses histograms before drawing where possible
% - tests properties while drawing
%
% the only "externally used" predicates it defines are the ProbLog interface predicates query/1 and evidence/1
%

% include everything in single place
:- use_module(library(lists)).
:- consult(terms_to_predicates).
:- consult(histograms).
:- consult(setup_aux).
:- consult(input_interface).
:- consult(property_based_merging).
:- consult(constraints).
:- consult(probabilistic_drawing).
:- consult(multiset_constraints).
:- consult(ge_solver_sparse).

%%%%% ProbLog interface
query(X) :- 
	probability(X).

evidence(X) :- 
	observe(X).

%%%%%% compressing histograms for used properties %%%%

% get regular fine grained instance first, then process it (no incoming constraint)
compressed_instance(From,SuperSetC) :-
	instance_constrained(From,none,SuperSet),
	property_compress(From,SuperSet,SuperSetC).

%%%%%% instance_constrained(+SetID,+Constraint)
% interface called from constraint translation above
instance_constrained(SetID,Constraint) :-
	instance_constrained(SetID,Constraint,_).

%%%%% for given Set and Constraint, enumerate Instance satisfying it (and all direct observations on the set)
%%% set is taken with replacement, so no rest
instance_constrained(Set,Constraint,Instance) :-
	drawing_class(Set,From,wr,SType,M,TrialID),
	directly_observed_or_given(Set,Constraint,ScopeC,AllC,AggC),
	compressed_instance(From,SuperSetC),
	draw_constraints_in_out(wr,SType,M,SuperSetC,Instance,_,TrialID,ScopeC,AllC,AggC,ScopeOut,AllOut,AggOut),
	test_instance(Instance,ScopeOut,AllOut,AggOut).  
%%% set is taken without replacement, consider rest's constraints as well
instance_constrained(Set,Constraint,Instance) :-
	drawing_class(Set,From,wor,SType,M,TrialID), 
	directly_observed_or_given(Set,Constraint,ScopeC,AllC,AggC),
	directly_observed(rest(Set),ScopeCRest,AllCRest,AggCRest),
	compressed_instance(From,SuperSetC),
	draw_constraints_in_out(wor,SType,M,SuperSetC,Instance,InstanceRest,TrialID,ScopeC,AllC,AggC,ScopeOut,AllOut,AggOut),
	test_instance(Instance,ScopeOut,AllOut,AggOut),
	test_instance(InstanceRest,ScopeCRest,AllCRest,AggCRest).
%%% set is rest -- same principle as case above but adding incoming constraint to the ones on rest
instance_constrained(rest(Set),Constraint,InstanceRest) :-
	drawing_class(Set,From,wor,SType,M,TrialID),
	directly_observed(Set,ScopeC,AllC,AggC),
	directly_observed_or_given(rest(Set),Constraint,ScopeCRest,AllCRest,AggCRest),
	compressed_instance(From,SuperSetC),
	draw_constraints_in_out(wor,SType,M,SuperSetC,Instance,InstanceRest,TrialID,ScopeC,AllC,AggC,ScopeOut,AllOut,AggOut),
	test_instance(Instance,ScopeOut,AllOut,AggOut),
	test_instance(InstanceRest,ScopeCRest,AllCRest,AggCRest).
%%% set is union -- generate & test for now
instance_constrained(Set,Constraint,Instance) :-
	union(Set,List),
	union_instance(List,[],Instance),
	directly_observed_or_given(Set,Constraint,ScopeC,AllC,AggC),
	test_instance(Instance,ScopeC,AllC,AggC).
%%% set is static - no incoming constraint to take into account
instance_constrained(Set,_,Instance) :-
	static_set(Set),
	static_instance(Set,Instance).%,debugprint(Set,Instance).
instance_constrained(Set,_,Instance) :-
	static_set(Set),
	\+ cstatins(Set),
	error('static instance computation failed ',Set).
%%% set is chosen from supergroup: the only constraints allowed as incoming are chosen(Set,Group) and none
instance_constrained(Set,chosen(Set,Group),Instance) :-
	choose_group(From,Set),
	super_group_instance(From,Super),
	draw_constraints_in_out(wr,set,1,Super,[1-Group],_,cg(From,Set),[],[],[],[],[],[]),
	compressed_instance(Group,Instance).
instance_constrained(Set,none,Instance) :-
	choose_group(From,Set),
	super_group_instance(From,Super),
	draw_constraints_in_out(wr,set,1,Super,[1-Group],_,cg(From,Set),[],[],[],[],[],[]),
	compressed_instance(Group,Instance).


% auxiliary for error catching
cstatins(Set) :-
	static_instance(Set,Instance).

% generate instance of union by generating & merging its parts
union_instance([],Inst,Inst).
union_instance([First|Rest],Acc,Instance) :-
	compressed_instance(First,FI),
	merge_hist(FI,Acc,Next),
	union_instance(Rest,Next,Instance).

% classify the cases where Set is Taken
% drawing_class(+Set,-From,-WType,-SType,-Num,-TrialID)
drawing_class(Set,From,wr,seq,M,take_wr(From,Set)) :-
	take_w(From,Set,M),
	set_is_sequence(Set).
drawing_class(Set,From,wr,set,M,take_wr(From,Set)) :-
	take_w(From,Set,M),
	\+ set_is_sequence(Set).
drawing_class(Set,From,wor,seq,M,take_wor(From,Set)) :-
	take_wo(From,Set,M),
	set_is_sequence(Set).
drawing_class(Set,From,wor,set,M,take_wor(From,Set)) :-
	take_wo(From,Set,M),
	\+ set_is_sequence(Set).


