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


%%% mapping object level terms in the input onto predicates 
%
% externally used predicates defined in this file:
% builtin_property/3
% is_builtin_property/1
% set_property_in_constraint/3
% constraint_top_set/3
%
%%% PROPERTY terms supported -- add new ones by adding a builtin_property/3 fact and a Prolog definition
% 
% cmp(OP,CONST)
%
% is_even
% is_odd
% is_prime
%
% is_vowel
% is_consonant
%
%%% CONSTRAINT terms supported -- add new ones by adding
% - top level clause linking to instance_constrained
% - set_property_in_constraint/3
% - constraint_top_set/3
%
% and(CONSTRAINT,CONSTRAINT)
% or(CONSTRAINT,CONSTRAINT)
% not(CONSTRAINT)
%
% all(MSET,PROPERTY)
% some(MSET,PROPERTY)
% none(MSET,PROPERTY)
%
% atleast(INT,MSET,PROPERTY)
% atmost(INT,MSET,PROPERTY)
% exactly(INT,MSET,PROPERTY)
% less_than(INT,MSET,PROPERTY)
% more_than(INT,MSET,PROPERTY)
% nth(INT,MSET,PROPERTY)
%
% all_diff(MSET,CLASS)
% all_same(MSET,CLASS)
%
% aggcmp(MSET,CLASS,AGG,PROPERTY)
% aggcmp(MSET,CLASS,AGG,OP,CONST)
%

%%%%% top level query predicates
%% Boolean combinations: unnest
and(Q1,Q2) :-
	call(Q1),
	call(Q2).

or(Q1,_) :-
	call(Q1).
or(_,Q2) :-
	call(Q2).

not(Q) :-
	\+ call(Q).

%% constraints: normalize to instance_constrained(Set,Constraint) for uniform processing
% 1. scope-constraints using exactly, at_least, at_most and nth only
all(SetID,Property) :-
	instance_constrained(SetID,scope(all,Property)). 
none(SetID,Property) :-
	instance_constrained(SetID,scope(exactly(0),Property)).
some(SetID,Property) :-
	not(none(SetID,Property)).
exactly(N,SetID,Property) :-
	instance_constrained(SetID,scope(exactly(N),Property)).
atleast(1,SetID,Property) :-
	some(SetID,Property).
atleast(N,SetID,Property) :-
	N > 1,
	instance_constrained(SetID,scope(at_least(N),Property)).
atmost(N,SetID,Property) :-
	instance_constrained(SetID,scope(at_most(N),Property)).
more_than(N,SetID,Property) :-
	NN is N+1,
	atleast(NN,SetID,Property).
less_than(N,SetID,Property) :-
	NN is N-1,
	atmost(NN,SetID,Property).
nth(N,SetID,Property) :-
	instance_constrained(SetID,scope(nth(N),Property)).
% 2. all-constraints on property classes
all_same(SetID,Class) :-
	instance_constrained(SetID,all(same,Class)).
all_diff(SetID,Class) :-
	instance_constrained(SetID,all(diff,Class)).
% 3. agg-constraints with Boolean test on result
aggcmp(SetID,Class,Agg,Comp,Val) :-
	builtin_property(cmp(Comp,Val),Res,Test),
	instance_constrained(SetID,agg(Class,Agg,Res,Test)).
aggcmp(SetID,Class,Agg,Pred) :-
	builtin_property(Pred,Res,Test),
	instance_constrained(SetID,agg(Class,Agg,Res,Test)).

chosen(Set,Value) :-
	instance_constrained(Set,chosen(Set,Value)).

% set_property_in_constraint(+Constraint,-Set,-Property)
% extract set-property-pairs from given constraint
% for global constraints, return every listed value of the class (SHOULD THIS INCLUDE INFERRED EXTRA?)
set_property_in_constraint(all(S,P),S,P).
set_property_in_constraint(some(S,P),S,P).
set_property_in_constraint(none(S,P),S,P).
set_property_in_constraint(exactly(_,S,P),S,P).
set_property_in_constraint(atleast(_,S,P),S,P).
set_property_in_constraint(atmost(_,S,P),S,P).
set_property_in_constraint(more_than(_,S,P),S,P).
set_property_in_constraint(less_than(_,S,P),S,P).
set_property_in_constraint(nth(_,S,P),S,P).
set_property_in_constraint(all_same(S,C),S,P) :-
	property_definition(C,List),
	member(P,List).
set_property_in_constraint(all_diff(S,C),S,P) :-
	property_definition(C,List),
	member(P,List).
set_property_in_constraint(aggcmp(S,_,min,_,Val),S,cmp(=:=,Val)).
set_property_in_constraint(aggcmp(S,_,min,_,Val),S,cmp(<,Val)).
set_property_in_constraint(aggcmp(S,_,min,_,Val),S,cmp(>,Val)).
set_property_in_constraint(aggcmp(S,_,max,_,Val),S,cmp(=:=,Val)).
set_property_in_constraint(aggcmp(S,_,max,_,Val),S,cmp(<,Val)).
set_property_in_constraint(aggcmp(S,_,max,_,Val),S,cmp(>,Val)).
set_property_in_constraint(aggcmp(S,_,Agg,_,_),S,P) :-
	Agg \= min,
	Agg \= max,
	property_definition(C,List),
	member(P,List).
set_property_in_constraint(aggcmp(S,C,_,_),S,P) :-
	property_definition(C,List),
	member(P,List).
set_property_in_constraint(and(C,_),S,P) :-
	set_property_in_constraint(C,S,P).
set_property_in_constraint(and(_,C),S,P) :-
	set_property_in_constraint(C,S,P).
set_property_in_constraint(or(C,_),S,P) :-
	set_property_in_constraint(C,S,P).
set_property_in_constraint(or(_,C),S,P) :-
	set_property_in_constraint(C,S,P).
set_property_in_constraint(not(C),S,P) :-
	set_property_in_constraint(C,S,P).

%%% constraint_top_set(+SetID,+Constraint,-NormalizedConstraint)
% only succeeds if Constraint is on SetID and of a form that can be used for pruning
% auxiliary for defining direct observations
constraint_top_set(SetID,all(SetID,Property),scope(all,Property)).
%constraint_top_set(SetID,some(SetID,Property),scope(at_least(1),Property)). % this is because pruning messes up with the not(none(...)) translation
constraint_top_set(SetID,none(SetID,Property),scope(exactly(0),Property)).
constraint_top_set(SetID,exactly(N,SetID,Property),scope(exactly(N),Property)).
constraint_top_set(SetID,atleast(N,SetID,Property),scope(at_least(N),Property)) :- N > 1. % this is because pruning messes up with the not(none(...)) translation
constraint_top_set(SetID,atmost(N,SetID,Property),scope(at_most(N),Property)).
constraint_top_set(SetID,more_than(N,SetID,Property),scope(at_least(NN),Property)) :-
	NN is N+1.
constraint_top_set(SetID,less_than(N,SetID,Property),scope(at_most(NN),Property)) :-
	NN is N-1.
constraint_top_set(SetID,nth(N,SetID,Property),scope(nth(N),Property)).
constraint_top_set(SetID,all_same(SetID,Class),all(same,Class)).
constraint_top_set(SetID,all_diff(SetID,Class),all(diff,Class)).
constraint_top_set(SetID,aggcmp(SetID,Class,Agg,Comp,Val),agg(Class,Agg,Res,Test)) :-
	builtin_property(cmp(Comp,Val),Res,Test).
constraint_top_set(SetID,aggcmp(SetID,Class,Agg,Pred),agg(Class,Agg,Res,Test)) :-
	builtin_property(Pred,Res,Test).
% we can also extract parts of observed ands for pruning (but not for or or not)
constraint_top_set(SetID,and(A,B),Norm) :-
	constraint_top_set(SetID,A,Norm).
constraint_top_set(SetID,and(A,B),Norm) :-
	constraint_top_set(SetID,B,Norm).


%%%%%%%%%%%%%%%%%%%%%%%%% supported property tests %%%%%%%%%%%%%%%%%%%%%%%%%
% to add new one, add builtin_property/3 & Prolog definition
is_builtin_property(P) :-
	builtin_property(P,_,_).

% builtin_property(+Term,+Element,-Predicate)
builtin_property(is_even,X,is_even_internal(X)).
builtin_property(is_odd,X,is_odd_internal(X)).
builtin_property(is_prime,X,is_prime_internal(X)).
builtin_property(is_vowel,X,is_vowel_internal(X)).
builtin_property(is_consonant,X,is_consonant_internal(X)).
builtin_property(cmp(Op,Const),X,cmp_internal(X,Op,Const)).

is_even_internal(0).
is_even_internal(N) :-
	integer(N), 
	N > 0, 
	NN is N-1,
	is_odd_internal(NN).

is_odd_internal(N) :-
	integer(N), 
	N > 0, 
	NN is N-1,
	is_even_internal(NN).

is_prime_internal(N) :-
	integer(N),
	check_prime(N,2).

check_prime(N,N).
check_prime(N,D) :-
	D < N,
	N mod D > 0,
	DD is D+1,
	check_prime(N,DD).

cmp_internal(Res,=:=,Val) :-
	Res =:= Val.
cmp_internal(Res,=\=,Val) :-
	Res =\= Val.
cmp_internal(Res,<,Val) :-
	Res < Val.
cmp_internal(Res,>,Val) :-
	Res > Val.
cmp_internal(Res,=<,Val) :-
	Res =< Val.
cmp_internal(Res,>=,Val) :-
	Res >= Val.

is_vowel_internal(a).
is_vowel_internal(e).
is_vowel_internal(i).
is_vowel_internal(o).
is_vowel_internal(u).

is_consonant_internal(X) :-
	member(X,[b,c,d,f,g,h,j,k,l,m,n,p,q,r,s,t,v,w,x,y,z]).

