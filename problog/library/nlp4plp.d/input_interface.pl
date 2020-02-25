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

% collection of predicates colelcting information from the NLP input
%
% externally used predicates defined/extended in this file:
% has_fracs_given/1
% value_in_conjunction/2
% set_is_sequence/1
% take_wo/3 -- this replaces take/3 in the input
% take_w/3  -- this extends take_wr/3 in the input
% static_set/1
% set_attribute_value/3
% given_conditional/2
% given_conditional/3
% given_conditional/4
% el_has_class_value/3
% has_property/2
% has_and_cond/1
% nary_att_set/1
% set_attribute/2
% top_level_class/2
% class_list/2
% multi_attribute_set/1
% nested_sizes/1
% set_propertylist/2
% directly_observed/4
% directly_observed_or_given/5
% no_size_given/1
% given_exactly_u/2
% given_exactly_integer/3
% given_exactly_rel/3
% given_exactly_rel_cond/4
% given_group_property/3
% no_prop_subset_size_given/2
% property_definition/2
%

%%%%%%%%%%%%% supported input predicates %%%%%%%%%%%%%%%%%%%%%%%%%%
% may be freely used outside this file:
group(_) :- fail.
size(_,_) :- fail.
super_group(_) :- fail.
% to be used outside this file for ProbLog interface only:
probability(_) :- fail.
observe(_) :- fail.
% actions that can be used outside, at least for now:
union(_,_) :- fail.
choose_group(_,_) :- fail.
% never use those outside this file, use appropriate interface predicates instead:
given(_) :- fail.
property(_,_) :- fail.
take(_,_,_) :- fail.
take_wr(_,_,_) :- fail.

%%%%%%%%%%%%% extended take_wr/take_wor interface %%%%%
take_w(Set,Taken,N) :-
	take_wr(Set,Taken,N).

take_w(Set,Taken,N) :-
	take(Set,Taken,N),
	no_size_given(Set),
	given(exactly(rel(_,Set),Set,_)).

% we take with replacement if that's written in the input
% or if the input declares a take of size 1 whose rest is not needed
take_w(Set,Taken,1) :-
	take(Set,Taken,1),
	\+ rest_needed(Taken).

% we take witout replacement if the input says so
% and the rules above don't turn it into with replacement
take_wo(A,B,C) :-
	take(A,B,C),
	\+ take_w(A,B,C).

% rest is needed if
% ... we take from it
rest_needed(Taken) :-
	take(rest(Taken),_,_).
% ... we take_wr from it
rest_needed(Taken) :-
	take_wr(rest(Taken),_,_).
% it's in a union
rest_needed(Taken) :-
	union(_,L),
	member(rest(Taken),L).
% it's in an observation
rest_needed(Taken) :-
	observe(Obs),
	set_in_constraint(rest(Taken),Obs).
% it's in a query
rest_needed(Taken) :-
	probability(Obs),
	set_in_constraint(rest(Taken),Obs).

set_in_constraint(S,C) :-
	constraint_top_set(S,C,_).
set_in_constraint(S,not(C)) :-
	set_in_constraint(S,C).
set_in_constraint(S,and(C,_)) :-
	set_in_constraint(S,C).
set_in_constraint(S,and(_,C)) :-
	set_in_constraint(S,C).
set_in_constraint(S,or(C,_)) :-
	set_in_constraint(S,C).
set_in_constraint(S,or(_,C)) :-
	set_in_constraint(S,C).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
has_and_cond(Set) :-
	given(exactly(_,Set,and(_,_))).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nested_sizes(Set) :-
	given(exactly(rel(_,Set,_),Set,_)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
set_attribute(Set,Att) :-
	set_attribute_value(Set,Att,_).

% is this complete ?
set_attribute_value(Set,Class,Prop) :-
	given(exactly(_,Set,AndProp)),
	value_in_conjunction(AndProp,Prop),
	attribute_value(Class,Prop).
set_attribute_value(Set,Class,Prop) :-
	given(one_each(Set,Class)),
	attribute_value(Class,Prop).
set_attribute_value(Prop,Class,Prop) :-
	attribute_value(Class,Prop).
set_attribute_value(and(P,_),Class,Prop) :-
	set_attribute_value(P,Class,Prop).
set_attribute_value(and(_,P),Class,Prop) :-
	set_attribute_value(P,Class,Prop).

value_in_conjunction(Prop,Prop).
value_in_conjunction(and(A,_),Prop) :-
	value_in_conjunction(A,Prop).
value_in_conjunction(and(_,A),Prop) :-
	value_in_conjunction(A,Prop).

% possible attribute-value pairs include the given ones and one "other" value
attribute_value(Class,Prop) :-
	property_definition(Class,List),
	member(Prop,List).
attribute_value(Class,none_of(List)) :-
	property_definition(Class,List).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% temporary (?) test
multi_attribute_set(Set) :-
	set_attribute_value(Set,C1,_),
	set_attribute_value(Set,C2,_),
	C1 \= C2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% does Set have any (easily recognized) non-Boolean attribute?
nary_att_set(Set) :-
	set_attribute(Set,Class),
	non_boolean_property(Class).

% non-Boolean attributes have ...
% ... at least three values
non_boolean_property(Class) :-
	property_definition(Class,[_,_,_|_]).
% ... two explicit values not summing up to the total
non_boolean_property(Class) :-
	property_definition(Class,[A,B]),
	size(Set,S),
	given(exactly(N,Set,A)),
	integer(N),
	given(exactly(M,Set,B)),
	integer(M),
	N + M < S.
non_boolean_property(Class) :-
	property_definition(Class,[A,B]),
	given(exactly(rel(N,Set),Set,A)),
	given(exactly(rel(M,Set),Set,B)),
	N + M < 1.
non_boolean_property(Class) :-
	property_definition(Class,[A,B]),
	given(exactly(rel(N,Set,C1),Set,A)),
	given(exactly(rel(M,Set,C2),Set,B)),
	equivalent_conditions(C1,C2),
	N + M < 1.

equivalent_conditions(C1,C2) :-
	all(X,(value_in_conjunction(X,C1),X\=and(_,_)),L1),
	all(Y,(value_in_conjunction(Y,C2),Y\=and(_,_)),L2),
	sort(L1,S),
	sort(L2,S).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set_propertylist(+Set,-PList)
% PList is sorted list of properties relevant for Set
% used for generating right granularity static set histograms 
set_propertylist(Set,PList) :-
	findall(Prop,relevant_set_property(Set,Prop),Ps),
	sort(Ps,PList).

% a property P is relevant for a set S if ...
% ... they appear together in a constraint
relevant_set_property(Set,Prop) :-
	used_constraint(C),
	set_property_in_constraint(C,Set,Prop).
% ... P is relevant for a set taken from S with replacement
relevant_set_property(Set,Prop) :-
	take_wr(Set,Taken,_),
	relevant_set_property(Taken,Prop).
% ... P is relevant for a set taken from S without replacement
relevant_set_property(Set,Prop) :-
	take(Set,Taken,_),
	relevant_set_property(Taken,Prop).
% ... P is relevant for the rest of taking from S
relevant_set_property(Set,Prop) :-
	take(Set,Taken,_),
	relevant_set_property(rest(Taken),Prop).
% ... S is part of a union for which P is relevant
relevant_set_property(Set,Prop) :-
	union(Union,Parts),
	member(Set,Parts),
	relevant_set_property(Union,Prop).

% constraints queried or observed
used_constraint(C) :-
	probability(C).
used_constraint(C) :-
	observe(C).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Classes is sorted list of property classes declared on set
class_list(SetID,Classes) :-
	findall(X,set_attribute_value(SetID,X,_),L),
	sort(L,Classes).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
top_level_class(Set,Class) :-
	given(exactly(N,Set,Prop)),
	N \= rel(_,_,_),
	attribute_value(Class,Prop),  % experimental extending here; solves the instance not unique, but not enough to make these cases work in general
	given(exactly(rel(_,Set,Prop2),Set,_)),
	attribute_value(Class,Prop2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% static sets have to be declared as group (but exclude dynamic ones that may have been wrongly declared as group as well)
static_set(Set) :-
	group(Set),
	\+ dynamic_set(Set).

% dynamic sets are those taken from others, formed by unions
dynamic_set(Set) :-
	take(_,Set,_).
dynamic_set(rest(Set)) :-
	take(_,Set,_).
dynamic_set(Set) :-
	take_wr(_,Set,_).
dynamic_set(Set) :-
	union(Set,_).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a set is a sequence if and only if it has an nth-constraint
% interface to know whether we draw a set or sequence
set_is_sequence(SetID) :-
	used_constraint(C),
	has_nth_on_set(C,SetID).
has_nth_on_set(nth(_,Set,_),Set).
has_nth_on_set(not(C),Set) :-
	has_nth_on_set(C,SetID).
has_nth_on_set(and(C,_),Set) :-
	has_nth_on_set(C,SetID).
has_nth_on_set(and(_,C),Set) :-
	has_nth_on_set(C,SetID).
has_nth_on_set(or(C,_),Set) :-
	has_nth_on_set(C,SetID).
has_nth_on_set(or(_,C),Set) :-
	has_nth_on_set(C,SetID).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% properties of elements of histograms
% has_property(Element,Property)
% Element is either a property value, a none_of(List) for a property class value range List (which serves just as padding), or a nested and of such
%%% types of properties:
% case Boolean combination: drill down
has_property_intern(Element,not(Prop)) :-
	\+ has_property_intern(Element,Prop).
has_property_intern(Element,and(P1,P2)) :-
	has_property_intern(Element,P1),
	has_property_intern(Element,P2).
has_property_intern(Element,or(P,_)) :-
	has_property_intern(Element,P).
has_property_intern(Element,or(_,P)) :-
	has_property_intern(Element,P).
% case builtin property: call
has_property_intern(Element,Prop) :-
	builtin_property(Prop,Element,Test),
	call(Test).
% else: must be a value of a class associated with the set
has_property_intern(Element,Prop) :-
	Prop \= not(_),
	Prop \= and(_,_),
	Prop \= or(_,_),
	\+ is_builtin_property(Prop),
	attribute_value(Class,Prop),
	set_attribute_value(Element,Class,Prop).
% (note that the latter decomposes and-nested Element, so we don't need to do this here)

has_property(Element,Prop) :-
	Element \= or(_,_),
	has_property_intern(Element,Prop).
has_property(or(E,Es),Prop) :-
	has_property(E,Prop),
	has_property(Es,Prop).
has_property(or(E,Es),Prop) :-
	mixed_prop_case(or(E,Es),Prop),
	error('cannot decide property ',Prop,' on ',or(E,Es)).

mixed_prop_case(or(E,Es),Prop) :-
	flatten_or(or(E,Es),List),
	member(X,List),
	has_property(X,Prop),
	member(Y,List),
	\+ has_property(Y,Prop).

flatten_or(Y,[Y]) :-
	Y \= or(_,_).
flatten_or(or(X,Y),[X|F]) :-
	flatten_or(Y,F).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
has_fracs_given(Set) :-
	given(exactly(rel(_,Set),Set,_)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% normalized interface to the rel/3 cases
% NormalizedCondition is the sorted list of class-value pairs for all values in Cond
% AddedCondition is the class-value pair that is new in Prop compared to those
given_conditional(Frac,Set,NormalizedCondition,AddedCondition) :-
	given(exactly(rel(Frac,Set,Cond),Set,Prop)),
	all(Class-Value,(value_in_conjunction(Cond,Value),attribute_value(Class,Value)),OldPairs),
	sort(OldPairs,NormalizedCondition),
	all(Class1-Value1,(value_in_conjunction(Prop,Value1),attribute_value(Class1,Value1)),NewPairs),
	sort(NewPairs,NP),
	member(AddedCondition,NP),
	\+ member(AddedCondition,NormalizedCondition).
	
given_conditional(Set,NormalizedCondition) :-
	given_conditional(Frac,Set,NormalizedCondition,AddedCondition).
given_conditional(Set,NormalizedCondition,Class) :-
	given_conditional(Frac,Set,NormalizedCondition,Class-_).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
el_has_class_value(Elem,Class,Val) :-
	attribute_value(Class,Val),
	has_property(Elem,Val).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
super_group_instance(SG,Inst) :-
	super_group(SG),
	findall(C-V,given(exactly(C,SG,V)),List),
	sort_histogram(List,Inst).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% directly_observed(+SetID,-Scope,-All,-Agg)
% given SetID, find sorted lists of all direct constraints of each type
% interface for pruning search
directly_observed(SetID,Scope,All,Agg) :-
	findall(scope(Q,P),direct_observation(SetID,scope(Q,P)),S),
	sort(S,Scope),
	findall(all(T,Pr),direct_observation(SetID,all(T,Pr)),A1),
	sort(A1,All),
	findall(agg(Class,Fun,Res,Test),direct_observation(SetID,agg(Class,Fun,Res,Test)),A2),
	sort(A2,Agg).%,debugprint(obs,SetID,Scope,All,Agg).

directly_observed_or_given(SetID,none,Scope,All,Agg) :-
	directly_observed(SetID,Scope,All,Agg).
directly_observed_or_given(SetID,scope(InQ,InP),Scope,All,Agg) :-
	findall(scope(Q,P),direct_observation(SetID,scope(Q,P)),S),
	sort([scope(InQ,InP)|S],Scope),
	findall(all(T,Pr),direct_observation(SetID,all(T,Pr)),A1),
	sort(A1,All),
	findall(agg(Class,Fun,Res,Test),direct_observation(SetID,agg(Class,Fun,Res,Test)),A2),
	sort(A2,Agg).%,debugprint(obsgiven1,SetID,Scope,All,Agg).
directly_observed_or_given(SetID,all(InQ,InP),Scope,All,Agg) :-
	findall(scope(Q,P),direct_observation(SetID,scope(Q,P)),S),
	sort(S,Scope),
	findall(all(T,Pr),direct_observation(SetID,all(T,Pr)),A1),
	sort([all(InQ,InP)|A1],All),
	findall(agg(Class,Fun,Res,Test),direct_observation(SetID,agg(Class,Fun,Res,Test)),A2),
	sort(A2,Agg).%,debugprint(obsgiven2,SetID,Scope,All,Agg).
directly_observed_or_given(SetID,agg(InC,InF,InR,InT),Scope,All,Agg) :-
	findall(scope(Q,P),direct_observation(SetID,scope(Q,P)),S),
	sort(S,Scope),
	findall(all(T,Pr),direct_observation(SetID,all(T,Pr)),A1),
	sort(A1,All),
	findall(agg(Class,Fun,Res,Test),direct_observation(SetID,agg(Class,Fun,Res,Test)),A2),
	sort([agg(InC,InF,InR,InT)|A2],Agg).%,debugprint(obsgiven3,SetID,Scope,All,Agg).


% observations that are not nested allow for pruning their set's instances
% turn them into the form used in constraint/2 for easier handling
direct_observation(Set,NormalizedConstraint) :-
	observe(Obs),%debugprint(Obs),
	constraint_top_set(Set,Obs,NormalizedConstraint).%,debugprint(NormalizedConstraint).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% interface to given(exactly(...))
size_given(Group) :-
	size(Group,_).
no_size_given(Group) :-
	\+ size_given(Group).

% do NOT use given_exactly/3 outside this file, it may miss information
given_exactly(N,Parent,Prop) :-
	given(exactly(N,Parent,Prop)).

given_exactly_u(Parent,Prop) :-
	given(exactly(u,Parent,Prop)).

given_exactly_integer(N,Parent,Prop) :-
	given(exactly(N,Parent,Prop)),
	integer(N).
given_exactly_integer(1,Parent,Value) :-
	given(one_each(Parent,Class)),
	attribute_value(Class,Value).

given_exactly_rel(Frac,Parent,Prop) :-
	given(exactly(rel(Frac,Parent),Parent,Prop)).
% if a fraction is given directly, assume it refers to the parent
given_exactly_rel(Frac,Parent,Prop) :-
	given(exactly(Frac,Parent,Prop)),
	Frac \= rel(_,_),
	Frac \= rel(_,_,_),
	Frac \= u,
	Frac < 1,
	Frac > 0.

given_exactly_rel_cond(Frac,Parent,Cond,Prop) :-
	given(exactly(rel(Frac,Parent,Cond),Parent,Prop)).

prop_subset_size_given(Parent,Prop) :-
	given(exactly(_,Parent,Prop)).
prop_subset_size_given(Parent,Prop) :-
	given(one_each(Parent,Class)),
	attribute_value(Class,Prop).
no_prop_subset_size_given(Parent,Prop) :-
	\+ prop_subset_size_given(Parent,Prop).

% for +Group, find properties in its given-statements
% we assume that in rel_cond, the child prop repeats the condition
given_group_property(Group,Prop) :-
	given_exactly_u(Group,Prop).
given_group_property(Group,Prop) :-
	given_exactly_integer(_,Group,Prop).
given_group_property(Group,Prop) :-
	given_exactly_rel(_,Group,Prop).
given_group_property(Group,Prop) :-
	given_exactly_rel_cond(_,Group,_,Prop).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% interface to property/2
% explicit as list
property_definition(Class,[A|B]) :-
	property(Class,[A|B]).
% implicit as named range
property_definition(Class,List) :-
	property(Class,between(Min,Max)),
	expand_between(Min,Max,List).
% referring to builtin properties; these are only defined if they are used and don't share a value with a user-defined property
property_definition(Class,List) :-
	given(one_each(_,Class)),
	builtin_property_class(Class,List),
	no_conflict_with_given_property(List).

builtin_property_class(builtin(digit),[0,1,2,3,4,5,6,7,8,9]).
builtin_property_class(builtin(alphabet),[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]).

no_conflict_with_given_property(List) :-
	\+ conflict_with_given_property(List).

conflict_with_given_property(List) :-
	property(_,Given),
	member(X,Given),
	member(X,List).

expand_between(N,N,[N]).
expand_between(Min,Max,[Min|List]) :-
	Min < Max,
	Next is Min+1,
	expand_between(Next,Max,List).
