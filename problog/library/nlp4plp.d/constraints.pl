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



% externally used predicates defined in this file:
% test_instance/2
%



%%%%%%%%%%%%%%%%% constraint testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% test a single constraint on an Instance (histogram or sequence; turn the latter into histogram before proceeding unless the constraint is scope nth)
test_instance(Instance,scope(nth(N),Prop)) :-
	test_nth(1,N,Instance,Prop).
test_instance(Instance,Constraint) :-
	is_histogram(Instance), 
	test_histogram(Instance,Constraint).
test_instance(Instance,Constraint) :-
	Constraint \= scope(nth(_),_),
	\+ is_histogram(Instance),
	list2hist(Instance,Hist),
	test_histogram(Hist,Constraint).

test_histogram([],scope(all,_)).
test_histogram([_-V|L],scope(all,Prop)) :-
	has_property(V,Prop),
	test_histogram(L,scope(all,Prop)).
test_histogram(Instance,scope(Quant,Prop)) :-
	Quant \= all,
	count_hist_prop(Instance,Prop,0,N),
	check_scope(N,Quant).
test_histogram(Instance,all(same,Class)) :-
	test_all_same(Instance,Class).
test_histogram(Instance,all(diff,Class)) :-
	test_all_diff(Instance,Class).
% this used to have an extra line to catch a unification bug in (find)all, but seems fine now
test_histogram(Instance,agg(Class,Agg,Res,Test)) :-
	aggregate(Instance,Class,Agg,Res),
	call(Test).

% test all three constraint groups on an Instance (histogram or sequence)
test_instance(Instance,Scopes,Alls,Aggs) :-
	test_instance_list(Instance,Scopes),
	test_instance_list(Instance,Alls),
	test_instance_list(Instance,Aggs).
test_instance_list(_,[]).
test_instance_list(Instance,[F|R]) :-
	test_instance(Instance,F),
	test_instance_list(Instance,R).

%%%%% scope constraints %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test_nth(This,N,[_|Rest],Prop) :-
	This < N,
	Next is This+1,
	test_nth(Next,N,Rest,Prop).
test_nth(N,N,[Element|_],Prop) :-
	has_property(Element,Prop).  

count_hist_prop([],_,N,N).
count_hist_prop([Count-Element|Rest],Prop,In,Out) :-
	has_property(Element,Prop),
	Next is In+Count,
	count_hist_prop(Rest,Prop,Next,Out).
count_hist_prop([Count-Element|Rest],Prop,In,Out) :-
	\+ has_property(Element,Prop),
	count_hist_prop(Rest,Prop,In,Out).

check_scope(N,exactly(N)).
check_scope(N,at_least(M)) :-
	N >= M.
check_scope(N,at_most(M)) :-
	N =< M.

%%%%% all-constraints %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test_all_same([],_).
test_all_same([_-Element|Rest],Class) :-
	set_attribute_value(Element,Class,Value),
	test_all_same_given(Rest,Class,Value).
test_all_same_given([],_,_).
test_all_same_given([_-Element|Rest],Class,Value) :-
	set_attribute_value(Element,Class,Value),
	test_all_same_given(Rest,Class,Value).

% note: if elements are repeated, they are not different
test_all_diff([],_).
test_all_diff([1-Element|Rest],Class) :-
	set_attribute_value(Element,Class,Value),
	test_all_diff_given(Rest,Class,[Value]).
test_all_diff_given([],_,_).
test_all_diff_given([1-Element|Rest],Class,Seen) :-
	set_attribute_value(Element,Class,Value),
	\+ member(Value,Seen),
	test_all_diff_given(Rest,Class,[Value|Seen]).
	
%%%%% aggregation (on non-empty histograms only) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% reduce average to sum
aggregate(Instance,Class,average,Result) :-
	aggregate(Instance,Class,sum,Sum),
	histogram_size(Instance,N),
	Result is Sum/N.
%% do other aggregates (sum,product,min,max) by accumulation	
aggregate([I|Instance],Class,Agg,Result) :-
	Agg \= average,
	I \= _-none_of(_),
	init_agg(I,Class,Agg,Acc),
	aggregate(Instance,Class,Agg,Acc,Result).
aggregate([_-none_of(_)|_],Class,Agg,_) :-
	error('cannot aggregate with unknown values ',Class,' ',Agg).
% accumulate
aggregate([],_,_,Res,Res).
aggregate([Count-Element|Instance],Class,Agg,Acc,Result) :-
	Element \= none_of(_),
	set_attribute_value(Element,Class,Value),  
	add_to_agg(Count,Value,Agg,Acc,Next),  
	aggregate(Instance,Class,Agg,Next,Result).
aggregate([_-none_of(_)|_],Class,Agg,_,_) :-
	error('cannot aggregate with unknown values',Class,' ',Agg).

% count and type influence initialization
init_agg(_-Element,Class,min,Value) :-
	set_attribute_value(Element,Class,Value).  
init_agg(_-Element,Class,max,Value) :-
	set_attribute_value(Element,Class,Value).  
init_agg(Count-Element,Class,sum,N) :- 
	set_attribute_value(Element,Class,Value),
	N is Count*Value.
init_agg(Count-Element,Class,product,N) :-
	set_attribute_value(Element,Class,Value),
	N is Value**Count.

add_to_agg(Count,Value,sum,In,Out) :-
	Out is In+Value*Count.
add_to_agg(Count,Value,product,In,Out) :-
	Out is In*(Value**Count).
add_to_agg(_,Value,min,In,Out) :-
	Out is min(In,Value).
add_to_agg(_,Value,max,In,Out) :-
	Out is max(In,Value).
	 
