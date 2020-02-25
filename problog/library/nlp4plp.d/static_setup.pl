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


% reasoning based on the input from the nlp part
%
% computes
% - a tree representation of the input set structure node/3
% - the sizes of all tree nodes based on partition constraints node_size/2
% - the properties of all tree nodes based on inheritance from anchestors has_property/2
% - histogram representations of any cut through the subtree rooted at a given node histogram/2
% - histogram representation of all leaves of the subtree rooted at a given node finest_histogram/2

% load libraries
:- use_module(library(lists)).
:- consult(histograms).

% avoid problems with undefined input predicates
partition(_,_) :- fail.
size(_,_) :- fail.
tuple(_) :- fail.
has_property(X,X). % by default, allow shortcut using set names as properties 
takeW(_,_) :- fail.
take(_,_) :- fail.
take(_,_,_) :- fail. 
question(_,_) :- fail.
observe(_) :- fail.

%%%%
% node(ID,ParentID,ListOfChildIDs)
% computed from partition/2 in input
% provides hierarchical structure of sets
%% root
node(N,'*',K) :- 
	partition(N,K),
	\+ is_child_of(N,_).
%% inner node
node(N,P,K) :- 
	is_child_of(N,P),
	partition(N,K).
%% leaf
node(N,P,[]) :- 
	is_child_of(N,P),
	\+ partition(N,_).
%% leaf and root
node(N,'*',[]) :- 
	size(N,_),
	\+ partition(N,_),
	static_set(N).

is_child_of(N,P) :-
	partition(P,Kids),
	member(N,Kids).

%%%%
% node_size(ID,Size)
% computed from size/2 in input and node/3 defined above
% provides sizes for all sets (as far as they can be inferred from input)
% requires tabling to avoid mutual recursion for the parents-sibling case (?)
%% given explicitly as fraction or otherwise: take
node_size(N,S) :- 
	node(N,_,_),
	size(N,fraction(Frac,Node)),
	node_size(Node,Size),
	S is integer(Frac*Size).
node_size(N,S) :-  
	node(N,_,_),
	size(N,S),
	S \= fraction(_,_).
%% not given explicitly, node has children: try summing their node_sizes
node_size(N,S) :-  
	node(N,_,Kids),
	\+ size(N,_),
	Kids \= [],
	sum_sizes(Kids,0,S).
%% not given explicitly: try subtracting the siblings' node_sizes from the one of the parent
node_size(N,S) :-  
	node(N,Parent,_),
	\+ size(N,_),
	node_size(Parent,PS),
	node(Parent,_,Kids),
	take_out(Kids,N,Siblings),
	sum_sizes(Siblings,0,SibS),
	S is PS-SibS.

take_out([El|From],El,From).
take_out([H|T],E,[H|TT]) :-
	H \= E,
	take_out(T,E,TT).

sum_sizes([],A,A).
sum_sizes([H|T],Acc,Res) :-
	node_size(H,S),
	Next is Acc+S,
	sum_sizes(T,Next,Res).

% being a tuple is inherited
tuple(Set) :-
	static_set(Set),
	node(Set,P,_),
	tuple(P).

%% new top level predicate:
set_size(Set,S) :-
	size(Set,S),
	S \= fraction(_,_).
set_size(Set,S) :-
	node_size(Set,S).
set_size(Set,S) :-
	take(From,Taken,Set),
	set_size(From,F),
	set_size(Taken,T),
	S is F-T.

%%%%
% has_property(ID,Prop)
% - adds inherited properties from ancestors to the one given as input,
% - handles nested binary and and or of properties, and constant 'true'
% - does list case for both representations
%% any property (except not(P), which is closed world) on set identifier: inherited from parent?
has_property(Node,Prop) :-
	\+ is_a_not(Prop),
	\+ is_list(Node),
	node(Node,Parent,_),
	has_property(Parent,Prop).
%% logical AND on set identifier: 
has_property(Node,and(P1,P2)) :-
	\+ is_list(Node),
	has_property(Node,P1),
	has_property(Node,P2).
%% logical OR on set identifier: 
has_property(Node,or(P1,_)) :-
	\+ is_list(Node),
	has_property(Node,P1).
has_property(Node,or(_,P1)) :-
	\+ is_list(Node),
	has_property(Node,P1).
%% logical NOT on set identifier: 
has_property(Node,not(P1)) :-
	\+ is_list(Node),
	\+ has_property(Node,P1).
%% every set has property true
has_property(_,true).

%%% quantified versions for dynamic sets 
%% any property on list or histogram of set identifiers (ALL case)
has_property(all,[],_).
has_property(all,[H|T],P) :- 
	H \= _-_,
	has_property(H,P), 
	has_property(all,T,P).
has_property(all,[_-H|T],P) :- 
	has_property(H,P), 
	has_property(all,T,P).
/*
%% any property on list or histogram of set identifiers (NONE case)
has_property(none,[],_).
has_property(none,[H|T],P) :- 
	H \= _-_,
	\+ has_property(H,P), 
	has_property(none,T,P).
has_property(none,[_-H|T],P) :- 
	\+ has_property(H,P), 
	has_property(none,T,P).
%%  any property on list or histogram of set identifiers (SOME case)
has_property(some,[H|_],P) :-
	H \= _-_,		 
	has_property(H,P).
has_property(some,[_-H|_],P) :-
	has_property(H,P).
has_property(some,[_|T],P) :-
	has_property(some,T,P).
*/
has_property(none,Hist,Prop) :-
	Hist = [_|_],
	has_property(all,Hist,not(Prop)).
has_property(some,Hist,Prop) :-
	Hist = [_|_],
	has_property(at_least(1),Hist,Prop).
has_property(exactly(N),Hist,Prop) :-
	count_prop(Hist,Prop,0,N).
has_property(at_least(N),Hist,Prop) :-
	count_prop(Hist,Prop,0,M),
	M >= N.
has_property(at_most(N),Hist,Prop) :-
	count_prop(Hist,Prop,0,M),
	M =< N.
has_property(nth(N),Hist,Prop) :-
	nth1(N,Hist,El),
	El \= _-_,
	has_property(El,Prop).

count_prop([],_,A,A).
count_prop([C-H|T],P,Acc,N) :-
	has_property(H,P),
	Next is Acc+C,
	count_prop(T,P,Next,N).
count_prop([C-H|T],P,Acc,N) :-
	\+ has_property(H,P),
	count_prop(T,P,Acc,N).
count_prop([H|T],P,Acc,N) :-
	H \= _-_,		 
	has_property(H,P),
	Next is Acc+1,
	count_prop(T,P,Next,N).
count_prop([H|T],P,Acc,N) :-
	H \= _-_,		 
	\+ has_property(H,P),
	count_prop(T,P,Acc,N).



% auxiliary to avoid operator clash for X \= not(_)
is_a_not(not(_)).

%%%%
% [finest_]histogram(Node,SortedCountValueList)
% a histogram of set N is a cut through the tree rooted at N, consisting of count-id pairs with count>0 ordered by node IDs
%% any
histogram(N,Hist) :-
	cut(N,Cut),
	sort_histogram(Cut,Hist).
%% the maximally split histogram is the one from the leaf-cut
finest_histogram(N,Hist) :-
	leaf_cut(N,Cut),
	sort_histogram(Cut,Hist).

%%%
% cut(Node,CountValueList)
% a cut of the subtree rooted at Node
cut(N,Cut) :-
	cut(N,[],Cut).
%% at any node, we can stop
cut(N,Acc,[S-N|Acc]) :-
	node(N,_,_),
	node_size(N,S).
%% or we go a level down (if there are children)
cut(N,Acc,Res) :-
	node(N,_,Kids),
	Kids \= [],
	cut_list(Kids,Acc,Res).

cut_list([],Acc,Acc).
cut_list([H|T],Acc,Res) :-
	cut(H,Acc,Int),
	cut_list(T,Int,Res).

%%%
% leaf_cut(Node,CountValueList)
% a leaf-cut is one that always goes down as far as possible, i.e., the list of leaves 
leaf_cut(N,Cut) :-
	leaf_cut(N,[],Cut).
leaf_cut(N,Acc,[S-N|Acc]) :-
	node(N,_,[]),
	\+ tuple(N),
	node_size(N,S).
leaf_cut(N,Acc,Hist) :-
	node(N,_,[]),
	tuple(N),
	node_size(N,S),
	enumeration_histogram(S,N,H),
	append(H,Acc,Hist).
leaf_cut(N,Acc,Res) :-
	node(N,_,Kids),
	Kids \= [],
	leaf_cut_list(Kids,Acc,Res).

leaf_cut_list([],Acc,Acc).
leaf_cut_list([H|T],Acc,Res) :-
	leaf_cut(H,Acc,Int),
	leaf_cut_list(T,Int,Res).

	
%%%%
% number_of_parts(+Set,-K)
%% if static with partitioning, length of finest one
number_of_parts(Set,K) :-
	partition(Set,_),
	finest_histogram(Set,Hist),
	length(Hist,K).
%% if static & ordered without partitioning, size
number_of_parts(Set,K) :-
	static_set(Set),
	tuple(Set),
	\+ partition(Set,_),
	node_size(Set,K).
%% if static & unordered without partitioning, 1
number_of_parts(Set,1) :-
	static_set(Set),
	\+ tuple(Set),
	\+ partition(Set,_).
%% dynamic ones not (yet) supported

%%%%
% sizes_of_parts(+Set,-Cs)
%% if static with partitioning, length of finest one
sizes_of_parts(Set,Cs) :-
	partition(Set,_),
	finest_histogram(Set,Hist),
	counts_and_sum(Hist,Cs,_).
%% if static & ordered without partitioning, size many 1s
sizes_of_parts(Set,Cs) :-
	static_set(Set),
	tuple(Set),
	\+ partition(Set,_),
	node_size(Set,K),
	list_of_ones(K,Cs).
%% if static & unordered without partitioning, size
sizes_of_parts(Set,[K]) :-
	static_set(Set),
	\+ tuple(Set),
	\+ partition(Set,_),
	node_size(Set,K).
%% dynamic ones not (yet) supported

	

%%%%%%%%%%%%
filter_by_prop([],_,[]).
filter_by_prop([C-V|H],P,[C-H|F]) :-
	has_property(V,P),
	filter_by_prop(H,P,F).
filter_by_prop([_-V|H],P,F) :-
	\+ has_property(V,P),
	filter_by_prop(H,P,F).

%%%%
% number_of_parts_with_prop(+Set,+Prop,-K)
%% if static with partitioning, length of finest one after filtering
number_of_parts_with_prop(Set,Prop,K) :-
	partition(Set,_),
	finest_histogram(Set,Hist),
	filter_by_prop(Hist,Prop,HistF),
	length(HistF,K).
%% if static & ordered without partitioning, size if Prop and 0 else
number_of_parts_with_prop(Set,Prop,K) :-
	static_set(Set),
	tuple(Set),
	\+ partition(Set,_),
	has_property(Set,Prop),
	node_size(Set,K).
number_of_parts_with_prop(Set,Prop,0) :-
	static_set(Set),
	tuple(Set),
	\+ partition(Set,_),
	\+ has_property(Set,Prop).
%% if static & unordered without partitioning, 1 if Prop and 0 else
number_of_parts_with_prop(Set,Prop,1) :-
	static_set(Set),
	\+ tuple(Set),
	\+ partition(Set,_),
	has_property(Set,Prop).
number_of_parts_with_prop(Set,Prop,0) :-
	static_set(Set),
	\+ tuple(Set),
	\+ partition(Set,_),
	\+ has_property(Set,Prop).
%% dynamic ones not (yet) supported

%%%%
% sizes_of_parts_with_prop(+Set,+Prop,-Cs)
%% if static with partitioning, counts of finest one after filtering
sizes_of_parts_with_prop(Set,Prop,Cs) :-
	partition(Set,_),
	finest_histogram(Set,Hist),
	filter_by_prop(Hist,Prop,HistF),
	counts_and_sum(HistF,Cs,_).
%% if static & ordered without partitioning, size many 1s if Prop and empty else
sizes_of_parts_with_prop(Set,Prop,Cs) :-
	static_set(Set),
	tuple(Set),
	\+ partition(Set,_),
	has_property(Set,Prop),
	node_size(Set,K),
	list_of_ones(K,Cs).
sizes_of_parts_with_prop(Set,Prop,[]) :-
	static_set(Set),
	tuple(Set),
	\+ partition(Set,_),
	\+ has_property(Set,Prop).
%% if static & unordered without partitioning, size if Prop and empty else
sizes_of_parts_with_prop(Set,Prop,[K]) :-
	static_set(Set),
	\+ tuple(Set),
	\+ partition(Set,_),
	has_property(Set,Prop),
	node_size(Set,K).
sizes_of_parts_with_prop(Set,Prop,[]) :-
	static_set(Set),
	\+ tuple(Set),
	\+ partition(Set,_),
	\+ has_property(Set,Prop).
%% dynamic ones not (yet) supported

node_size_with_prop(Set,Prop,N) :- 
	sizes_of_parts_with_prop(Set,Prop,List),
	sum_list(List,N).




%%%%%%%%%%%%

list_of_ones(0,[]).
list_of_ones(N,[1|L]) :-
	N > 0,
	NN is N-1,
	list_of_ones(NN,L).


static_set(S) :-
	\+ dynamic_set(S).

dynamic_set(S) :-
	takeW(_,S).
dynamic_set(S) :-
	take(_,S).
dynamic_set(S) :-
	take(_,S,_).
dynamic_set(S) :-
	take(_,_,S).
