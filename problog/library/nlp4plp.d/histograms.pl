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


% a histogram is a list of entries [c1-v1,...,cK-vK] with cI positive integers and v(i) @< v(i+1) for all i=1,...,K-1
%
% externally used predicates defined in this file:
% merge_hist/3
% delete_from_hist/3
% add_to_hist/4
% histogram_size/2
% sort_histogram/2
% list2hist/2
% 

% enumeration_histogram(+N,+Name,-Hist)
% Hist contains 1-Name(x) for x=1,..,N
enumeration_histogram(N,Name,H) :-
	gen_enum_hist(N,Name,[],H).
gen_enum_hist(0,_,H,H).
gen_enum_hist(N,Name,Acc,Hist) :-
	N > 0,
	Value =.. [Name,N],
	add_to_hist(Acc,1,Value,Next),
	NN is N-1,
	gen_enum_hist(NN,Name,Next,Hist).
	
	

% add_to_hist(+OldHist,+Count,+Element,-NewHist)
% NewHist is OldHist with Count copies of Element added
add_to_hist(H,0,_,H).
add_to_hist([],C,E,[C-E]) :- C>0.
add_to_hist([K-E|L],C,E,[N-E|L]) :-
	C > 0,
	N is K+C.
add_to_hist([K-D|L],C,E,[K-D|L2]) :-
	C > 0,
	D @< E,
	add_to_hist(L,C,E,L2).
add_to_hist([K-D|L],C,E,[C-E,K-D|L]) :-
	C > 0,
	D @> E.

% delete_from_hist(+OldHist,+Element,-NewHist)
% NewHist is OldHist with one copy of Element deleted
delete_from_hist([X-E|L],E,L) :-
	X >= 1-10**(-10), X =< 1+10**(-10).
delete_from_hist([K-E|L],E,[KK-E|L]) :-
	K > 1+10**(-10),
	KK is K-1.
delete_from_hist([K-D|L],E,[K-D|L2]) :-
	D @< E,
	delete_from_hist(L,E,L2).
delete_from_hist([K-_|_],E,_) :-
	K < 1-10**(-10),
	error('cannot delete fractional element from histogram ',K,E).

% list2hist(+List,-Hist)
% Hist is histogram of sequence List
list2hist(L,H) :-
	list2hist(L,[],H).
list2hist([],H,H).
list2hist([A|B],H,H2) :-
	add_to_hist(H,1,A,H1),
	list2hist(B,H1,H2).

% filter_zeros(+OldHist,-NewHist)
% NewHist is OldHist with count 0 elements removed
filter_zeros([],[]).
filter_zeros([Z-E|T],F) :-
	Z =:= 0,
	filter_zeros(T,F).
filter_zeros([C-E|T],[C-E|F]) :-
	C > 0,
	filter_zeros(T,F).



sort_histogram(Cut,Hist) :- 
	filter_zeros(Cut,Int),
	sort_histogram(Int,[],Hist).
sort_histogram([],Acc,Acc).
sort_histogram([C-V|T],Acc,Hist) :-
	add_to_hist(Acc,C,V,Int),
	sort_histogram(T,Int,Hist).
	

% all_unique(+Hist,-List)
% if all counts are 1, List is the list of values in Hist
% else, fails
all_unique([],[]).
all_unique([1-E|T],[E|R]) :-
	all_unique(T,R).

% counts_and_sum(+Hist,-Counts,-Sum)
counts_and_sum(H,C,S) :- 
	counts_and_sum(H,C,0,S).
counts_and_sum([],[],S,S).
counts_and_sum([C1-_|T],[C1|Cs],Acc,Res) :- 
	Next is Acc+C1,
	counts_and_sum(T,Cs,Next,Res).


% choose_from_hist(+Hist,-Element)
% Element is a value in Hist
choose_from_hist([_-V|_],V).
choose_from_hist([_|T],V) :-
	choose_from_hist(T,V).

% histogram_size(+Hist,-Size)
histogram_size([],0).
histogram_size([C-_|H],S) :-
	histogram_size(H,N),
	S is N+C.

is_histogram([]).
is_histogram([_-_|_]).

merge_hist([],H,H).
merge_hist([C-E|R],H2,Merged) :-
	add_to_hist(H2,C,E,NextH),
	merge_hist(R,NextH,Merged).
