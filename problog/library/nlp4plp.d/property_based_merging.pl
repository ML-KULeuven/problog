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
% property_compress/3
% merge_histogram_on_properties/3
%

% property_compress(+SetID,+FullHistogram,-CompressedHistogram)
property_compress(SetID,FullHist,CompressedHist) :-
	set_propertylist(SetID,Properties),
	merge_histogram_on_properties(Properties,FullHist,CompressedHist).

% merge_histogram_on_properties(+Properties,+FullHist,-CompressedHist)
% this seems needed to get the choose_group action working
merge_histogram_on_properties([],H,H).
% single prop is easier, as we don't need to form cross product, just split on property
merge_histogram_on_properties([Prop],FullHist,CompressedHist) :- 
	divide_hist_on_prop(FullHist,Prop,Pos,Neg),
	histogram_size(Pos,PP),
	histogram_size(Neg,NN),
	Pos = [_-PV|_],
	Neg = [_-NV|_],
	add_to_hist([],PP,PV,Int),
	add_to_hist(Int,NN,NV,CompressedHist).
merge_histogram_on_properties([Prop],FullHist,CompressedHist) :- 
	divide_hist_on_prop(FullHist,Prop,Pos,[]),
	histogram_size(Pos,PP),
	Pos = [_-PV|_],
	add_to_hist([],PP,PV,CompressedHist).
merge_histogram_on_properties([Prop],FullHist,CompressedHist) :- 
	divide_hist_on_prop(FullHist,Prop,[],Neg),
	histogram_size(Neg,NN),
	Neg = [_-NV|_],
	add_to_hist([],NN,NV,CompressedHist).
% if we have at least two properties, we annotate the histogram entries with a code on which properties it satisfies, and merge based on that
merge_histogram_on_properties([A,B|Prop],FullHist,CompressedHist) :- 
	annotate_with_truthlist(FullHist,[A,B|Prop],Annotated),
	merge_on_truthlist(Annotated,CompressedHist).

% divide_hist_on_prop(+Hist,+Property,-HistSatisfiesProp,-HistDoesNotSatisfyProp)
divide_hist_on_prop([],_,[],[]).
divide_hist_on_prop([C-V|H],Prop,[C-V|Pos],Neg) :-
	has_property(V,Prop),
	divide_hist_on_prop(H,Prop,Pos,Neg).
divide_hist_on_prop([C-V|H],Prop,Pos,[C-V|Neg]) :-
	\+ has_property(V,Prop),
	divide_hist_on_prop(H,Prop,Pos,Neg).

% annotate_with_truthlist(+Histogram,+PropertyList,-AnnotatedHist)
annotate_with_truthlist([],_,[]).
annotate_with_truthlist([C-V|H],Props,[t(TL,C,V)|L]) :-
	get_truthlist(Props,V,TL),
	annotate_with_truthlist(H,Props,L).

% get_truthlist(+Properties,+HistElement,-TruthVector)
get_truthlist([],_,[]).
get_truthlist([P|Ps],V,[1|Ts]) :-
	has_property(V,P),
	get_truthlist(Ps,V,Ts).
get_truthlist([P|Ps],V,[0|Ts]) :-
	\+ has_property(V,P),
	get_truthlist(Ps,V,Ts).

% merge_on_truthlist(+AnnotatedHist,-CompressedHist)
% for each truth list in the annotated histogram, get the total number of elements annotated with it
% and add one group to the compressed histogram, using the value of the first occurence for the whole group
merge_on_truthlist(Annotated,CompressedHist) :-
	merge_tl(Annotated,[],CompressedHist).
merge_tl([],Acc,Acc).
merge_tl([t(TL,C,V)|Ts],Acc,Final) :-
	divide_tvs(TL,Ts,C,Num,Rest),
	add_to_hist(Acc,Num,V,Next),
	merge_tl(Rest,Next,Final).

% divide_tvs(+TruthList,+AnnotatedHist,+CurrentCount,-FinalCount,-RemainingAnnotatedHist)
% given a specific truthlist, count how many elements in the annotated histogram have this truthlist, and remove them 
divide_tvs(_,[],C,C,[]).
divide_tvs(TL,[t(TL,X,_)|Ts],C,Num,Rest) :-
	Next is C+X,
	divide_tvs(TL,Ts,Next,Num,Rest).
divide_tvs(TL,[t(TL1,X,V)|Ts],C,Num,[t(TL1,X,V)|Rest]) :-
	TL \= TL1,
	divide_tvs(TL,Ts,C,Num,Rest).


