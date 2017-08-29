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


% additional auxiliaries for static_setup
%
% externally used predicates defined in this file:
% add_class_to_hist/3
% compute_joint/2
% subset_of_pairs/4
%

add_class_to_hist([],_,[]).
add_class_to_hist([C-V|H],Class,[C-[Class-V]|R]) :-
	add_class_to_hist(H,Class,R).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute_joint(+Hist,-Dist)
% given list Hist of independent distributions, compute their joint Dist 
compute_joint([H],N) :-
	normalized_hist_dist(H,N).%,debugprint(base,N).
compute_joint([H|Hs],Instance) :-
	compute_joint(Hs,One),%debugprint(calling,cross_joint(H,One,Instance)),
	cross_joint(H,One,Instance).

cross_joint(Outer,Inner,Instance) :-
	normalized_hist_dist(Outer,Norm),
	cjoint(Norm,Inner,Instance).

cjoint([],_,[]).
cjoint([P-V|H],Inner,Instance) :-
	mult_each(P-V,Inner,This),
	cjoint(H,Inner,That),
	append(This,That,Instance).

mult_each(_,[],[]).
mult_each(P-V,[PP-VV|R],[A-and(V,VV)|T]) :-
	A is P*PP,
	mult_each(P-V,R,T).

normalized_hist_dist(H,H) :-
	histogram_size(H,S),
	abs(S-1) < 10**(-10).
normalized_hist_dist(H,N) :-
	histogram_size(H,S),
	S > 1,
	div_each(S,H,N).

div_each(_,[],[]).
div_each(N,[C-V|R],[A-V|T]) :-
	A is C/N,
	div_each(N,R,T).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subset_of_pairs([],_,_,[]).
subset_of_pairs([N-and(V1,V2)|L],L1,L2,[N-and(V1,V2)|R]) :-
	member(V1,L1),
	member(V2,L2),
	subset_of_pairs(L,L1,L2,R).
subset_of_pairs([N-and(V2,V1)|L],L1,L2,[N-and(V1,V2)|R]) :-
	member(V1,L1),
	member(V2,L2),
	subset_of_pairs(L,L1,L2,R).
