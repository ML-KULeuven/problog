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


% uniform constraint solving to compute static instances, using complex/2 instead of fraction/3 and sum/2
% TODO: check for potential numeric trouble

% externally used predicates defined in this file:
% static_instance/2

% relevant from input_interface:
% group/1
% size/2
% given_exactly_*
% property_definition/2
% attribute_value/2
% also needs histograms, lists, setup_aux for compute_join

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% top level predicate: turns a group into its multiset, if defined
% steps are
% 1. find all groups of correlated attributes -> Systems
% 2. for each of them, construct and solve the constraint system -> Solutions
% 3. combine the solutions into a single multiset -> MultiSet 
% (no need to compress on properties here as this will happen before drawing anyways)
static_instance(Group,MultiSet) :-
	group(Group),
	split_group_attributes_in_mccs(Group,Sys),
	sort(Sys,Systems),
	construct_and_solve_systems(Group,Systems,Solutions),
	solutions_to_multiset(Solutions,MultiSet).%,debugprint(Group,MultiSet).

%%% step 3
% single system: the multiset is its solution
solutions_to_multiset([Single],Single).
% two or more systems: they have to be independent distributions, so compute joint distribution (see setup_aux)
solutions_to_multiset([A,B|C],MultiSet) :-
	compute_joint([A,B|C],MultiSet).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% predicates for step 1: split_group_attributes_in_mccs(+G,-MCCs)
%%% for group G, MCCs is a maximal sorted list of correlated attributes on G
split_group_attributes_in_mccs(G,MCCs) :-
	group(G),
	all_classes_in_group(G,All),
	split_classes_in_mccs(All,G,MCCs).

% given G, find all classes used in its given-statements
all_classes_in_group(G,All) :-
	all(Class,class_in_group(G,Class),A),
	sort(A,All).

% given G and its attributes Atts, find all maximally connected components
split_classes_in_mccs(Atts,G,MCCs) :-
	all(MCC,corr_mcc(Atts,Group,MCC),MCCs).

%%%%%%%%% attributes correlated on group
% corr_mcc(+Atts,+Group,-MCC)
% for each attribute, find all reachable ones
% not the most efficient way to do this, as it generates groups multiple times
corr_mcc(Atts,Group,MCC) :-
	member(A,Atts),
	all(B,corr_reachable(A,B,Group),L),
	sort(L,MCC).

% attributes are correlated on group G if they appear in the same given
correlated_attributes_on_group(G,A,B) :-
	jointly_given_group_classes(Group,List),
	member(A,List),
	member(B,List).

% transitive closure of correlated_attributes_on_group
corr_reachable(A,B,G) :-
	correlated_attributes_on_group(G,A,B).
corr_reachable(A,B,G) :-
	correlated_attributes_on_group(G,A,C),
	corr_reachable(C,B,G).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% predicates for step 2
% construct_and_solve_systems(+Group,+Systems,-Solutions)
% given Group & list of lists of correlated attributes, get corresponding list of solutions (=multisets)
construct_and_solve_systems(_,[],[]).
construct_and_solve_systems(Group,[Task|Systems],[Sol|Solutions]) :-
	construct_and_solve(Group,Task,Sol),
	construct_and_solve_systems(Group,Systems,Solutions).

% construct and solve a single constraint system
% construct_and_solve(+Group,+SortedClassList,-Histogram)
construct_and_solve(Group,ClassList,MultiSet) :-
	get_eq_constraints(Group,ClassList,PartialSol0,ECons),%debugprint(eq,PartialSol0,ECons),
	get_frac_constraints(Group,ClassList,FCons),%debugprint(frac,FCons),
	append(ECons,FCons,EFCons),
	get_general_constraints(ClassList,GCons),%debugprint(g,GCons),
	append(EFCons,GCons,Constraints0),%debugprint(complex,Constraints0),
	outer_fixpoint_prop_solve(PartialSol0,Constraints0,PartialSol1,Constraints1),%debugprint(fp1,PartialSol1,Constraints1),
	second_fixpoint(PartialSol1,Constraints1,ClassList,PartialSol),%debugprint(fp2,PartialSol),
	extract_multiset(PartialSol,ClassList,MultiSet).%,debugprint(ms,MultiSet).

% if no complex constraints are left after the first fixpoint, return the current solution
second_fixpoint(PartialSol,[],_,PartialSol).
% if we have constraints and no other-variables can be set zero, also return the current solution
second_fixpoint(PartialSol1,[_|_],ClassList,PartialSol1) :-
	set_other_zero(PartialSol1,ClassList,[]).%,debugprint(skip,fp2).
% if we have constraints and some other-variables can be set to zero, we need to compute the second set of eqs and add it to the first 
second_fixpoint(PartialSol1,[C|Constraints1],ClassList,PartialSol) :-
	set_other_zero(PartialSol1,ClassList,[E|Qs]),%debugprint(PartialSol1),debugprint([E|Qs]),
	outer_fixpoint_prop_solve([E|Qs],[C|Constraints1],PartialSol3,_X),%debugprint(fp2,PartialSol3,_X),
	append(PartialSol1,PartialSol3,PartialSol).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% constraint solving

%%%%%%%%%%%%%%%%%%% outer_fixpoint_prop_solve(+PartialSol0,+Constraints0,-PartialSol1,-Constraints1)
% outer fixpoint keeps alternating between solving by propagating equalities and extracting more equalities through GE
outer_fixpoint_prop_solve(SolIn,ConsIn,SolIn,ConsIn) :-
	outer_step_prop_solve(SolIn,ConsIn,SolIn,ConsIn).%debugprint(outer,SolIn,ConsIn).
outer_fixpoint_prop_solve(SolIn,ConsIn,SolOut,ConsOut) :-
	outer_step_prop_solve(SolIn,ConsIn,SolNext,ConsNext),
	(SolIn,ConsIn) \= (SolNext,ConsNext),%debugprint(changed,SolNext,ConsNext),
	outer_fixpoint_prop_solve(SolNext,ConsNext,SolOut,ConsOut).

% if the inner fixpoint solves fully, done
outer_step_prop_solve(SolIn,ConsIn,SolOut,[]) :-
	fixpoint_prop_solve(SolIn,ConsIn,SolOut,[]).
% if the inner fixpoint leaves constraints, call GE to find more Eqs, and add those for the next round to propagate 
% note that the vars in SolNext do not appear in [Cons|Next], while those in Eqs do
outer_step_prop_solve(SolIn,ConsIn,SolOut,[Cons|Next]) :- 
	fixpoint_prop_solve(SolIn,ConsIn,SolNext,[Cons|Next]),
	eqs_from_ge([Cons|Next],Eqs),
	append(Eqs,SolNext,SolOut).  % can use append instead of merge_sol here, as we know that SolNext's variables don't appear in Eqs


%%%%%%%%%%%%%%%%%%% fixpoint_prop_solve(+PartialSol0,+Constraints0,-PartialSol1,-Constraints1)
% iterate between propagating PartSol into Constr & solving Constr for more PartialSol until no more changes
% a partial solution is a list of eq(index,size) atoms with size a number
% a constraint set is a list of complex(list,total) atoms with list a list of (coefficient,index) pairs and total a number
fixpoint_prop_solve(SolIn,ConsIn,SolIn,ConsIn) :-
	step_prop_solve(SolIn,ConsIn,SolIn,ConsIn).
fixpoint_prop_solve(SolIn,ConsIn,SolOut,ConsOut) :-
	step_prop_solve(SolIn,ConsIn,SolNext,ConsNext),
	(SolIn,ConsIn) \= (SolNext,ConsNext),
	fixpoint_prop_solve(SolNext,ConsNext,SolOut,ConsOut).

%%%%%%%%%%%%%% single solving step within the fixpoint computation
% 1. substitue numbers for all indices in constraints appearing in the partial solution
% 2. go over constraints to extract new eq/2-atoms
step_prop_solve(SolIn,ConsIn,SolOut,ConsOut) :-
	propagate_eq(SolIn,ConsIn,ConsNext),
	solve_for_more_eq(ConsNext,SolIn,SolOut,ConsOut).

%%%%%%%%% solve_for_more_eq(+ConsNext,+SolIn,SolOut,ConsOut)
% go over frac/sum constraints and see if any have a single variable that can be moved to solution
% no more constraints: done
solve_for_more_eq([],S,S,[]).
solve_for_more_eq([C|Cs],SIn,SFinal,COut) :- 
	solution(C,E),
	solve_for_more_eq(Cs,SIn,SOut,COut),
	merge_sol(E,SOut,SFinal).
solve_for_more_eq([C|Cs],SIn,SOut,[C|COut]) :-
	\+ solution(C),
	solve_for_more_eq(Cs,SIn,SOut,COut).

merge_sol([],L,L).
merge_sol([eq(Var,Val)|E],Sol,Final) :-
	insert_sol(Var,Val,Sol,Next),
	merge_sol(E,Next,Final).
insert_sol(Var,Val,[],[eq(Var,Val)]).
insert_sol(Var,Val,[eq(Var,Val2)|S],[eq(Var,Val2)|S]) :-
	abs(Val-Val2) =< 10**(-10).
insert_sol(Var,Val,[eq(Var2,Val2)|S],[eq(Var2,Val2)|SS]) :-
	Var \= Var2,
	insert_sol(Var,Val,S,SS).

solution(C) :-
	solution(C,_).

%%%%%%%%% conditions under which a constraint produces a new eq/2-atom
% single variable left
solution(complex([(K,X)],Total),[eq(X,Frac)]) :-
	Frac is Total/K.
% trivial constraint is just dropped
solution(complex([],Total),[]) :-
	Total =:= 0.
% sum is zero and all of the at least two coefficients are non-negative
solution(complex([L1,L2|List],Total),Eqs) :-
	Total =:= 0,
	check_pos_coeff_to_eq([L1,L2|List],Eqs).

check_pos_coeff_to_eq([],[]).
check_pos_coeff_to_eq([(K,X)|L],[eq(X,0)|E]) :-
	K >= 0,
	check_pos_coeff_to_eq(L,E).

%%%%%%%%% propagating eq/2-atoms into constraints
propagate_eq([],Cons,Cons).
propagate_eq([eq(Var,Size)|Es],CurrentSystem,FinalSystem) :-
	replace_var_by_size(CurrentSystem,Var,Size,NextSystem),
	propagate_eq(Es,NextSystem,FinalSystem).
			   
replace_var_by_size([],_,_,[]).
replace_var_by_size([C|Cs],Var,Size,[R|Rs]) :-
	replace_var_by_size_in_constraint(C,Var,Size,R),
	replace_var_by_size(Cs,Var,Size,Rs).

% we assume constraints are well-formed, i.e., mention each variable at most once
replace_var_by_size_in_constraint(complex(List,Total),Var,Size,complex(NewList,NewTotal)) :-
	get_var_coeff_from_list(List,Var,C,NewList),
	NewTotal is Total - C*Size.
% end reached: var not in constraint, thus C=0
get_var_coeff_from_list([],_,0,[]).
% found var: return its c
get_var_coeff_from_list([(C,Var)|List],Var,C,List).
% not this var: keep pair & continue
get_var_coeff_from_list([(K,X)|List],Var,C,[(K,X)|NList]) :-
	X \= Var,
	get_var_coeff_from_list(List,Var,C,NList).

%%%%%%%%%%%%%%%%%%% set_other_zero(+PartialSol1,+ClassList,-PartialSol2)
%%%%%%% set all open other-varsets 0
% last index position M is length of classlist
set_other_zero(SolIn,ClassList,SolOut) :-
	length(ClassList,M),
	set_other_zero(1,M,SolIn,ClassList,[],SolOut).

%%%%% iterate over index positions
% beyond last index position, return current 
set_other_zero(I,M,_,_,S,S) :-
	I > M.
% while still going over positions, process "other" for that position
set_other_zero(I,M,In,Classes,Acc,Out) :-
	I =< M,
	set_other_zero_i(I,In,Classes,Acc,Next),
	II is I+1,
	set_other_zero(II,M,In,Classes,Next,Out).

%%% set ith other vars 0, but only if it is not a class with a single given value
set_other_zero_i(I,_,Classes,Acc,Acc) :-
	nth1(I,Classes,C),
	prop_def(C,_,a(2)).
set_other_zero_i(I,In,Classes,Acc,Next) :-
	all(O,other_i(I,Classes,O),OVars),
	none_appears_pos(OVars,In),
	patch_zeros(OVars,Acc,Next).
set_other_zero_i(I,In,Classes,Acc,Acc) :-
	all(O,other_i(I,Classes,O),OVars),
	\+ none_appears_pos(OVars,In).

other_i(I,Classes,Index) :-
	nth1(I,Classes,C),
	prop_def(C,_,a(K)),
	K > 2,
	make_index(Classes,Index),
	nth1(I,Index,K).

none_appears_pos([],_).
none_appears_pos([V|Vs],Sol) :-
	\+ member(eq(V,_),Sol),
	none_appears_pos(Vs,Sol).
none_appears_pos([V|Vs],Sol) :-
	member(eq(V,S),Sol),
	S =:= 0,
	none_appears_pos(Vs,Sol).

patch_zeros([],L,L).
patch_zeros([V|Vs],Acc,Sol) :-
	patch_zeros(Vs,[eq(V,0)|Acc],Sol).


%%%%%%%%%%%%%%%%%%%% extract_multiset(+PartialSol3,-MultiSet)
extract_multiset(Solution,ClassList,MultiSet) :-
	max_solution(Solution,ClassList,Nodes),
	make_multiset(Nodes,ClassList,MultiSet).

%%% eliminate duplicates and turn into histogram
make_multiset(Nodes,ClassList,MultiSet) :-
	sort(Nodes,NoDup),
	eq2hist(NoDup,ClassList,[],MultiSet).

%%% translate a set of eq/2-atoms for a given classlist to a histogram
eq2hist([],_,H,SH) :-
	sort_histogram(H,SH).
eq2hist([eq(Index,Size)|L],ClassList,Acc,Hist) :-
	Size >= 10**(-10),
	index_to_dnf(Index,ClassList,And),
	add_to_hist(Acc,Size,And,Next),
	eq2hist(L,ClassList,Next,Hist).
eq2hist([eq(_,Size)|L],ClassList,Acc,Hist) :-
	Size < 10**(-10),
	Size >= -10**(-10),
	eq2hist(L,ClassList,Acc,Hist).
eq2hist([eq(I,Size)|L],ClassList,Acc,Hist) :-
	Size < -10**(-10),
	error('negative set size ',I,' ',Size).

% the index can be a list of indices, which form an or
index_to_dnf([Index],ClassList,And) :-
	zipped_index(Index,ClassList,Pairs),
	indexpairs2and(Pairs,And).
index_to_dnf([I1,I2|Is],ClassList,or(A1,AR)) :-
	zipped_index(I1,ClassList,Pairs),
	indexpairs2and(Pairs,A1),
	index_to_dnf([I2|Is],ClassList,AR).
		
zipped_index([],[],[]).
zipped_index([I|Is],[C|Cs],[(C,I)|Zs]) :-
	I \= a(_),
	zipped_index(Is,Cs,Zs).
zipped_index([a(_)|Is],[C|Cs],Zs) :-
	zipped_index(Is,Cs,Zs).

indexpairs2and([(C,I)],Val) :-
	prop_def(C,List,_),
	nth1(I,List,Val).
indexpairs2and([(C,I),A|B],and(Val,Rest)) :-
	prop_def(C,List,_),
	nth1(I,List,Val),
	indexpairs2and([A|B],Rest).

%%%%%%%%%%%%%%%%% max_solution(+EqAtoms,+ClassList,-MaxSolEqAtoms)
%%% given all eq/2-atoms obtained from constraint solving and the classlist, extract eq/2-atoms corresponding to maximal solution
% 1. find the leaves to be covered
% 2. sort the known equalities from specific to general
% 3. greedily add variables from the equalities to cover leaves
% 4. if necessary, patch with difference variables to get a full cover
max_solution(Sizes,ClassList,Eqs) :-
	all(L,leaf(ClassList,L),AllLeaves),
	specific_to_general(Sizes,Sorted),%debugprint(Sorted),
	greedy_cover(Sorted,AllLeaves,Uncovered,Rest,MultiSet),%debugprint(MultiSet),
	complete_cover(Rest,Uncovered,MultiSet,Eqs).%,debugprint(Eqs).

specific_to_general(Eqs,Sorted) :-
	tag_by_count(Eqs,Tagged),
	sort(Tagged,ST),
	untag(ST,Sorted).

tag_by_count([],[]).
tag_by_count([eq(I,N)|E],[(C,eq(I,N))|T]) :-
	count_wildcards(I,0,C),
	tag_by_count(E,T).

count_wildcards([],N,N).
count_wildcards([a(_)|L],I,O) :-
	N is I+1,
	count_wildcards(L,N,O).
count_wildcards([A|L],I,O) :-
	A \= a(_),
	count_wildcards(L,I,O).
	
untag([],[]).
untag([(_,E)|Es],[E|Os]) :-
	untag(Es,Os).


% no more Eqs
greedy_cover([],L,L,[],[]).
% no more Uncovered
greedy_cover(_,[],[],[],[]).
% if variable covers new leaves only, add to multiset
greedy_cover([eq(I,N)|Eqs],[U|Now],UFinal,Rest,[eq([I],N)|MultiSet]) :-
	covered_leaves(I,S),
	subset(S,[U|Now],UNext),
	greedy_cover(Eqs,UNext,UFinal,Rest,MultiSet).
% if variable covers no new leaves, forget it
greedy_cover([eq(I,N)|Eqs],[U|Now],UFinal,Rest,MultiSet) :-
	covered_leaves(I,S),
	disjoint(S,[U|Now]),
	greedy_cover(Eqs,[U|Now],UFinal,Rest,MultiSet).
% if variable covers old and new leaves, add to rest
greedy_cover([eq(I,N)|Eqs],[U|Now],UFinal,[eq(I,N)|Rest],MultiSet) :-
	covered_leaves(I,S),
	\+ subset(S,[U|Now]),
	\+ disjoint(S,[U|Now]),
	greedy_cover(Eqs,[U|Now],UFinal,Rest,MultiSet).

covered_leaves(I,S) :-
	all(L,covered_leaf(I,L),S).

% complete_cover(+Rest,+Uncovered,+MultiSet,-Eqs)
% everything covered -> done
complete_cover(_,[],MS,MS).
% no more information -> done
complete_cover([],_,MS,MS).
% else, need to complete ([eq(I,N)|Rest] is sorted because we sorted the initial equalities and greedy_cover keeps them ordered)
% if it does not cover uncovered, drop
complete_cover([eq(I,N)|Rest],[U|Ncovered],Multiset,Eqs) :-
	covered_leaves(I,S),
	split_wrt(S,[U|Ncovered],[],_),
	complete_cover(Rest,[U|Ncovered],Multiset,Eqs).
% if it does cover uncovered, construct an entry
complete_cover([eq(I,N)|Rest],[U|Ncovered],Multiset,Eqs) :-
	covered_leaves(I,S),
	split_wrt(S,[U|Ncovered],[SU|SUs],SM),
	sum_ms(Multiset,SM,M),
	K is N-M,
	subset([U|Ncovered],[SU|SUs],UNext),
	complete_cover(Rest,UNext,[eq([SU|SUs],K)|Multiset],Eqs).

split_wrt([],_,[],[]).
split_wrt([A|As],Ref,[A|In],Out) :-
	member(A,Ref),
	split_wrt(As,Ref,In,Out).
split_wrt([A|As],Ref,In,[A|Out]) :-
	\+ member(A,Ref),
	split_wrt(As,Ref,In,Out).

% sum_ms(+MultiSet,+Vars,-Sum)
% we iterate over the multiset because we might have entries covering several variables
sum_ms([],_,0).
sum_ms(_,[],0).
sum_ms([eq(Set,N)|E],[V|Vars],Sum) :-
	subset(Set,[V|Vars],Next),
	sum_ms(E,Next,Rest),
	Sum is Rest+N.
sum_ms([eq(Set,_)|E],[V|Vars],Sum) :-
	\+ subset(Set,[V|Vars]),
	sum_ms(E,[V|Vars],Sum).
	

disjoint([],_).
disjoint([A|B],C) :-
	\+ member(A,C),
	disjoint(B,C).

% given a subset and a total set, compute the complement (and fail if it isn't a subset)
subset(Sub,Total,Rest) :-
	sort(Sub,A),
	sort(Total,B),
	split_sets(A,B,Rest).
split_sets([],B,B).
split_sets([A|As],[A|Bs],Rest) :-
	split_sets(As,Bs,Rest).
split_sets([A|As],[B|Bs],[B|Rest]) :-
	A \= B,
	split_sets([A|As],Bs,Rest).

subset(A,B) :-
	subset(A,B,_).

leaf(ClassList,L) :-
	make_index(ClassList,L),
	\+ member(a(_),L).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% collecting constraints
% get_general_constraints(+ClassList,-Constraints)
% Constraints is a list of sum/2-atoms using indices only
get_general_constraints(ClassList,GCons) :-
	length(ClassList,M),
	top_index(ClassList,Top),
	get_general_constraints_top(M,Top,GCons).

% initially called with Pos=length(TopIndex), i.e., iterates over index backwards
get_general_constraints_top(0,_,[]).	
get_general_constraints_top(M,Top,GCons) :-
	M > 0,
	all(G,general_constraint(M,Top,G),This),
	MM is M-1,
	get_general_constraints_top(MM,Top,Others),
	append(This,Others,GCons).

% get_eq_constraints(+Group,+ClassList,-EqConstraints,-ComplexC)
% EqConstraints is a list of eq/2-atoms ([] in case there are only given-rels)
% ComplexC is a list of complex/2-atoms (only non-empty if some given relates an integer to a formula with or/not)
get_eq_constraints(Group,ClassList,Eqs,Compl) :-
	all(E,model_specific_constraint_eq(Group,ClassList,E),PartialSol0),
	unfold_eqs(PartialSol0,Eqs,Compl).
get_eq_constraints(Group,ClassList,[],[]) :-
	\+ model_specific_constraint_eq(Group,ClassList).

unfold_eqs([],[],[]).
% if the eq is over a list of at least two lists, we need to unfold
unfold_eqs([eq([[H|T],Snd|Rest],Size)|In],Eqs,[complex(C,Size)|Cs]) :-
	complex_from_list([[H|T],Snd|Rest],C),
	unfold_eqs(In,Eqs,Cs).
% if it is a list of a single list, strip one of
unfold_eqs([eq([[H|T]],Size)|In],[eq([H|T],Size)|Eqs],Cs) :-
	unfold_eqs(In,Eqs,Cs).
% else, it's really an eq
unfold_eqs([I|In],[I|Eqs],Cs) :- 
	I \= eq([[_|_]|_],_),
	unfold_eqs(In,Eqs,Cs).

complex_from_list([],[]).
complex_from_list([H|T],[(1,H)|OT]) :-
	complex_from_list(T,OT).
	
model_specific_constraint_eq(Group,ClassList) :-
	model_specific_constraint_eq(Group,ClassList,_).

% get_frac_constraints(+Group,+ClassList,-Constraints)
% Constraints is a list of complex/2-atoms ([] in case there are only given-ints)
get_frac_constraints(Group,ClassList,FFCons) :-
	all(F,model_specific_constraint_frac(Group,ClassList,F),FCons),
	flatten_constraints(FCons,FFCons).
get_frac_constraints(Group,ClassList,[]) :-
	\+ model_specific_constraint_frac(Group,ClassList).

flatten_constraints([],[]).
flatten_constraints([complex([(1,LA),(Frac,LF)],0)|In],[complex(List,0)|Out]) :-
	flatten_two_sums(LA,LF,Frac,List),
	flatten_constraints(In,Out).

% if both are single variables, the constraint is flat
flatten_two_sums(L1,LFrac,Frac,[(1,L1),(Frac,LFrac)]) :-
	sum_is_var(L1),
	sum_is_var(LFrac).
% if one is a var and the other a list, map on two sorted lists case
flatten_two_sums(L1,LFrac,Frac,Sum) :-
	sum_is_var(L1),
	\+ sum_is_var(LFrac),
	sort(LFrac,SFrac),
	flatten_two_sumlists([L1],SFrac,Frac,Sum).
flatten_two_sums(L1,LFrac,Frac,Sum) :-
	\+ sum_is_var(L1),
	sum_is_var(LFrac),
	sort(L1,S1),
	flatten_two_sumlists(S1,[LFrac],Frac,Sum).
% if both are lists, map on two sorted lists case
flatten_two_sums(L1,LFrac,Frac,Sum) :-
	\+ sum_is_var(L1),
	\+ sum_is_var(LFrac),
	sort(L1,S1),
	sort(LFrac,SFrac),
	flatten_two_sumlists(S1,SFrac,Frac,Sum).

% in go two sorted lists of lists, weighted 1 and Frac, respectively, out comes a list of weight-list pairs
% if the first is empty, weight the second
flatten_two_sumlists([],LFrac,Frac,Sum) :-
	weight_list_by(LFrac,Frac,Sum).
% if the second is empty, but the first not, weight the first
flatten_two_sumlists([H|T],[],_,Sum) :-
	weight_list_by([H|T],1,Sum).
% if the heads of both are equal, sum the weights
flatten_two_sumlists([H|T1],[H|TW],F,[(W,H)|Sum]) :-
	W is 1+F,
	flatten_two_sumlists(T1,TW,F,Sum).
% if first head is smaller, move it to sum
flatten_two_sumlists([H1|T1],[HW|TW],F,[(1,H1)|Sum]) :-
	H1 @< HW,
	flatten_two_sumlists(T1,[HW|TW],F,Sum).
% if second head is smaller, move it to sum
flatten_two_sumlists([H1|T1],[HW|TW],F,[(F,HW)|Sum]) :-
	HW @< H1,
	flatten_two_sumlists([H1|T1],TW,F,Sum).

weight_list_by([],_,[]).
weight_list_by([H|T],W,[(W,H)|WT]) :-
	weight_list_by(T,W,WT).


sum_is_var(S) :-
	S \= [[_|_]|_].

model_specific_constraint_frac(Group,ClassList) :-
	model_specific_constraint_frac(Group,ClassList,_).

%%% general constraints
% given position i and top-index, generate corresponding general constraints (by backtracking)
% 1. split the top-index into the part to the left of position i, the element at position i, and the part to the right of it
% 2. choose values for the left and right part
% 3. put the i-th class's any-index between the two to get the left hand side of the constraint
% 4. make the list of indices obtained by putting every other value of the i-th class in between once to get the right hand side of the constraint
% 5. add 1 as coefficient of every rhs variable, and -1 for the lhs
general_constraint(I,Top,complex([(-1,A)|CList],0)) :-
	split_top_on_i(1,I,Top,Left,a(K),Right),
	ground_index(Left,GLeft),
	ground_index(Right,GRight),
	append(GLeft,[a(K)|GRight],A),
	make_sum_rhs(1,K,GLeft,GRight,List),
	add_ones(List,CList).

add_ones([],[]).
add_ones([A|B],[(1,A)|C]) :-
	add_ones(B,C).

split_top_on_i(I,I,[H|Right],[],H,Right).
split_top_on_i(N,I,[H|Top],[H|Left],M,Right) :-
	N < I,
	NN is N+1,
	split_top_on_i(NN,I,Top,Left,M,Right).

make_sum_rhs(K,K,GLeft,GRight,[This]) :-
	append(GLeft,[K|GRight],This).
make_sum_rhs(I,K,GLeft,GRight,[This|List]) :-
	I < K,
	append(GLeft,[I|GRight],This),
	II is I+1,
	make_sum_rhs(II,K,GLeft,GRight,List).

%%% eq-constraints
% model_specific_constraint_eq(+Group,+AttList,-Constraint)
% if input gives integer size, get constraint indexvar = size
model_specific_constraint_eq(Group,AttList,eq(Index,Size)) :-
	given_size_index_integer(Group,AttList,Size,Index).

% integer sized parts
given_size_index_integer(Group,AttList,Size,Index) :-
	given_exactly_integer(Size,Group,Prop),
	make_index_for_prop(AttList,Prop,Index).
% explicit group size
given_size_index_integer(Group,AttList,Size,Index) :-
	size(Group,Size),
	top_index(AttList,Index).
% implicit group size 1
given_size_index_integer(Group,AttList,1,Index) :-
	no_size_given(Group),
	no_pos_integer_parts_given(Group),
	top_index(AttList,Index).

%%% fraction-constraints
% model_specific_constraint_frac(+Group,+AttList,-Constraint)
% if input gives fractional size, get constraint indexvar = frac * topvar
model_specific_constraint_frac(Group,AttList,complex([(1,Index),(F,Top)],0)) :-
	given_size_index_rel(Group,AttList,Frac,Top,Index), F is -Frac.

% rel/2: parent is top
given_size_index_rel(Group,AttList,Frac,Top,Index) :-
	given_exactly_rel(Frac,Group,Prop),
	make_index_for_prop(AttList,Prop,Index),
	top_index(AttList,Top).
% rel/3: parent is condition
given_size_index_rel(Group,AttList,Frac,CIndex,Index):-
	given_exactly_rel_cond(Frac,Group,Cond,Prop),
	make_index_for_prop(AttList,Cond,CIndex),
	make_index_for_prop(AttList,Prop,Index).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% constructing indices

% make_index(+ClassList,-Index)
% given class list, make index: for every class, choose a number between 1 and the maximum, or the any-index
% TODO: replace by top_index,ground_index ?
make_index([],[]).
make_index([C|Classes],[I|Index]) :-
	prop_indexval(C,I),
	make_index(Classes,Index).

% top_index(+ClassList,-Index)
% given list of classes, make list of their any-indices
top_index([],[]).
top_index([A|As],[I|Is]) :-
	prop_def(A,_,I),
	top_index(As,Is).

% leaf_index(+ClassList,-Index,-Values)
% given list of classes, make an index without any-indices and its list of values
leaf_index([],[],[]).
leaf_index([A|As],[I|Is],[V|Vs]) :-
	prop_def(A,Values,_),
	is_nth(Values,V,I),
	leaf_index(As,Is,Vs).

% ground_index(+TopIndex,-Index)
% given top-index, enumerate all indices it covers
ground_index([],[]).
ground_index([a(K)|Index],[J|Rest]) :-
	between(1,K,J),
	ground_index(Index,Rest).
ground_index([a(K)|Index],[a(K)|Rest]) :-
	ground_index(Index,Rest).

% covered_leaf(+Index,-Leaf)
covered_leaf([],[]).
covered_leaf([a(K)|I],[J|L]) :-
	between(1,K,J),
	covered_leaf(I,L).
covered_leaf([K|I],[K|L]) :-
	K \= a(_),
	covered_leaf(I,L).

% prop_indexval(+Class,-IndexValue)
% given Class, get index value
prop_indexval(Class,Value) :-
	prop_def(Class,_,a(M)),
	between(1,M,Value).
prop_indexval(Class,Value) :-
	prop_def(Class,_,Value).

% make_index_for_prop(+ClassList,+Property,-Index)
% given list of classes and a property,
% if it is a positive and, make the corresponding index list
make_index_for_prop(AttList,Prop,Index) :-
	prop_to_att_vals(Prop,List),
	atts_in_list_are_subset(List,AttList),
	make_index_with_list(AttList,List,Index).
% else, make the list of leaf indices
make_index_for_prop(AttList,Prop,Index) :-
	\+ prop_to_att_vals(Prop),
	all(I,sat_leaf(Prop,AttList,I),Index).

sat_leaf(Prop,AttList,Index) :-
	leaf_index(AttList,Index,Values),
	prop_sat_in_leaf(Prop,Values).
	
prop_to_att_vals(Prop) :-
	prop_to_att_vals(Prop,_).

make_index_with_list([],_,[]).
make_index_with_list([A|AttList],List,[I|Index]) :-
	member((A,V),List),
	prop_def(A,Values,_),
	is_nth(Values,V,I),
	make_index_with_list(AttList,List,Index).
make_index_with_list([A|AttList],List,[I|Index]) :-
	\+ member((A,V),List),
	prop_def(A,_,I),
	make_index_with_list(AttList,List,Index).

% given list of (Class,Value) pairs and list of classes, test whether all classes in the former are in the latter
atts_in_list_are_subset([],_).
atts_in_list_are_subset([(A,_)|L],All) :-
	member(A,All),
	atts_in_list_are_subset(L,All).

% prop_sat_in_leaf(+Prop,+LeafValues)
% does the property hold in a leaf given as list of its values?
prop_sat_in_leaf(and(P1,P2),Leaf):-
	prop_sat_in_leaf(P1,Leaf),
	prop_sat_in_leaf(P2,Leaf).
prop_sat_in_leaf(or(P,_),Leaf) :-
	prop_sat_in_leaf(P,Leaf).
prop_sat_in_leaf(or(_,P),Leaf) :-
	prop_sat_in_leaf(P,Leaf).
prop_sat_in_leaf(not(P),Leaf) :-
	\+ prop_sat_in_leaf(P,Leaf).
prop_sat_in_leaf(P,Leaf) :-
	P \= and(_,_),
	P \= or(_,_),
	P \= not(_),
	member(P,Leaf).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% interface predicates to input interface

% extend the property definition interface to include the implicit value and the any-index
prop_def(Class,ExtVal,a(T)) :-
	property_definition(Class,Values),
	length(Values,N),
	T is N+1,
	append(Values,[none_of(Values)],ExtVal).

% check for explicit positive integer sizes
no_pos_integer_parts_given(G) :-
	\+ integer_parts_given(G).

integer_parts_given(G) :-
	given_exactly_integer(N,G,_),
	N > 0.

%%%%%%%%% step 1
% -Class appears in a given statement for +Group
class_in_group(Group,Class) :-
	jointly_given_group_classes(Group,List),
	member(Class,List).
	
% translate the property in a given statement to the sorted list of property classes it uses 
jointly_given_group_classes(Group,ClassProp) :-
	given_group_property(Group,Prop),
	extract_classes(Prop,ClassProp).

%%%%%%%%% across steps
% for now, we expect that all given statements use conjunctions of (positive) values only
% TODO: needs to be generalized later (at which point we also need to check where the fact that variable names = indices = lists is exploited)
prop_to_att_vals(P,[(Class,P)]) :-
	attribute_value(Class,P).
prop_to_att_vals(and(P1,P2),L) :-
	prop_to_att_vals(P1,L1),
	prop_to_att_vals(P2,L2),
	append(L1,L2,L).

% given +Prop (= arbitrary boolean formula over class values), get sorted -List of classes with values appearing in it
extract_classes(Prop,List) :-
	extract_classes(Prop,[],All),
	sort(All,List).

% un-nest Boolean properties
extract_classes(not(P),Acc,All) :-
	extract_classes(P,Acc,All).
extract_classes(or(P1,P2),Acc,All) :-
	extract_classes(P1,Acc,Left),
	extract_classes(P2,Left,All).
extract_classes(and(P1,P2),Acc,All) :-
	extract_classes(P1,Acc,Left),
	extract_classes(P2,Left,All).
% and translate values to their class
extract_classes(P,Acc,[Class|Acc]) :-
	P \= not(_),
	P \= or(_,_),
	P \= and(_,_),
	attribute_value(Class,P).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% other auxiliaries
% is_nth(+List,+Element,-Position)
% needed because nth1 in library(lists)) doesn't work in this mode
is_nth(Values,V,N) :-
	is_nth(1,Values,V,N).
is_nth(N,[V|_],V,N).
is_nth(I,[V|Vs],S,N) :-
	J is I+1,
	is_nth(J,Vs,S,N).
