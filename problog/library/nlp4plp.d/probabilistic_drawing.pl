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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% basic recursion structures to enumerate all answers with M elements drawn probabilistically from histogram H,
% potentially checking property constraints along the way
%
% externally used predicates defined in this file:
% draw_constraints_in_out/13
%

%%%%%%%%%%%%%%%%% new top always enters with three lists of constraints %%%%%%%%%%%%%%%%% 
% draw_constraints_in_out(WType,SType,M,SuperSetC,Instance,InstanceRest,TrialID,ScopeIn,AllIn,AggIn,ScopeOut,AllOut,AggOut)

% if there are scope constraints, use them
draw_constraints_in_out(WType,SType,M,SuperSetC,Instance,InstanceRest,TrialID,[scope(Quant,Prop)|ScopeOut],AllIn,AggIn,ScopeOut,AllIn,AggIn) :-
	draw_constrained_list(WType,SType,M,SuperSetC,Instance,InstanceRest,TrialID,[scope(Quant,Prop)|ScopeOut]).
% if there are no scopes but an all, use that (draw_all still has rest/no rest entries)
draw_constraints_in_out(wor,SType,M,SuperSetC,Instance,InstanceRest,TrialID,[],[all(SorD,Class)|AllOut],AggIn,[],AllOut,AggIn) :-
	draw_all(SorD,Class,wor,SType,M,SuperSetC,Instance,InstanceRest,TrialID).
draw_constraints_in_out(wr,SType,M,SuperSetC,Instance,InstanceRest,TrialID,[],[all(SorD,Class)|AllOut],AggIn,[],AllOut,AggIn) :-
	draw_all(SorD,Class,wr,SType,M,SuperSetC,Instance,TrialID).
% neither scope nor all: just draw for now (via empty constraint list)
draw_constraints_in_out(WType,SType,M,SuperSetC,Instance,InstanceRest,TrialID,[],[],AggIn,[],[],AggIn) :-
	draw_constrained_list(WType,SType,M,SuperSetC,Instance,InstanceRest,TrialID,[]).

% branch on SType and init appropriate extra information
draw_constrained_list(WType,set,M,SuperSetC,Instance,InstanceRest,TrialID,ScopeC) :-
	draw_clist_set(M,WType,SuperSetC,[],Instance,InstanceRest,TrialID,ScopeC).
draw_constrained_list(WType,seq,M,SuperSetC,Instance,InstanceRest,TrialID,ScopeC) :-
	draw_clist_seq(M,WType,SuperSetC,Instance,InstanceRest,TrialID,ScopeC,1).

% recursion for set
draw_clist_set(0,_,Hist,Inst,Inst,Hist,_,ScopeC) :-
	all_scope_sat(ScopeC).
draw_clist_set(M,WType,Hist,Acc,Instance,InstanceRest,TrialID,ScopeC) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	update_scope_constraints(ScopeC,E,xx,NewScope),  % xx is number in case of seq
	next_histogram(WType,E,Hist,NewHist),
	add_to_hist(Acc,1,E,NewAcc),
	MM is M-1,
	draw_clist_set(MM,WType,NewHist,NewAcc,Instance,InstanceRest,TrialID,NewScope).

% recursion for sequence
draw_clist_seq(0,_,Hist,[],Hist,_,ScopeC,_) :-
	all_scope_sat(ScopeC).
draw_clist_seq(M,WType,Hist,[E|Instance],InstanceRest,TrialID,ScopeC,This) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	update_scope_constraints(ScopeC,E,This,NewScope),
	next_histogram(WType,E,Hist,NewHist),
	MM is M-1,
	Next is This+1,
	draw_clist_seq(MM,WType,NewHist,Instance,InstanceRest,TrialID,NewScope,Next).

% check whether remaining list of constraints are satisfied, i.e.,
% only constraints left if any are all, exactly(0) or at_most(_)
all_scope_sat([]).
all_scope_sat([scope(all,_)|Rest]) :-
	all_scope_sat(Rest).
all_scope_sat([scope(exactly(0),_)|Rest]) :-
	all_scope_sat(Rest).
all_scope_sat([scope(at_most(_),_)|Rest]) :-
	all_scope_sat(Rest).

% update constraint list based on the element just sampled (and its number in the draw)
% update_scope_constraints(+OldScopes,+Elem,+Num,-NewScopes)
% no more constraints to update
update_scope_constraints([],_,_,[]).
% nth on this one: drop if satisfied (and fail else)
update_scope_constraints([scope(nth(N),Prop)|Rest],E,N,NewScope) :-
	has_property(E,Prop),
	update_scope_constraints(Rest,E,N,NewScope).
% nth on other: keep
update_scope_constraints([scope(nth(M),Prop)|Rest],E,N,[scope(nth(M),Prop)|NewScope]) :-
	M \= N,
	update_scope_constraints(Rest,E,N,NewScope).
% at_least(1) and sats: drop
update_scope_constraints([scope(at_least(1),Prop)|Rest],E,N,NewScope) :-
	has_property(E,Prop),
	update_scope_constraints(Rest,E,N,NewScope).
% at_least(>1) and sats: decrease counter
update_scope_constraints([scope(at_least(K),Prop)|Rest],E,N,[scope(at_least(KK),Prop)|NewScope]) :-
	K > 1,
	has_property(E,Prop),
	KK is K-1,
	update_scope_constraints(Rest,E,N,NewScope).
% at_least but unsat: keep
update_scope_constraints([scope(at_least(K),Prop)|Rest],E,N,[scope(at_least(K),Prop)|NewScope]) :-
	\+ has_property(E,Prop),
	update_scope_constraints(Rest,E,N,NewScope).
% at_most sat: decrease counter, but only if it was positive (else fail)
update_scope_constraints([scope(at_most(K),Prop)|Rest],E,N,[scope(at_most(KK),Prop)|NewScope]) :-
	K > 0,
	has_property(E,Prop),
	KK is K-1,
	update_scope_constraints(Rest,E,N,NewScope).
% at_most unsat: keep
update_scope_constraints([scope(at_most(K),Prop)|Rest],E,N,[scope(at_most(K),Prop)|NewScope]) :-
	\+ has_property(E,Prop),
	update_scope_constraints(Rest,E,N,NewScope).
% exactly sat: decrease counter, but only if it was positive (else fail)
update_scope_constraints([scope(exactly(K),Prop)|Rest],E,N,[scope(exactly(KK),Prop)|NewScope]) :-
	K > 0,
	has_property(E,Prop),
	KK is K-1,
	update_scope_constraints(Rest,E,N,NewScope).
% exactly unsat: keep
update_scope_constraints([scope(exactly(K),Prop)|Rest],E,N,[scope(exactly(K),Prop)|NewScope]) :-
	\+ has_property(E,Prop),
	update_scope_constraints(Rest,E,N,NewScope).
% all sat: keep (fail on unsat)
update_scope_constraints([scope(all,Prop)|Rest],E,N,[scope(all,Prop)|NewScope]) :-
	has_property(E,Prop),
	update_scope_constraints(Rest,E,N,NewScope).

% update histogram based WType and element just sampled
% draw with replacement: same as before
next_histogram(wr,_,H,H).
% draw without replacement: delete element just sampled
next_histogram(wor,E,In,Out) :-
	delete_from_hist(In,E,Out).

%%%%%%%%%%%%%%%%% all constrained version %%%%%%%%%%%%%%%%%
%% all diff -- note limited scalability (cf birthday)
draw_all(diff,Class,wr,set,M,Hist,Instance,TrialID) :-
	draw_all_diff_wr_set(M,Hist,[],Instance,TrialID,[],Class).
draw_all(diff,Class,wr,seq,M,Hist,Instance,TrialID) :-
	draw_all_diff_wr_seq(M,Hist,Instance,TrialID,[],Class).
draw_all(diff,Class,wor,set,M,Hist,Instance,TrialID) :-
	draw_all_diff_wor_set(M,Hist,[],Instance,_,TrialID,[],Class).
draw_all(diff,Class,wor,seq,M,Hist,Instance,TrialID) :-
	draw_all_diff_wor_seq(M,Hist,Instance,_,TrialID,[],Class).
	
draw_all(diff,Class,wor,set,M,Hist,Instance,Rest,TrialID) :-
	draw_all_diff_wor_set(M,Hist,[],Instance,Rest,TrialID,[],Class).
draw_all(diff,Class,wor,seq,M,Hist,Instance,Rest,TrialID) :-
	draw_all_diff_wor_seq(M,Hist,Instance,Rest,TrialID,[],Class).

draw_all_diff_wr_seq(0,_,[],_,_,_).
draw_all_diff_wr_seq(M,Hist,[E|Draw],TrialID,Seen,Class) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	el_has_class_value(E,Class,Val),
	\+ member(Val,Seen),
	MM is M-1,
	draw_all_diff_wr_seq(MM,Hist,Draw,TrialID,[Val|Seen],Class).

draw_all_diff_wor_seq(0,_,[],_,_,_).
draw_all_diff_wor_seq(M,Hist,[E|Draw],TrialID,Seen,Class) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	el_has_class_value(E,Class,Val),
	\+ member(Val,Seen),
	delete_from_hist(Hist,E,Hist2),
	MM is M-1,
	draw_all_diff_wor_seq(MM,Hist2,Draw,TrialID,[Val|Seen],Class).

draw_all_diff_wr_set(0,_,Acc,Acc,_,_,_).
draw_all_diff_wr_set(M,Hist,Acc,Inst,TrialID,Seen,Class) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	el_has_class_value(E,Class,Val),
	\+ member(Val,Seen),
	add_to_hist(Acc,1,E,Next),
	MM is M-1,
	draw_all_diff_wr_set(MM,Hist,Next,Inst,TrialID,[Val|Seen],Class).

draw_all_diff_wor_set(0,Hist,Acc,Acc,Hist,_,_,_).
draw_all_diff_wor_set(M,Hist,Acc,Inst,Rest,TrialID,Seen,Class) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	el_has_class_value(E,Class,Val),
	\+ member(Val,Seen),
	add_to_hist(Acc,1,E,Next),
	delete_from_hist(Hist,E,Hist2),
	MM is M-1,
	draw_all_diff_wor_set(MM,Hist2,Next,Inst,Rest,TrialID,[Val|Seen],Class).

%% all same
draw_all(same,Class,wr,set,M,Hist,Instance,TrialID) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	add_to_hist([],1,E,Next),
	MM is M-1,
	el_has_class_value(E,Class,Val),
	draw_all_same_wr_set(MM,Hist,Next,Instance,TrialID,Val,Class).
draw_all(same,Class,wr,seq,M,Hist,[E|Instance],TrialID) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	MM is M-1,
	el_has_class_value(E,Class,Val),
	draw_all_same_wr_seq(MM,Hist,Instance,TrialID,Val,Class).
draw_all(same,Class,wor,set,M,Hist,Instance,TrialID) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	delete_from_hist(Hist,E,Hist2),
	add_to_hist([],1,E,Next),
	MM is M-1,
	el_has_class_value(E,Class,Val),
	draw_all_same_wor_set(MM,Hist2,Next,Instance,_,TrialID,Val,Class).
draw_all(same,Class,wor,seq,M,Hist,[E|Instance],TrialID) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	delete_from_hist(Hist,E,Hist2),
	MM is M-1,
	el_has_class_value(E,Class,Val),
	draw_all_same_wor_seq(MM,Hist2,Instance,_,TrialID,Val,Class).
	
draw_all(same,Class,wor,set,M,Hist,Instance,Rest,TrialID) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	delete_from_hist(Hist,E,Hist2),
	add_to_hist([],1,E,Next),
	MM is M-1,
	el_has_class_value(E,Class,Val),
	draw_all_same_wor_set(MM,Hist2,Next,Instance,Rest,TrialID,Val,Class).
draw_all(same,Class,wor,seq,M,Hist,[E|Instance],Rest,TrialID) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	delete_from_hist(Hist,E,Hist2),
	MM is M-1,
	el_has_class_value(E,Class,Val),
	draw_all_same_wor_seq(MM,Hist2,Instance,Rest,TrialID,Val,Class).

draw_all_same_wr_seq(0,_,[],_,_,_).
draw_all_same_wr_seq(M,Hist,[E|Draw],TrialID,Val,Class) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	el_has_class_value(E,Class,Val),
	MM is M-1,
	draw_all_same_wr_seq(MM,Hist,Draw,TrialID,Val,Class).

draw_all_same_wor_seq(0,_,[],_,_,_).
draw_all_same_wor_seq(M,Hist,[E|Draw],TrialID,Val,Class) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	el_has_class_value(E,Class,Val),
	delete_from_hist(Hist,E,Hist2),
	MM is M-1,
	draw_all_same_wor_seq(MM,Hist2,Draw,TrialID,Val,Class).

draw_all_same_wr_set(0,_,Acc,Acc,_,_,_).
draw_all_same_wr_set(M,Hist,Acc,Inst,TrialID,Val,Class) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	el_has_class_value(E,Class,Val),
	add_to_hist(Acc,1,E,Next),
	MM is M-1,
	draw_all_same_wr_set(MM,Hist,Next,Inst,TrialID,Val,Class).

draw_all_same_wor_set(0,Hist,Acc,Acc,Hist,_,_,_).
draw_all_same_wor_set(M,Hist,Acc,Inst,Rest,TrialID,Val,Class) :-
	M > 0,
	sample(Hist,E,id(M,TrialID)),
	el_has_class_value(E,Class,Val),
	add_to_hist(Acc,1,E,Next),
	delete_from_hist(Hist,E,Hist2),
	MM is M-1,
	draw_all_same_wor_set(MM,Hist2,Next,Inst,Rest,TrialID,Val,Class).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% sampling base pred %%%%%%%%%%%%%%%%%%%

% sample(+Hist,-Element,+ID)
% Element is a value from Hist sampled on trial ID according to the histogram weights
% normalizing not really needed if Hist happens to be a distribution already... 
sample(H,E,ID) :-
	normalize(H,PL),
	sample_discrete(PL,E,ID).

% normalize a weighted list [w1-v1,...,wn-vn] into a distribution [p1:v1,...,pn:vn]
normalize(WL,PL) :-
	sum_weights(WL,0,N),
	divide_weights(WL,N,PL).
sum_weights([],N,N).
sum_weights([W-_|T],A,N) :-
	M is A+W,
	sum_weights(T,M,N).
divide_weights([],_,[]).
divide_weights([W-E|T],N,[P-E|S]) :-
	P is W/N,
	divide_weights(T,N,S).

% sample from a discrete distribution [p1-v1,...,pn-vn] - same principle, but probabilities given
% first prunes all elements with p=0, i.e., will not return those (to avoid zero-division errors)
sample_discrete(WL,E,ID) :-
    prune_zeros(WL,NL),
    sample_d(NL,1.0,E,ID).
sample_d([P-E|T],A,E,ID) :-
    Q is P/A,
    sd(Q,P,E,T,A,ID).
sample_d([P-H|T],A,E,ID) :-
    Q is P/A,
    \+ sd(Q,P,H,T,A,ID),
    N is A*(1-Q),
    sample_d(T,N,E,ID).
P::sd(P,_,_,_,_,_).

prune_zeros([],[]).
prune_zeros([P-H|T],[P-H|TP]) :-
    P > 0,
    prune_zeros(T,TP).
prune_zeros([P-_|T],TP) :-
    P =:= 0,
    prune_zeros(T,TP).
	
