prove(Q,Q,Proof,P) :- or(Q,Proof,1.0,P,[]).


or(Q,builtin(Q),P,P,_) :-
    predicate_property(Q, built_in),
    Q,!.

%or(Q,foreign(Q),P,P,_) :-
%    predicate_property(Q, foreign),
%    Q.

or(Q,cycle(Q),P,P,Context) :- ground(Q), member(Q,Context),!,fail ,!. % No cycles for now

or(Q,and(Q, Proof),Pprev,P,Context) :-
    rule(Q, Body),
    and(Body,Proof,Pprev,P,[Q|Context]).

or(Q,neural_fact(P,Q,I,Net,Vars),Pprev,P,_) :-
    fact(nn(Net,Vars),Q,I),
    apply(Net,[Pnet|Vars]),
    nonvar(Pnet),
    P is Pnet*Pprev,
    get_flag(pmin,Pmin),
    P > Pmin.

or(Q,fact(P,Q,I),Pprev,P,_) :-
    fact(Pfact,Q,I),
    number(Pfact),
    P is Pfact*Pprev,
    get_flag(pmin,Pmin),
    P > Pmin.
    
    
and([],[],P,P,_).

and([H|T],[H2|T2],Pprev,P,Context) :-
    or(H,H2,Pprev,P1,Context),
    and(T,T2,P1,P,Context).


k_best(K,Q,X,Y, Goal, K_Best) :-
   set_flag(pmin,0.0),
   State = state([]),
   (  call(Goal),
      arg(1, State, S0),
      kbest_add(K,S0, Y-proof(Q,X), S,Pmin2),
      nb_setarg(1, State, S),
      set_flag(pmin,Pmin2),
      fail
   ;  arg(1, State, K_Best)
   ).

kbest_add(K,List_Of_Proofs, New_Proof, New_List_Of_Proofs, Pmin) :-% writeln(New_Proof),
    ( length(List_Of_Proofs, L), L < K
    -> New_List_Of_Proofs = [New_Proof | List_Of_Proofs], Pmin = 0.0
    ;  select_k_best(K, [New_Proof | List_Of_Proofs], New_List_Of_Proofs, Pmin)
    ).

select_k_best(K, Prob_Proofs, K_Best_Proofs, Pmin) :-
    keysort(Prob_Proofs, Prob_Proofs_Asc),
    length(K_Best_Proofs, K),
    append(_, K_Best_Proofs, Prob_Proofs_Asc),
    K_Best_Proofs = [Pmin-_|_].
    
top(K,Query,Proofs) :- k_best(K,Q,X,Y,prove(Query,Q,X,Y), Xs),pairs_keys_values(Xs,_,Proofs).%, writeln(Proofs).
