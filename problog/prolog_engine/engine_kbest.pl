prove(Q,Q,Proof,P) :- or(Q,Proof,P,[]).

or(Q,cycle(Q),1.0,Context) :- ground(Q), member(Q,Context), !.

or(Q,and(Q, Proof),P,Context) :-
    rule(Q, Body),
    and(Body,Proof,P,[Q|Context]).

or(Q,fact(P,Q,I),P,_) :-
    fact(P,Q,I),
    get_flag(pmin,Pmin),
    P > Pmin.
    
    
and([],[],1.0,_).

and([H|T],[H2|T2],P,Context) :-
    or(H,H2,P1,Context),
    and(T,T2,P2,Context),
    P is P1*P2,
    get_flag(pmin,Pmin),
    P > Pmin.   


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

kbest_add(K,List_Of_Proofs, New_Proof, New_List_Of_Proofs, Pmin) :-
    ( length(List_Of_Proofs, L), L < K
    -> New_List_Of_Proofs = [New_Proof | List_Of_Proofs], Pmin = 0.0
    ;  select_k_best(K, [New_Proof | List_Of_Proofs], New_List_Of_Proofs, Pmin)
    ).

select_k_best(K, Prob_Proofs, K_Best_Proofs, Pmin) :-
    keysort(Prob_Proofs, Prob_Proofs_Asc),
    length(K_Best_Proofs, K),
    append(_, K_Best_Proofs, Prob_Proofs_Asc),
    K_Best_Proofs = [Pmin-_|_].
    
top(K,Query) :- k_best(K,Q,X,Y,prove(Query,Q,X,Y), Xs),pairs_keys_values(Xs,_,Proofs), writeln(Proofs).
