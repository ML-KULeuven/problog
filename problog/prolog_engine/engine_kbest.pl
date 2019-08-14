ad([p(0.5,edge(1,2),1)],[]).
ad([p(0.6,edge(1,3),2)],[]).
ad([p(0.9,edge(2,4),3)],[]).
ad([p(0.9,edge(2,5),4)],[]).
ad([p(0.6,edge(3,6),5)],[]).
ad([p(0.9,edge(4,6),6)],[]).
ad([p(0.9,edge(5,6),7)],[]).
ad([p(1.0,path(A1,A2),8)],[edge(A1,A2)]).
ad([p(1.0,path(A1,A2),9)],[edge(A3,A2),path(A1,A3)]).
ad([p(1.0,query(path(1,6)),10)],[]).


or(Q,and(p(P,Q,I), Proof),P3) :-
    get_flag(pmin,Pmin),
    ad(Heads, Body), 
    member(p(P,Q,I),Heads),
    and(Body, Proof,P2),
    P3 is P*P2,
    P3 > Pmin.
    
    
and([],[],1.0).

and([H|T],[H2|T2],P) :-
    get_flag(pmin,Pmin),
    or(H,H2,P1),
    and(T,T2,P2),
    P is P1*P2,
    P > Pmin.


k_best(X,Y, Goal, K_Best) :-
   State = state([]),
   set_flag(pmin,0.0),
   (  call(Goal),
      arg(1, State, S0),
      kbest_add(S0, Y-X, S,Pmin2),
      nb_setarg(1, State, S),
      set_flag(pmin,Pmin2),
      fail
   ;  arg(1, State, K_Best)
   ).

kbest_add(List_Of_Proofs, New_Proof, New_List_Of_Proofs, Pmin) :-
    K = 1,
    ( length(List_Of_Proofs, L), L < K
    -> New_List_Of_Proofs = [New_Proof | List_Of_Proofs], Pmin = 0.0
    ;  select_k_best(K, [New_Proof | List_Of_Proofs], New_List_Of_Proofs, Pmin)
    ).

select_k_best(K, Prob_Proofs, K_Best_Proofs, Pmin) :-
    %findall(Probability-Proof,
    %        (member(proof(Proof,Probability), Proofs)),
    %        Prob_Proofs),
    keysort(Prob_Proofs, Prob_Proofs_Asc),
    length(K_Best_Proofs, K),
    append(_, K_Best_Proofs, Prob_Proofs_Asc),
 %   pairs_values(K_Best_Proofs0, K_Best_Proofs),
    K_Best_Proofs = [Pmin-_|_].
    
top(Xs) :- k_best(X,Y, or(path(1,6), X,Y), Xs).
