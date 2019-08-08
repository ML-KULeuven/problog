ad([p(0.3,p(1),1)],[]).
ad([p(0.2,p(2),2)],[]).
ad([p(0.6,p(1),3)],[]).
ad([p(1.0,query(p(1)),4)],[]).
ad([p(1.0,query(p(2)),5)],[]).

add_to_proof(X,[],Proof,[X|Proof]).

add_to_proof(_,[_|_],Proof,Proof).

prove([],[]).

prove([neg(H)|T], Proof) :-
    predicate_property(H,built_in),!,
    \+H,
    prove(T,Proof).

prove([neg(H)|T], Proof) :-
    ad(L,B),
    member(p(P,H,I),L),
    prove(B, Proof1),
    prove(T,Proof2),
    append(Proof1,Proof2,Proof3),
    add_to_proof(neg(p(P,H,I)), B, Proof3, Proof).

    

prove([H|T], Proof) :-
    predicate_property(H,built_in),
    H,
    prove(T,Proof).
    

prove([H|T], Proof) :-
    ad(L,B),
    member(p(P,H,I),L),
    prove(B, Proof1),
    prove(T,Proof2),
    append(Proof1,Proof2,Proof3),
    add_to_proof(p(P,H,I), B, Proof3, Proof).

prove(X,X,Proof) :- prove([X],Proof).
