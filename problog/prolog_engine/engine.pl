add_to_proof(X,[],Proof,[X|Proof]).

add_to_proof(_,[_|_],Proof,Proof).

prove([],[]).

prove([neg(H)|T], Proof) :-
    predicate_property(H,built_in),!,
    \+H,
    prove(T,Proof).

prove([neg(H)|T], Proof) :-
    ad(L,B),
    member(p(P,H),L),
    prove(B, Proof1),
    prove(T,Proof2),
    append(Proof1,Proof2,Proof3),
    add_to_proof(neg(p(P,H)), B, Proof3, Proof).

    

prove([H|T], Proof) :-
    predicate_property(H,built_in),
    H,
    prove(T,Proof).
    

prove([H|T], Proof) :-
    ad(L,B),
    member(p(P,H),L),
    prove(B, Proof1),
    prove(T,Proof2),
    append(Proof1,Proof2,Proof3),
    add_to_proof(p(P,H), B, Proof3, Proof).

prove(X,X,Proof) :- prove([X],Proof).
