prove([],[]).

prove([neg(H)|T], [Proof]) :-
    predicate_property(H,built_in),!,
    \+H,
    prove(T,Proof).

prove([neg(H)|T], [neg(p(P,H)),Proof]) :-
    ad(L,B),
    member(p(P,H),L),
    prove(B, Proof1),
    prove(T,Proof2),
    append(Proof1,Proof2,Proof).

    

prove([H|T], [Proof]) :-
    predicate_property(H,built_in),
    H,
    prove(T,Proof).
    

prove([H|T], [p(P,H)|Proof]) :-
    ad(L,B),
    member(p(P,H),L),
    prove(B, Proof1),
    prove(T,Proof2),
    append(Proof1,Proof2,Proof).
