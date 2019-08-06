ad([p(1.0,a(A1))],[neg(b(A2)),c(A1)]).
ad([p(0.5,b(0))],[]).
ad([p(1.0,c(A1))],[d(A1)]).
ad([p(1.0,d(A1))],['is'(A1,2-1)]).


prove([],[]).

prove([neg(H)|T], [Proof]) :-
    predicate_property(H,built_in),!
    \+H,
    prove(T,Proof).

prove([neg(H)|T], [Proof]) :-
    ad(L,B),
    member(p(_,H),L),
    prove(B, Proof1),
    prove(T,Proof2),
    append(Proof1,Proof2,Proof).

    

prove([H|T], [Proof]) :-
    predicate_property(H,built_in),
    H,
    prove(T,Proof).
    

prove([H|T], [ad(L,B)|Proof]) :-
    ad(L,B),
    member(p(_,H),L),
    prove(B, Proof1),
    prove(T,Proof2),
    append(Proof1,Proof2,Proof).
