
or(neg(Q), neg(Q,Proofs)) :- !,
    findall(P, or(Q, P), Proofs).
    
or(Q,and(p(P,Q,I), Proof)) :-
    ad(Heads, Body), 
    member(p(P,Q,I),Heads),
    and(Body, Proof).
    
or(Q, and(p(1.0,Q,Q),[])) :-
    predicate_property(Q,built_in), !,
    Q.

%or(Q, _) :-
%    \+ (ad(Heads, _), member(p(_,Q,_),Heads)),
%    throw(unknown_clause(Q)).

    
and([],[]).

and([H|T],[H2|T2]) :-
    or(H,H2),
    and(T,T2).

prove(Q,Q,Proofs) :- or(Q,Proofs).
