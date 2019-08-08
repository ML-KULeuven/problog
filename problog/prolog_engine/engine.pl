ad([p(1.0,a(A1),1)],[neg(b(A1))]).
ad([p(1.0,b(A1),2)],[c(A1),d(A1)]).
ad([p(0.5,c(0),3)],[]).
ad([p(0.5,d(0),4)],[]).
%ad([p(1.0,a(A1),5)],[e(A1)]).
%ad([p(1.0,e(1),6)],[]).
%ad([p(1.0,e(2),7)],[]).

prove(Q, Proofs) :-
    findall(and(p(P,Q,I),Proof), (ad(Heads, Body), member(p(P,Q,I),Heads), and(Body, Proof), Proofs)).
    
or(neg(Q), neg(Proofs)) :-
    findall(P, or(Q, P), Proofs).
    
or(Q,and(p(P,Q,I), Proof)) :-
    ad(Heads, Body), 
    member(p(P,Q,I),Heads), 
    and(Body, Proof).

and([],[]).

and([H|T],[H2|T2]) :-
    or(H,H2),
    and(T,T2).

