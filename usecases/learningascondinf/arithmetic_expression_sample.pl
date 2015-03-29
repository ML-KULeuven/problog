0.7::leaf(T).

0.5::operator('+',T) ; 0.5::operator('-',T).

Px::l(x,T); P::l(0,T) ; P::l(1,T) ; P::l(2,T) ; P::l(3,T) ; P::l(4,T) ; P::l(5,T) ; P::l(6,T) ; P::l(7,T) ; P::l(8,T) ; P::l(9,T) :- P is 0.5/10, Px is 0.5.

expr(L) :- expr(L,0,Tr).

expr(L,T1,T2) :- leaf(T1), T2 is T1+1, l(L,T1).
expr([EL,O,ER],T1,T2) :-
    \+leaf(T1),
    Ta is T1+1, operator(O,Ta),
    expr(EL,Ta,Tb),
    expr(ER,Tb,T2).

query(expr(E)).

