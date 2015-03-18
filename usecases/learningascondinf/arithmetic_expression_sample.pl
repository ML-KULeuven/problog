0.7::leaf(T).
0.5::operator('+',T) ; 0.5::operator('-',T).
Px::l('x',T); P::l(0,T) ; P::l(1,T) ; P::l(2,T) ; P::l(3,T) ; P::l(4,T) ; P::l(5,T) ; P::l(6,T) ; P::l(7,T) ; P::l(8,T) ; P::l(9,T) :- P is 0.5/10, Px is 0.5.

expr(L) :- expr(L,0,Tr).
expr(L,T,T) :- leaf(T), l(L,T).
expr([L,O,R],T,Tr) :-
    \+leaf(T), operator(O,T),
    Tn1 is T+1, expr(L,Tn1,Tr1),
    Tn2 is Tr1+1, expr(R,Tn2,Tr).

query(expr(E)).

