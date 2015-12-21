
member(X,[X|_]).
member(X,[_|T]) :- member(X,T).

0.5::c1(X) :- member(X, [0,1,2,3,4,5,6,8,9]).
0.5::e :- c1(X).

query(e).

