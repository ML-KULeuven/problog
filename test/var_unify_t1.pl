%Expected outcome:
% p(1,1) 1
% p(2,2) 1
% p(3,3) 1


a(1).
a(2).
a(3).

eq(X,X).

p(X,Y) :- X=Y, a(X).

query(p(_,_)).