0.2::a.
normal(20,2)~x(1):-a.
normal(X,4)~t:- x(1)~=X.
1/10::no_cool.
c:- t~=T, conS(T>20).
broken:- no_cool, \+c.

query(broken).
