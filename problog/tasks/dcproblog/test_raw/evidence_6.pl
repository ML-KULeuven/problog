0.2::a.
normal(20,2)~x(1):-a.
normal(X,4)~t:- x(1)~=X.
1/10::no_cool.
broken:- no_cool, t~=T, conS(T>20).

T<40::c:- t~=T.
evidence(\+c).
query(broken).
