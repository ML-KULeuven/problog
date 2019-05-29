0.2::a.
1/10::no_cool.
normal(20,2)~x(1):-a, no_cool.
normal(30,2)~x(1):-\+a.
broken:- x(1)~=X, conS(X>20).
query(broken).
