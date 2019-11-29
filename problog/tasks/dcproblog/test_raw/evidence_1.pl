0.2::a.
normal(20,2)~x(1):-a.
normal(30,2)~x(1):-\+a.
normal(X,4)~t:- X as x(1).
1/10::no_cool.
broken:- no_cool, T as t, T>20.
evidence(\+a).
query(broken).
