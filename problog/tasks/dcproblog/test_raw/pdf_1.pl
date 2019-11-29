0.2::a.
normal(30,5)~x:-a.
normal(X,4)~t:- x~=X.
1/10::no_cool.
broken:- no_cool, t~=T, conS(T>20).
query(broken).
