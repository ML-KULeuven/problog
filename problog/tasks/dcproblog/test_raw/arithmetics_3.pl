0.2::a.
normal(20,2)~t:-a.
normal(21,2)~x:-a.

1/10::no_cool.
broken:- no_cool, t~=T, x~=X, addS(T,X,S), conS(S>20).
query(broken).
