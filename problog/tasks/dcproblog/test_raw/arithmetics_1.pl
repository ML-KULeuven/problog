0.2::a.
normal(20,2)~t:-a.
1/10::no_cool.
broken:- no_cool, t~=T, addS(T,3,S), conS(S>20).
query(broken).
