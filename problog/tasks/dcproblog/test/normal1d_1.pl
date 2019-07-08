1/5::a.
normal(0,4)~t:-a.
1/10::no_cool.
broken:- no_cool, t~=T, conS(T>0).
query(broken).
