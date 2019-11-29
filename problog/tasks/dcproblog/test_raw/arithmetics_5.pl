normal(15,5)~s1.
normal(10,5)~s2.
normal(S,2)~t:- s1~=S1, s2~=S2, addS(S1,S2,S).
1/10::no_cool.
broken:- no_cool, t~=T, conS(T>20).
broken:- t~=T, conS(T>30).

query(broken).
