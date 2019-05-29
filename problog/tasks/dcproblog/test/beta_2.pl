0.2::a.
beta(1,2)~b:-a.
beta(2,2)~b:-\+a.
0.4::t:- b~=B, conS(B<0.5).
1/10::no_cool.
broken:- no_cool, t.
query(broken).
