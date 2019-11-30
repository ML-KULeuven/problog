0.2::a.
b~beta(1,2):-a.
b~beta(2,2):-\+a.
0.4::t:- B is b, B<0.5.
1/10::no_cool.
broken:- no_cool, t.
query(broken).
