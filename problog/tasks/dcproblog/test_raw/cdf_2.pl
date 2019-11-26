0.2::a.
normal(20,4)~x:-a.
sigmoid(X)::t:- x~=X.
1/10::no_cool.
broken:- no_cool, \+t.
query(broken).
