%Expected outcome:
% b_body_clause(a,1.0) 1
% a_head_clause(b,1.0) 1
% c_head_clause((d, e); f,1.0) 1
% long_body_clause(c,1.0) 1
% long_body_clause2(c,1.0) 1
a :- b.

c :- d,e;f.

b_body_clause(X, Prob) :- clause(X,b, Prob).
a_head_clause(X, Prob) :- clause(a,X, Prob).
c_head_clause(X, Prob) :- clause(c,X, Prob).
long_body_clause(X, Prob) :- clause(X,(d,e;f), Prob).
long_body_clause2(X, Prob) :- clause(X,((d,e);f), Prob).

query(b_body_clause(_,_)).
query(a_head_clause(_,_)).
query(c_head_clause(_,_)).
query(long_body_clause(_,_)).
query(long_body_clause2(_,_)).