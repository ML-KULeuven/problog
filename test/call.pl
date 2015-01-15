%Expected outcome:
% q1 0.58
% q2 0.58
% q3 0.3
% q4 0.3
% q5 0.58

0.3::p(1).
0.4::p(2).

q1 :- call(p(X)).
q2 :- p(X).

q3 :- call(p(1)).
q4 :- p(1).

q5 :- call(p,X).


query(q1).
query(q2).
query(q3).
query(q4).
query(q5).
