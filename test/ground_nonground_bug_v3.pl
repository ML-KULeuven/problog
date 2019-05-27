%Expected outcome:  
% q  0.08 

0.1::p1.
0.8::p2.

p(A, B) :- var(A), p1, A = unknown.
p(A, B) :- var(B), p2, B = unknown.

fill(A, B) :- A == unknown, B == unknown.
fill(A, B) :- p(A, B), fill(A, B).

q :- fill(A, B).

query(q).

