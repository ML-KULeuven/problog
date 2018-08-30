%Expected outcome:  
% q 0.1

0.1::p1.

p(A) :- p1, A = unknown.

fill(A) :- A == unknown.
fill(A) :- p(A), fill(A).

q :- fill(A).

query(q).


