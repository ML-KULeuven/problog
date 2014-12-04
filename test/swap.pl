%Expected outcome:
% s1(1) 0.734375
% s2(1) 0.734375

0.5::f(1,2).
0.5::f(2,1).
0.5::f(1,3).
0.5::f(2,3).

0.5::b(X).

s1(X) :- b(X).
s1(X) :- f(X,Y),s1(Y).

s2(X) :- f(X,Y),s2(Y).
s2(X) :- b(X).

% s3(X) :- s3(Y), f(X,Y).
% s3(X) :- b(X).

query(s1(1)).
query(s2(1)).
% query(s3(1)).
