%Expected outcome:%
%        q([])   0.336
%    q([a, a])   0.096
% q([a, b, a])   0.024
%    q([a, b])   0.036
%       q([a])   0.368
%    q([b, a])   0.056
%       q([b])   0.084
% s([a, b, a])   1
%   s2([a, b])   1
%       s3(X2)   0

0.3::p(a).
0.2::p(b).
0.4::p(a).

q(L) :- findall(X, p(X), L).

query(q(L)).


r(a).
r(b).
r(a).

t(_) :- fail.

s(L) :- findall(X, r(X), L).

query(s(L)).

s2(L) :- all(X, r(X), L).

query(s2(L)).

s3(L) :- all(X, t(X), L).

query(s3(L)).