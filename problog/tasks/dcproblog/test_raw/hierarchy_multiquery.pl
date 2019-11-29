normal(5,5)~m.
uniform(4,M)~x:- m~=M, conS(4=<M).
q1:- x~=X, conS(X>6).
q2:- m~=M, conS(M>1/2).


% TODO add extra bookkeeping for multiple queries.
% TODO add extra bookkeeping for hierarchical models when doing partial integration.
query(q1).
query(q2).
