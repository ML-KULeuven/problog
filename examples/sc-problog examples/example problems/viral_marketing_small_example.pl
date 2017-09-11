% Background knowledge
person(a).
person(b).
person(c).
person(d).

% Probabilistic facts
0.7::directed(a,b).
0.2::directed(b,c).
0.4::directed(b,d).
0.6::directed(c,d).

% Relations
trusts(X,Y) :- directed(X,Y).
trusts(X,Y) :- directed(Y,X).

buys(X) :- marketed(X).
buys(X) :- trusts(X,Y), buys(Y).

% Decision variables
?::marketed(P) :- person(P).

query(buys(a)).
query(buys(b)).
query(buys(c)).
query(buys(d)).
