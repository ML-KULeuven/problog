%Expected outcome:
% doublecard 0.25
% samecard(q,h) 0.0625
% samecard(q,s) 0.0625
% samecard(k,h) 0.0625
% samecard(k,s) 0.0625


position(1).
position(2).

0.25::card(C,q,h);0.25::card(C,k,h);0.25::card(C,q,s);0.25::card(C,k,s) <- position(C).

doublecard :- card(C1,X,Y), card(C2,X,Y), C1 < C2.

spade(C) :- card(C,_,s).

samecard(A,B) :- card(1,A,B),card(2,A,B).

query(doublecard).
query(samecard(q,h)).
query(samecard(q,s)).
query(samecard(k,h)).
query(samecard(k,s)).
