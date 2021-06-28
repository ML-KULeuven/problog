%Expected outcome:
% 0.6::a(1,Y); 0.4::b(1,Y) :- 1=Y, between(1,2,1), between(1,2,Y).
% 0.5::a(2,Y); 0.5::b(2,Y) :- 2=Y, between(1,2,2), between(1,2,Y).

t(_,X)::a(X,Y); t(_,X)::b(X,Y) :- X=Y, between(1,2,X), between(1,2,Y).