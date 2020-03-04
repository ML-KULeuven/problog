% This test fails.
%Expected outcome:
% 0.5::a(X); 0.5::b(Y) :- X == Y, between(1, 2, X), between(1, 2, Y).


t(_)::a(X); t(_)::b(Y) :- X == Y, between(1, 2, X), between(1, 2, Y).