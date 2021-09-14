%Expected outcome:
% 0.8::p(0) :- between(0,2,0).
% 0.0::p(2) :- between(0,2,2).
% 0.2::p(1) :- between(0,2,1).
% 0.5::q(2) :- p(0).
% 1.0::q(2) :- p(1).
% 0.0::q(2) :- p(2).

t(_, X)::p(X):-between(0,2,X).
t(_, X)::q(2) :- p(X).