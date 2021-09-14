%Expected outcome:
% 0.75::p(a0).
% 0.75::p(a1).
% 0.5::q(b2) :- p(a0).
% 0.5::q(b2) :- p(a1).


t(_)::p(a0).
t(_)::p(a1).

t(_, X)::q(b2) :- p(X).