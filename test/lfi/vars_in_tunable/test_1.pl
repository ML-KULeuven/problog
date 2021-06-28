%Expected outcome:
% person(alice).
% person(bob).
% 0.5::hair(alice) :- person(alice).
% 0.75::hair(bob) :- person(bob).

person(alice).
person(bob).

t(_,X)::hair(X) :- person(X).

% see Simple/test_18.pl