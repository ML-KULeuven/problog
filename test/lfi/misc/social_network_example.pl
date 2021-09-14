% Social network (friends & smokers) LFI example
% Source: https://dtai.cs.kuleuven.be/problog/tutorial/learning/02_smokers.html
%Expected outcome:
% 0.666666666666667::stress(X) :- person(X).
% 0.333333333333333::influences(X,Y) :- person(X), person(Y).
% smokes(X) :- stress(X).
% smokes(X) :- friend(X,Y), influences(Y,X), smokes(Y).
% person(1).
% person(2).
% person(3).
% person(4).
% friend(1,2).
% friend(2,1).
% friend(2,4).
% friend(3,2).
% friend(4,2).

t(_)::stress(X) :- person(X).
t(_)::influences(X,Y) :- person(X), person(Y).

smokes(X) :- stress(X).
smokes(X) :- friend(X,Y), influences(Y,X), smokes(Y).

person(1).
person(2).
person(3).
person(4).

friend(1,2).
friend(2,1).
friend(2,4).
friend(3,2).
friend(4,2).

