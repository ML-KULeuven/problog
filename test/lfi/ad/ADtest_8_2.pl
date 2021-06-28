%Expected outcome:
% x1.
% x2.
% x3.
% x4.
% my_succ(x1,x2).
% my_succ(x2,x3).
% my_succ(x3,x4).
% 0.333333333333333::a(X); 0.166666666666667::b(Y); 0.5::c(Z) :- my_succ(X,Y), my_succ(Y,Z).


x1.
x2.
x3.
x4.
my_succ(x1,x2).
my_succ(x2,x3).
my_succ(x3,x4).
t(_)::a(X); t(_)::b(Y); t(_)::c(Z) :- my_succ(X,Y), my_succ(Y,Z).