%Expected outcome:
% 0.0::a(3).
% 0.5::a(1).
% 1.0::a(2).
% 0.5::b(4).
% 0.0::b(5).

t(_,X)::a(X).
t(_,X)::b(X).