%Expected outcome:
% 0.75::b(X); 0.25::c(X).
% 0.5::a(X):-b(X).
t(_)::b(X) ; t(_)::c(X).
t(_)::a(X) :-b(X).