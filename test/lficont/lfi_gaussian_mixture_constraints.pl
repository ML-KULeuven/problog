t(0.5)::c(1,ID); t(0.5)::c(2,ID).

t(normal(1 ,10))::fa(ID) :- c(1,ID).
t(normal(10,10))::fa(ID) :- c(2,ID).

% Constraints
%constraint_f :-   c(C,1), \+c(C,7).
%constraint_f :- \+c(C,1),   c(C,7).
%constraint_f :-   c(C,3), \+c(C,14).
%constraint_f :- \+c(C,3),   c(C,14).
%constraint_f :-   c(C,3), \+c(C,15).
%constraint_f :- \+c(C,3),   c(C,15).
%constraint_f :-   c(C,5), \+c(C,14).
%constraint_f :- \+c(C,5),   c(C,14).

