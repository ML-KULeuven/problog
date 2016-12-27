%Expected outcome:
% test(1,a,b)  1
% test(2,a,b)  1
% test(3,b,c)  1
% test(4,a,c)  1 

:- use_module(library(cut)).

rule(1, a, b).
rule(2, a, c).
rule(3, b, c).


test(1, A, B) :- cut(rule(A, B)).
test(2, A, B) :- A = a, cut(rule(A, B)).
test(3, A, B) :- A = b, cut(rule(A, B)).
test(4, A, B) :- B = c, cut(rule(A, B)).

query(test(_, X, Y)).

