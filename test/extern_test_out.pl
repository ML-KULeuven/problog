%Expected outcome:
% int_plus(1,2,3) 1
% int_plus_times(1,2,3,2) 1
% int_plus_times(1,2,3,3) 0
% p 1
% b(1) 1
% b(2) 1
% b(3) 1
% b(4) 1
% c(3) 1
% c(5) 0

:- use_module('extern_lib.py').

query(int_plus(1,2,3)).

query(int_plus_times(1,2,3,2)).

query(int_plus_times(1,2,3,3)).

p :- concat_str(x,y,xy).

query(p).

b(X) :- int_between(1, 4, X).

query(b(X)).

c(X) :- int_between(1, 4, X).

query(c(3)).

query(c(5)).