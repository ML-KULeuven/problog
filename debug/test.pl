:- use_module('mymodule.py').

my_int("1").

my_sum(I) :- my_int(X), my_int(Y), str_sum(X, Y, I).

query(my_sum(I)).
