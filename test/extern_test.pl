%Expected outcome:
% concat_list([a, b],[c, d],[a, b, c, d]) 1
% concat_list([a, b],[c],[a, b, c]) 1
% concat_str(a,b,ab) 1
% int_plus(1,2,3) 1
% p(xy) 1
:- use_module('extern_lib.py').

query(concat_str(a,b,Z)).

query(concat_list([a,b],[c,d],Z)).

query(int_plus(1,2,Z)).

query(concat_list([a,b],[c],Y)).

p(Z) :- concat_str(x,y,Z).

query(p(Z)).