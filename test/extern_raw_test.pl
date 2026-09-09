%Expected outcome:
% both_bound 1
% first_free 1
% second_free 1
% both_free 1
% first_wrong 0
% second_wrong 0
% first_wrong_second_free 0
% first_free_second_wrong 0
:- use_module('extern_lib.py').

% pair/2 answers pair(one, two), so a bound argument naming anything else
% has to make the call fail, whichever argument it is.
both_bound :- pair(one, two).
first_free :- pair(X, two), X == one.
second_free :- pair(one, X), X == two.
both_free :- pair(X, Y), X == one, Y == two.
first_wrong :- pair(nope, two).
second_wrong :- pair(one, nope).
first_wrong_second_free :- pair(nope, _).
first_free_second_wrong :- pair(_, nope).

query(both_bound). query(first_free). query(second_free). query(both_free).
query(first_wrong). query(second_wrong).
query(first_wrong_second_free). query(first_free_second_wrong).
