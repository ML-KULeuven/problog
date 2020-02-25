:- use_module(library(lists)).

% Allow scoping over conjunctions.
C:R :- nonvar(R), R = (A, B),  C:A, C:B.

% Transform simple-scoped rules to list-scoped rules.
C:R :- nonvar(C), C = [C1|_], C1:R.
C:R :- nonvar(C), \+ is_list(C), [C]:R.

% Allow scope lists to be in arbitrary order.
List:Predicate :- nonvar(List), select(X, List, R), [X|R]:Predicate.

% Need to wrap all predicates in a scope, including builtins!
_:Pred :- nonvar(Pred), try_call(Pred).