%Expected outcome: 
% num_balls(15) 3.9220542e-09

:- use_module(library(lists)).

probs([3.13035286e-01, 3.13035286e-01, 2.08690190e-01, 1.04345095e-01, 4.17380381e-02, 1.39126794e-02, 3.97505125e-03, 9.93762812e-04, 2.20836180e-04, 4.41672361e-05, 8.03040656e-06, 1.33840109e-06, 2.05907860e-07, 2.94154086e-08, 3.92205449e-09]).

num_balls(X) :-
    findall(N, between(1,15,N), L),
    probs(Probs),
    select_weighted(1, Probs, L, X, _).

query(num_balls(15)).

