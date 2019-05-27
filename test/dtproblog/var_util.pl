:- use_module(library(lists)).

?::bet(w1, 5); ?::bet(w2, 7); ?::bet(w3, 10).

valid_bets(MaxCost) :-
    findall(Cost, bet(Bet, Cost), Bets),
    sum_list(Bets, Sum),
    Sum =< MaxCost.

0.1::win(w1); 0.5::win(w2); 0.3::win(w3).

%evidence(valid_bets(12)).

utility(bet(X, C), -C) :- bet(X, C).
utility(win(X), 10) :- bet(X, _).
