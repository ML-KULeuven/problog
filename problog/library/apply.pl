
include(P, [], []).
include(P, [X|T], [X|S]) :-
    call(P, X),
    include(P, T, S).
include(P, [X|T], S) :-
    \+ call(P, X),
    include(P, T, S).

exclude(P, [], []).
exclude(P, [X|T], [X|S]) :-
    \+ call(P, X),
    include(P, T, S).
exclude(P, [X|T], S) :-
    call(P, X),
    include(P, T, S).

partition(_, [], [], []).
partition(P, [X|R], [X|S], T) :-
    call(P, X),
    partition(P, R, S, T).
partition(P, [X|R], S, [X|T]) :-
    \+ call(P, X),
    partition(P, R, S, T).

% TODO implement partition/5

maplist(_, []).
maplist(P, [A|S]) :-
    call(P, A),
    maplist(P, S).

% maplist(+Pred, +ListIn, -ListOut)
% Apply the predicate to each element in the list.
% The given predicate can be partially defined.
% e.g. maplist(plus(1), [1,2,3,4], Result) yields [2,3,4,5]
maplist(_, [], []).
maplist(P, [A|S], [B|T]) :-
    call(P, A, B),
    maplist(P, S, T).

maplist(_, [], [], []).
maplist(P, [A|S], [B|T], [C|R]) :-
    call(P, A, B, C),
    maplist(P, S, T, R).

maplist(_, [], [], [], []).
maplist(P, [A|S], [B|T], [C|R], [D|U]) :-
    call(P, A, B, C, D),
    maplist(P, S, T, R, U).

% foldl(+Pred, +List, +Init, -Result)
% Fold the list by recursively applying the predicate
%  to the Initial element and the first element of the list.
% e.g. foldl(plus, [1,2,3,4], 0, Sum) computes the sum of the list
foldl(_, [], RR, RR).
foldl(P, [L1|LR], L0,  RR) :-
    call(P, L1, L0, R),
    foldl(P, LR, R, RR).

foldl(_, [], [], RR, RR).
foldl(P, [L1|LR1], [L2|LR2], L0,  RR) :-
    call(P, L1, L2, L0, R),
    foldl(P, LR1, LR2, R, RR).

foldl(_, [], [], [], RR, RR).
foldl(P, [L1|LR1], [L2|LR2], [L3|LR3], L0,  RR) :-
    call(P, L1, L2, L3, L0, R),
    foldl(P, LR1, LR2, LR3, R, RR).

foldl(_, [], [], [], [], RR, RR).
foldl(P, [L1|LR1], [L2|LR2], [L3|LR3], [L4|LR4], L0,  RR) :-
    call(P, L1, L2, L3, L4, L0, R),
    foldl(P, LR1, LR2, LR3, LR4, R, RR).


% foldl_s(+Pred, +List, -Result)
% Fold the list by recursively applying the predicate
%  to the Initial element and the first element of the list.
% Takes the first element of the list as the Initial result.
% e.g. foldl_s(plus, [1,2,3,4], Sum) computes the sum of the list
foldl_noinit(P, [], []).
foldl_noinit(P, [L1|LR], RR) :-
    foldl(P, LR, L1, RR).


scanl(_, [], _, []).
scanl(P, [L1|LR], L0,  [R|RR]) :-
    call(P, L1, L0, R),
    scanl(P, LR, R, RR).

scanl(_, [], [], _, []).
scanl(P, [L1|LR1], [L2|LR2], L0, [R|RR]) :-
    call(P, L1, L2, L0, R),
    scanl(P, LR1, LR2, R, RR).

scanl(_, [], [], [], _, []).
scanl(P, [L1|LR1], [L2|LR2], [L3|LR3], L0, [R|RR]) :-
    call(P, L1, L2, L3, L0, R),
    scanl(P, LR1, LR2, LR3, R, RR).

scanl(_, [], [], [], [], _, []).
scanl(P, [L1|LR1], [L2|LR2], [L3|LR3], [L4|LR4], L0, [R|RR]) :-
    call(P, L1, L2, L3, L4, L0, R),
    scanl(P, LR1, LR2, LR3, LR4, R, RR).
