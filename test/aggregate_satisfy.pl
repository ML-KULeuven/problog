%Expected outcome:
% send_more_money([2, 8, 1, 7, 0, 3, 6, 5]) 1
% send_more_money([2, 8, 1, 9, 0, 3, 6, 7]) 1
:- use_module(library(collect)).
:- use_module(library(aggregate)).
:- use_module(library(lists)).

% Simply execute the code block.
collect_satisfy(CB, none, Result) :- execute(CB).

% Execute a conjunction (note that call/N does not support conjunctions.
execute((A,B)) :-
    execute(A),
    execute(B).

% Execute any other goal.
execute(G) :-
    G \= (_, _),
    call(G).

% Implement domain: enumerates the values from the domain (e.g. with between/3).
domain(X, DS) :- call(DS, X).
% Implement constraint: check the constraint by evaluating it in Prolog.
constraint(X) :- call(X).
% Implement the all-different constraint.
alldifferent(L) :- sort(L, Ls), length(Ls, LL), length(L, LL).

send_more_money([S,E,N,D,M,O,R,Y]) :-
(
            domain(S, between(2, 2)),
            domain(E, between(8, 8)),
            domain(N, between(1, 1)),
            domain(D, between(7, 9)),
            domain(M, between(0, 0)),
            domain(O, between(3, 3)),
            domain(R, between(6, 6)),
            domain(Y, between(5, 7)),
            constraint(                 1000 * S + 100 * E + 10 * N + D +
                                        1000 * M + 100 * O + 10 * R + E
                        =:= 10000 * M + 1000 * O + 100 * N + 10 * E + Y),
            constraint(alldifferent([S,E,N,D,M,O,R,Y]))
        ) => satisfy([S,E,N,D,M,O,R,Y]).



query(send_more_money(_)).