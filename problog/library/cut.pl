:- module(cut, [cut/1, cut2]).

% Soft-cut.
%
%  Setup:
%   1. define a set of indexed-clauses (index is first argument)
%       r(1, a, b)
%       r(2, a, c)
%       r(3, b, c).
%   2. call the rule using cut where you should remove the first argument
%       cut(r(A, B))
%   => this will evaluate the rules in order of their index (note: NOT order in the file)
%       only ONE rule will match (the first one that succeeds)
%   e.g.:
%       cut(r(A, B)) => A = a, B = b
%       cut(r(a, X)) => X = b
%       cut(r(X, c)) => X = a
%       cut(r(b, X)) => X = c
%
%  The predicate cut/2 unifies the second argument with the Index of the matching rule.
%

cut(Call) :-
    Call =.. [Pred|Args],
    RCall =.. [Pred, Index | Args],
    all(Index, clause(RCall, _), List),
    sort(List, OList),
    cut(RCall, Index, OList, Call).

cut(Call, Index) :-
    Call =.. [Pred|Args],
    RCall =.. [Pred, Index | Args],
    all(Index, clause(RCall, _), List),
    sort(List, OList),
    cut(RCall, Index, OList, Call).


cut(RCall, Index, [Index | Rest], Call) :-
    call(RCall).
cut(RCall, Index, [Value | Rest], Call) :-
    \+ (Value = Index, call(RCall)),
    cut(RCall, Index, Rest, Call).