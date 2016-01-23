% Evaluate rules in-order until first one is satisfied.
%  evaluate_rules(+Callable, +ListOfIDs)
% Rule should be of the form 'predicate(ID, ...)'
% Predicate is of the form 'predicate(...)' (no ID)
cut(Predicate, List) :-
    cut(Predicate, List, _).

cut(Predicate, [X|T], ID) :-
    Predicate =.. [Pred|PArgs],     % split up predicate
    PredCall =.. [Pred, X | PArgs], % insert ID as first argument
    (
        call(PredCall),      % call rule instance and stop if it succeeds
        ID = X
    ;
        \+ call(PredCall),  % make sure call doesn't succeed
        cut(Predicate, T, ID)   % try another instance
    ).

on_fail(Call, _) :-
    call(Call).

on_fail(Call, OnFail) :-
    \+ call(Call), call(OnFail).
