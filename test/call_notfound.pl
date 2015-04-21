%Expected outcome:
% ERROR UnknownClause

%% Your program
x(X) :- help_w(template7,[a,5],1,X).
help_w(F2,In,N,X) :-
    append_vars_to_list(N,In,L),
    X=..[F2|L],
    call(X). 

append_vars_to_list(0,In,In).
append_vars_to_list(N,In,[_|Out]) :-
    N > 0,
    NN is N-1,
    append_vars_to_list(NN,In,Out).

template7(C,a1,M,C).
template7(C,a2,M,C).

query(x(_)).    