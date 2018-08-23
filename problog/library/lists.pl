% member(X,L)
%  X is an element from list L
memberchk(X,[X|_]).
memberchk(X,[Y|T]) :- X \= Y, memberchk(X,T).


% member(X,L)
%  X is an element from list L
member(X,[X|_]).
member(X,[_|T]) :- member(X,T).

% select(X,L,R)
%  X is element from list L
%  R is remaining list. 
select(X,[X|T],T).
select(X,[Y|T],[Y|S]) :-
    select(X,T,S).

selectchk(X,[X|T],T).
selectchk(X,[Y|T],[Y|S]) :-
    X \= Y,
    selectchk(X,T,S).

% select_uniform(ID,L,X,R)
%  select a single element probabilistically
%  
select_uniform(ID, Values, Value, Rest) :-
    length(Values, Len),
    Len > 0,
    Weight is 1/Len,
    make_list(Len, Weight, Weights),
    select_weighted(ID, Weights, Values, Value, Rest).

select_weighted(ID, Weights, Values, Value, Rest) :-
    sum_list(Weights,Total),
    Total > 0,
    sw(ID, Total, Weights, Values, Value, Rest).
select_weighted(ID, WeightsValues, Value, Rest) :-
    unzip(WeightsValues,Weights,Values),
    select_weighted(ID, Weights, Values, Value, Rest).

P::sw_p(ID,P,_,_,_).
% Last is always selected to avoid numerical instabilities.
sw(ID,PW,[W|WT],[X|[]],X,[]).
sw(ID,PW,[W|WT],[X|XT],X,XT) :-
    XT \= [],
    W1 is W/PW,
    sw_p(ID,W1,WT,X,XT).
sw(ID,PW,[W|WT],[X|XT],Y,[X|RT]) :-
    XT \= [],
    W1 is W/PW,
    not sw_p(ID,W1,WT,X,XT),
    PW1 is PW-W,
    sw(ID,PW1,WT,XT,Y,RT).

% S is sum of elements of list L
sum_list(L,S) :- sum_list(L,0,S).
sum_list([],S,S).
sum_list([X|T],A,S) :-
    B is A+X,
    sum_list(T,B,S).
    
max_list([X|L],S) :- max_list(L,X,S).
max_list([],S,S).
max_list([X|L],M,S) :-
    M > X,
    max_list(L,M,S).
max_list([X|L],M,S) :-
    M =< X,
    max_list(L,X,S).

min_list([X|L],S) :- min_list(L,X,S).
min_list([],S,S).
min_list([X|L],M,S) :-
    M < X,
    min_list(L,M,S).
min_list([X|L],M,S) :-
    M >= X,
    min_list(L,X,S).

max_member([X|L],S) :- max_member(L,X,S).
max_member([],S,S).
max_member([X|L],M,S) :-
    M @> X,
    max_member(L,M,S).
max_member([X|L],M,S) :-
    M @=< X,
    max_member(L,X,S).

min_member([X|L],S) :- min_member(L,X,S).
min_member([],S,S).
min_member([X|L],M,S) :-
    M @< X,
    min_member(L,X,S).
min_member([X|L],M,S) :-
    M @>= X,
    min_member(L,M,S).

numlist(Low, Low, [Low]).
numlist(Low, High, [Low|Rest]) :-
    Low < High,
    Low1 is Low + 1,
    numlist(Low1, High, Rest).

    
unzip([],[],[]).
unzip([(X,Y)|T],[X|R],[Y|S]) :-
    unzip(T,R,S).
    
zip(A,B,AB) :- unzip(AB,A,B).
    
% make_list(X,N,L)
%  make a list of length N filled with element X
make_list(0,X,[]).
make_list(Len,X,[X|L]) :-
    Len > 0,
    Len1 is Len-1,
    make_list(Len1,X,L).
    
append([],T,T).
append([X|R],S,[X|T]) :- append(R,S,T).

append(LoL,L) :- append2(LoL,[],L).
    
append2([],Acc,Acc).
append2([L|T],Acc,Out) :-
    append(Acc,L,Acc1),
    append2(T,Acc1,Out).
    
prefix(A,B) :- append(A,_,B).

select(X,[X|_],Y,[Y|_]).
select(X,[_|XList],Y,[_|YList]) :-
    select(X,XList,Y,YList).

selectchk(X,[X|_],Y,[Y|_]).
selectchk(X,[X1|XList],Y,[Y1|YList]) :-
    \+ (X1 = X, Y1 = Y),
    selectchk(X,XList,Y,YList).


nth0(0,[X|L],X,L).
nth0(I,[Y|L],X,[Y|R]) :-
    length(L, Len),
    between(1, Len, I),
    J is I - 1,
    nth0(J,L,X,R).

nth0(I, L, X) :- nth0(I, L, X,_).
nth1(I, L, X) :- nth1(I, L, X,_).
    
nth1(I,L,X,R) :-
    J is I - 1,
    nth0(J,L,X,R).
    
last(List,Last) :- append(_, [Last], List).
head([H|_], H).

reverse(L1,L2) :- reverse(L1,[],L2).
reverse([],L,L).
reverse([X|R],S,T) :-
    reverse(R,[X|S],T).

permutation([],[]).
permutation([X|R],[Y|S]) :-
    select(Y,[X|R],T),
    permutation(T,S).


% Flatten a list (e.g. [1, [2, [3, 4], [5, 6], 7]] => [1,2,3,4,5,6,7]
flatten(In, Out) :- flatten(In, [], Out).

flatten([], Acc, Acc).
flatten([H|T], Acc, List) :-
    flatten(H, Acc, Acc1),
    flatten(T, Acc1, List).
flatten(H, Acc, List) :-
    \+ is_list(H),
    append(Acc, [H], List).

is_set(List) :-
    sort(List, Set),
    same_length(List, Set).


groupby([], []).
groupby([[G|X]|T], Out) :-
    groupby(T, G, [X], Out).

% groupby(ListIn, CurEl, CurGroup, Out)
groupby([], G, S, [[G,S]]).

groupby([[G|X]|T], G, S,  Out) :-
    groupby(T, G, [X|S], Out).

groupby([[G|X]|T], G1, S,  [[G1,S]|Out]) :-
    G \= G1,
    groupby(T, G, [X], Out).


sub_list(List, Before, Length, After, SubList) :-
    sub_list(List, Before, 0, Length, After, SubList).

sub_list([X|List], [X|Before], 0, Length, After, SubList) :-
    sub_list(List, Before, 0, Length, After, SubList).

sub_list(List, [], Length, Length, List, []).

sub_list([X|List], [], AccLen, Length, After, [X|SubList]) :-
    TmpLen is AccLen + 1,
    sub_list(List, [], TmpLen, Length, After, SubList).

proper_length(List, Length) :-
    is_list(List),
    length(List, Length).

same_length(List1, List2) :-
    length(List1, L),
    length(List2, L).


intersection([], _, []).
intersection([X|T], Set2, [X|R]) :-
    memberchk(X, Set2),
    intersection(T, Set2, R).
intersection([X|T], Set2, R) :-
    \+ memberchk(X, Set2),
    intersection(T, Set2, R).

union([], Set2, Set2).
union([X|T], Set2, R) :-
    memberchk(X, Set2),
    union(T, Set2, R).
union([X|T], Set2, [X|R]) :-
    \+ memberchk(X, Set2),
    union(T, Set2, R).

subset([], _).
subset([X|T], Set) :-
    memberchk(X, Set),
    subset(T, Set).

subtract([], _, []).
subtract([X|T], Delete, R) :-
    memberchk(X, Delete),
    subtract(T, Delete, R).
subtract([X|T], Delete, [X|R]) :-
    \+memberchk(X, Delete),
    subtract(T, Delete, R).

delete([], _, []).
delete([H|T], Elem, [H|R]) :-
    Elem \= H,
    delete(T, R).
delete([H|T], Elem, R) :-
    \+ Elem \= H,
    delete(T, R).

nextto(X, Y, [X, Y|_]).
nextto(X, Y, [_|T]) :-
   nextto(X, Y, T).

:- use_module(library('lists.py')).

list_to_set(List, Set) :-
    length(List, Len),                % Determine the length of the input list.
    numlist(1, Len, Index),           % Create a list of numbers 1..length.
    zip(List, Index, IndexedList),    % Make tuples of (value, index)
    all((Position, Value), (
        enum_groups(IndexedList, Value, Positions), % Find positions for same value.
        min_list(Positions, Position)               % Take minimal position for each value.
    ), MinList),                      % Create a list of (min position, value) for unique values.
    sort(MinList, SortedMinList),     % Sort the list such that first occurring values come first.
    unzip(SortedMinList, _, Set).     % Remove the index information from the list.




