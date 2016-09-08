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
    min_list(L,X,S).
min_list([X|L],M,S) :-
    M >= X,
    min_list(L,M,S).
    
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
    
nth0(0,[X|L],X).
nth0(I,[_|L],X) :-
    length(L, Len),
    between(1, Len, I),
    J is I - 1,
    nth0(J,L,X).
    
nth1(I,L,X) :- 
    J is I - 1,
    nth0(J,L,X).
    
last(List,Last) :- append(_, [Last], List).

reverse(L1,L2) :- reverse(L1,[],L2).
reverse([],L,L).
reverse([X|R],S,T) :-
    reverse(R,[X|S],T).

permutation([],[]).
permutation([X|R],[Y|S]) :-
    select(Y,[X|R],T),
    permutation(T,S).
    