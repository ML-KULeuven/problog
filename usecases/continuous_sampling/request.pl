0.8::residential(1).
0.9::business(2).
has_mall(2).

area(X) :- between(1,2,X).
time(T) :- between(6,11,T).


morning(T) :- T > 6, T =< 10.
daytime(T) :- T > 9, T =< 20.

% Request reasons: clauses number to make unique.
poisson(30) :: 
req(1,A,B,T) :- 
    residential(A),
    business(B),
    morning(T).

poisson(10) :: 
req(2,_,B,T) :-
    has_mall(B),
    daytime(T).

% Sample all request reasons and combine the samples.
request_h(A,B,T,X)  :-
    findall( V, sample(req(I,A,B,T),V), L ),
    sumlist(L,X),
    X > 0.
    
% Transform combined samples back to distribution form.
constant(X) :: request(A,B,T) :- request_h(A,B,T,X).
    
sumlist(L,S) :- sumlist(L,0,S).
sumlist([],S,S).
sumlist([X|T],A,S) :- A1 is A + X, sumlist(T,A1,S).
    
query(request(A,B,T)) :- 
    area(A), 
    area(B), 
    time(T).

