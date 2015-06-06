%Expected outcome:
% bidding(i1,t1,[a1, a2, a3])  1
% bidding(i2,t2,[a1])  1
% bidding(i3,t3,[a2, a3])  1

%% Your program
% who is interested in what
interested(a1,i1).
interested(a1,i2).
interested(a2,i1).
interested(a2,i3).
interested(a3,i1).
interested(a3,i3).

% which item at which time
appears(i1,t1).
appears(i2,t2).
appears(i3,t3).


% deterministic bidding
bids(A,I,T) :- appears(I,T), interested(A,I).

bidding(I,T,As) :- findall(A,bids(A,I,T),As).

query(bidding(I,T,As)) :- appears(I,T).


