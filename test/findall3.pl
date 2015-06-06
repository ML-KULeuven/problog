%Expected outcome:
% 	      bidding(i2,2,[])  0.8
%     bidding(i2,2,[a1])  0.2
%       bidding(i3,3,[])  0.7
% bidding(i3,3,[a2, a3])  0.3

%% Your program
% who is interested in what
interested(a1,i1).
interested(a1,i2).
interested(a2,i1).
interested(a2,i3).
interested(a3,i1).
interested(a3,i3).

% which item at which time
%0.1::appears(i1,1).
0.2::appears(i2,2).
0.3::appears(i3,3).


% deterministic bidding
bids(A,I,T) :- appears(I,T), interested(A,I).

bidding(I,T,As) :- findall(A,bids(A,I,T),As).

query(bidding(I,T,As)) :- appears(I,T).


