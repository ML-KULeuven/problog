%System test 1 - tossing coins:
%Description: two coins - one biased and one not. 
%Query: what is the probability of throwing two heads
%Expected outcome: 
% heads1 0.49999999999999994
% heads2 0.6
% twoHeads 0.30000000000000004
%%% Probabilistic facts: 
0.5::heads1.
0.6::heads2.

%%% Rules:
twoHeads :- heads1, heads2.

query(heads1).
query(heads2).
query(twoHeads).
