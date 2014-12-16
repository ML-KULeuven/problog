%System test 2 - tossing coins:
%Description: two coins - one biased and one not. 
%Query: what is the probability of throwing some heads
%Expected outcome: 
% heads1 0.5
% heads2 0.6
% someHeads 0.8
%%% Probabilistic facts:
0.5::heads1.
0.6::heads2.

%%% Rules:
someHeads :- heads1.
someHeads :- heads2.

query(heads1).
query(heads2).
query(someHeads).
