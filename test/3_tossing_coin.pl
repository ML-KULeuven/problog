%System test 3 - tossing coins:
%Description: A general rule (written as intentional probabilistic facts) determines the probability of the outcome of a tossed coin to be heads as 0.6. There are four coins.
%Query: what is the probability of throwing some heads
%Expected outcome: 
% someHeads 0.9744
%%% Probabilistic facts:
0.6::heads(C) :- coin(C).

%%% Background information:
coin(c1).
coin(c2).
coin(c3).
coin(c4).

%%% Rules:
someHeads :- heads(_).

query(someHeads).
