%System test 3 - tossing coins:
%Test ADs with only one atom (aka groups with 1 choice).
%
%Expected outcome:
%Factor (c0) = 0, 1
%(): [0.4, 0.6]
%
%OrCPT someHeads [0,1] -- c0
%('c0', 1)

%%% Probabilistic facts:
0.6::heads(C) :- coin(C).

%%% Background information:
coin(c1).

%%% Rules:
someHeads :- heads(_).

query(someHeads).