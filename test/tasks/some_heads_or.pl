0.5::heads1.
0.6::heads2.
someHeads :- (heads1 ; heads2 ; \+ (\+heads1, \+heads2)), (heads2 ; heads1).

query(someHeads).