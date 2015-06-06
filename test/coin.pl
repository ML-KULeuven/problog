%Expected outcome:  
% someHeads 0.8
% twoHeads 0.3

0.5 :: heads1.
0.6 :: heads2.

twoHeads :- heads1, heads2.

someHeads :- heads1.
someHeads :- heads2.

query(someHeads).
query(twoHeads).