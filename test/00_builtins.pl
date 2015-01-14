% Test true/0
q_001 :- true.
query(q_001). % outcome: 1

q_002 :- \+true.
query(q_002). % outcome: 0

% Test fail/0 false/0
q_003 :- fail.
query(q_003). % outcome: 0

q_004 :- false.
query(q_004). % outcome: 0

q_005 :- \+fail.
query(q_005). % outcome: 1

% Test =/2
q1 :- A = 1.
query(q1). % outcome: 1

% Test ==/2
q2 :- 1 == 1.
query(q2). % outcome: 1

q3 :- 1 \== 1.
query(q3). % outcome: 0


%Expected outcome:
% between_000(1) 1
% between_000(2) 1
% between_000(3) 1
between_000(X) :- between(1,3,X).
query(between_000(X)).

between_001 :- between(1,3,3).
query(between_001). % outcome: 1

between_002 :- between(1,3,4).
query(between_002). % outcome: 0

between_003 :- between(-3,2,0).
query(between_003). % outcome: 1

%Expected outcome:
% between_004(-3) 1
% between_004(-2) 1
% between_004(-1) 1
% between_004(0) 1
% between_004(1) 1
% between_004(2) 1
between_004(X) :- between(-3,2,X).
query(between_004(X)). 
