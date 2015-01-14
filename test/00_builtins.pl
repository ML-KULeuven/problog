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
