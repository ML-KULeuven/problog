%System test 5 - a Bayesian network.
%Description: The burglary-alarm-earthquake BN with people - John and Mary. The alarm goes off in three different cases. The people can call either in case of alarm or just by accident.
%Query: what is the probability of burglary and what is probability of an earthquake
%Evidence: we know that John calls and Mary calls.
%Expected outcome: 
% burglary 0.9819392647842303
% earthquake 0.22685135855087904

person(john).
person(mary).

0.7::burglary.
0.2::earthquake.

0.9::alarm <- burglary, earthquake.
0.8::alarm <- burglary, \+earthquake.
0.1::alarm <- \+burglary, earthquake.

0.8::calls(X) <- alarm, person(X).
0.1::calls(X) <- \+alarm, person(X).


%%% Evidence
evidence(calls(john),true).
evidence(calls(mary),true).

%%% Queries
query(burglary).
query(earthquake).

