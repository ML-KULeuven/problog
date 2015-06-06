%System test 4 - a Bayesian network.
%Description: The burglary-alarm-earthquake BN without any people. The alarm goes off in three different cases.
%Query: what is the probability of burglary and what is probability of an earthquake
%Evidence: we know that the alarm has gone off.
%Expected outcome: 
% burglary 0.9896551724137932
% earthquake 0.2275862068965517
0.7::burglary.
0.2::earthquake.
0.9::p_alarm1.
0.8::p_alarm2.
0.1::p_alarm3.

alarm :- burglary, earthquake, p_alarm1.
alarm :- burglary, \+earthquake, p_alarm2.
alarm :- \+burglary, earthquake, p_alarm3.


%%% Evidence
evidence(alarm,true).

%%% Queries
query(burglary).
query(earthquake).

