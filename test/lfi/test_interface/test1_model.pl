%Expected outcome:
% 0.333333333333333::burglary.
% 0.2::earthquake.
% 0.553671728005259::p_alarm1.
% 1.0::p_alarm2.
% 0.0::p_alarm3.
% alarm :- burglary, earthquake, p_alarm1.
% alarm :- burglary, \\+earthquake, p_alarm2.
% alarm :- \\+burglary, earthquake, p_alarm3.


t(0.5)::burglary.
0.2::earthquake.
t(_)::p_alarm1.
t(_)::p_alarm2.
t(_)::p_alarm3.

alarm :- burglary, earthquake, p_alarm1.
alarm :- burglary, \+earthquake, p_alarm2.
alarm :- \+burglary, earthquake, p_alarm3.