%Expected outcome:
% 0.666666666666667::burglary.
% 0.333333333333333::earthquake.
% 0.0::al.
% alarm :- burglary.
% alarm :- earthquake.
% calls :- alarm, al.


t(0.5)::burglary.
t(0.5)::earthquake.
t(0.5)::al.

alarm :- burglary.
alarm :- earthquake.
calls :- alarm, al.


