%Note: no example has both burglary and earthquake true, so the evidence says
%nothing about the parameter of the first alarm rule. It has to stay at the
%0.5 it was given rather than be driven to 0, which is what the equivalent
%model with the parameter on a separate fact does. See issue #98.
%Expected outcome:
% 0.333333333333333::burglary.
% 0.2::earthquake.
% 0.5::alarm :- burglary, earthquake.
% 1.0::alarm :- burglary, \+earthquake.
% 0.0::alarm :- \+burglary, earthquake.

t(0.5)::burglary.
0.2::earthquake.

t(0.5)::alarm :- burglary, earthquake.
t(0.5)::alarm :- burglary, \+earthquake.
t(0.5)::alarm :- \+burglary, earthquake.
