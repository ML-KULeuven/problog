%Expected outcome:
% 0.6::wind.
% day(d1).
% day(d2).
% 0.7::umbrella(d1) :- day(d1), \+wind.
% 0.3::umbrella(d2) :- day(d2), \+wind.

0.6::wind.
day(d1). day(d2).

t(_,D)::umbrella(D) :- day(D), \+wind.
