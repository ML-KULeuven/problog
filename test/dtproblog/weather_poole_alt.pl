% DTProbLog variant of example 9.11 in Poole's book
% http://artint.info/html/ArtInt_219.html

% chance node weather has no parents and takes values {rain, norain}
0.3::weather(rain); 0.7::weather(norain).

% chance node forecast has parent weather and takes values {sunny, cloudy, rainy}
0.7::forecast(sunny); 0.2::forecast(cloudy); 0.1::forecast(rainy) :- weather(norain).
0.15::forecast(sunny); 0.25::forecast(cloudy); 0.6::forecast(rainy) :- weather(rain).

% decision node umbrella has parent forecast and values {takeIt, leaveIt}
% so there is a separate decision to be made for each value of forecast
%
% thus, we'd want something like
%
% ?::umbrella(takeIt) ; ?::umbrella(leaveIt) :- forecast(_).
%
% as shorthand for
%
% ?::umbrella(takeIt) ; ?::umbrella(leaveIt) :- forecast(sunny).
% ?::umbrella(takeIt) ; ?::umbrella(leaveIt) :- forecast(cloudy).
% ?::umbrella(takeIt) ; ?::umbrella(leaveIt) :- forecast(rainy).
%
% and in a way that the strategy found by DTProbLog shows the context of each umbrella decision
%
% as these decisions are binary, we can unfold them into explicit decision facts in the current system, much as we can rewrite two-headed ADs:
%
umbrella(takeIt) :- forecast(X), decide_u(X).
umbrella(leaveIt) :- forecast(X), \+ decide_u(X).

?::decide_u(sunny).
?::decide_u(rainy).
?::decide_u(cloudy).

% the utility node has parents weather and umbrella, and its tabular definition has an entry for each parent value combination
line1 :- weather(norain), umbrella(takeIt).
line2 :- weather(norain), umbrella(leaveIt).
line3 :- weather(rain), umbrella(takeIt).
line4 :- weather(rain), umbrella(leaveIt).

utility(line1,20).
utility(line2,100).
utility(line3,70).
utility(line4,0).

