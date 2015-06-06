%System test 6 - a Hidden Markv Model.
%Description: An HMM which represents the state of the weather every day based on the previous observations. 
%Query: what is the probability of sunny weather in 10 days.
%Expected outcome: 
% weather(sun,100) 0.0

%% Make it fail, because NNF can't compute.
%% At time T=0
%0.5::weather(sun,0) ; 0.5::weather(rain,0) <- true.

%% Time T>0
0.6::weather(sun,T) ; 0.4::weather(rain,T) <- T>0, Tprev is T-1, weather(sun,Tprev).
0.2::weather(sun,T) ; 0.8::weather(rain,T) <- T>0, Tprev is T-1, weather(rain,Tprev).

%%% Queries
query(weather(sun,100)).
%query(weather(rain,0)).
