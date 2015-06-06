%System test 9 - the packing problem
%Description: We want to pack our bag for a trip, but we cannot exceed some limmits. We want to know whether we may exceed the weight limit at the airport and with what probability.
%Query: what is the probability that we pack number of objects which exceed the weight limit (weight of 6) for our bag.
%Expected outcome:  
% excess(8) 0.11805555555555566

weight(skis,6).
weight(boots,4).
weight(helmet,3).
weight(gloves,2).

% intensional probabilistic fact with flexible probability:
P::pack(Item) :- weight(Item,Weight),  P is 1.0/Weight.

excess(Limit) :- excess([skis,boots,helmet,gloves],Limit). % all possible items
excess([],Limit) :- Limit<0.
excess([I|R],Limit) :- pack(I), weight(I,W), L is Limit-W, excess(R,L).
excess([I|R],Limit) :- \+pack(I), excess(R,Limit).


%%% Queries
query(excess(8)).
