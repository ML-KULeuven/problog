%Expected outcome:  
% asthma(1) 0.19918213
% asthma(2) 0.4
% asthma(3) 0.176
% asthma(4) 0.19918213
% smokes(1) 0.49795533
% smokes(2) 1
% smokes(3) 0.44
% smokes(4) 0.49795533


0.3::stress(X) :- person(X).

0.2::influences(X,Y) :- person(X), person(Y).

smokes(X) :- stress(X).
smokes(X) :- friend(X,Y), influences(Y,X), smokes(Y).

0.4::asthma(X) :- smokes(X).

person(1).
person(2).
person(3).
person(4).

friend(1,2).
friend(2,1).
friend(2,4).
friend(3,2).
friend(4,2).


%%% Evidence
evidence(smokes(2),true).
%evidence(influences(4,2),false).

%%% Queries
% query(smokes(1)).



query(smokes(1)).
query(smokes(2)).
query(smokes(3)).
query(smokes(4)).
query(asthma(1)).
query(asthma(2)).
query(asthma(3)).
query(asthma(4)).


% query(stress(1)).
% query(stress(2)).
% query(stress(3)).
% query(stress(4)).

