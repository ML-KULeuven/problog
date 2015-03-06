%System test 8 - a social network
%Description: A network of people (friends) some of which smoken due to different reasons. Somking then can cause asthma with a common probability for every person.
%Query: what is the probability that person 1 smokes; what is the probability that person 3 smokes; what is the probability that person 4 smokes; what is the probability that person 2 has asthma; what is the probability that person 3 has asthma; what is the probability that person 4 has asthma.
%Expected outcome:  
% smokes(1) 0.5087719298245614
% smokes(2) 1.0
% smokes(3) 0.44000000000000006
% smokes(4) 0.44000000000000006
% asthma(1) 0.20350877192982458
% asthma(2) 0.4000000000000001
% asthma(3) 0.176
% asthma(4) 0.176

0.3::stress(X) :- person(X).
0.2::influences(X,Y) :- person(X), person(Y).

smokes(X) :- stress(X) ; friend(X,Y), influences(Y,X), smokes(Y).

0.4::asthma(X) <- smokes(X).

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
evidence(smokes(2)).
evidence(\+influences(4,2)).

%%% Queries
query(smokes(1)).
query(smokes(2)).
query(smokes(3)).
query(smokes(4)).
query(asthma(1)).
query(asthma(2)).
query(asthma(3)).
query(asthma(4)).
