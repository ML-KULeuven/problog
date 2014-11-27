%System test 7 - a Probabilistic Graph.
%Description: A probabilistic graph of six nodes connected with 7 (prbabilistic) edges. 
%Query: what is the probability of existing a path from node 1 to node 6 and from node 1 to node 5.
%Expected outcome: 
% path(1,5) 0.25824
% path(1,6) 0.2167295999999999

0.6::edge(1,2).
0.1::edge(1,3).
0.4::edge(2,5).
0.3::edge(2,6).
0.3::edge(3,4).
0.8::edge(4,5).
0.2::edge(5,6).

path(X,Y) :- edge(X,Y).
path(X,Y) :- edge(X,Z),
             Y \== Z,
         path(Z,Y).


query(path(1,5)).
query(path(1,6)).
