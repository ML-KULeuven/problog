%System test External functions:
%Description: Compute probability of a fact using an external (imperative) function
%Query: What is the probability that two strings are similar
%Expected outcome: 
% ERROR UnknownExternal

:- load_external('external.py').

targetstring("aa").

P::sim(X,Y) :- call_external(pysim(X,Y), P).
P::sim2(Y) :- targetstring(X), call_external(pysim1(X,Y), P).
%P::sim2(Y) :- call_external(pysim(X,Y), P), targetstring(X). % Not allowed because of non-ground

member(X,[X|_]).
member(X,[_|T]) :- member(X,T).

P::edge(X,Y) :- call_external(pyedge(X), L), member((Y,P), L).


query(sim("aa","aa")).
query(sim("aa","ab")).
query(sim("aa","bb")).
query(sim("aa","aab")).
query(sim("aa","abb")).

query(sim2("bb")).

query(edge("v1",Y)).
