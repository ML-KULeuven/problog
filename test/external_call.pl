%System test External functions:
%Description: Compute probability of a fact using an external (imperative) function
%Query: What is the probability that two strings are similar
%Expected outcome: 
% sim(""aa"",""aa"") 1.0
% sim(""aa"",""ab"") 0.5
% sim(""aa"",""bb"") 0.3333333
% sim(""aa"",""aab"") 0.5
% sim(""aa"",""abb"") 0.3333333
% sim2(""bb"") 0.3333333

load_external("external.py").

targetstring("aa").

P::sim(X,Y) :- call_external(pysim(X,Y), P).
P::sim2(Y) :- targetstring(X), call_external(pysim(X,Y), P).
%P::sim2(Y) :- call_external(pysim(X,Y), P), targetstring(X). % Not allowed because of non-ground

query(sim("aa","aa")).
query(sim("aa","ab")).
query(sim("aa","bb")).
query(sim("aa","aab")).
query(sim("aa","abb")).

query(sim2("bb")).

