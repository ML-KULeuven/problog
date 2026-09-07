%Note: p(2) is false in every example, so nothing in the evidence says
%anything about the parameter of "q(2) :- p(2)". It is left at the value it
%was initialised with, which for a t(_, X) annotation is the random draw made
%under the seed this test fixes. It is deliberately not 0.0: see issue #98.
%Expected outcome:
% 0.8::p(0) :- between(0,2,0).
% 0.0::p(2) :- between(0,2,2).
% 0.2::p(1) :- between(0,2,1).
% 0.5::q(2) :- p(0).
% 1.0::q(2) :- p(1).
% 0.780718014018315::q(2) :- p(2).

t(_, X)::p(X):-between(0,2,X).
t(_, X)::q(2) :- p(X).