%Expected outcome:  
% fill([unknown, unknown])  0.08 

0.1::p1.
0.8::p2.

p([A,B]) :- p1, A = unknown.
p([A,B]) :- p2, B = unknown.

fill([A,B]) :- p([A,B]), A == unknown, B == unknown.
fill([A,B]) :- p([A,B]), fill([A,B]).

query(fill([A,B])).

