%Expected outcome:
% q([60-a1, 240-a2, 60-b2]) 1


has_property(b1,x).
q(Z) :- findall(C-V,(member(C-V,[60-a1,240-a2,40-b1,60-b2]),\+ has_property(V,x)),Z).
query(q(_)).
member(A,[A|_]).
member(A,[_|L]) :- member(A,L).