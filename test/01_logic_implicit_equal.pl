%Expected outcome:
% q1 1

length(L,N) :- length3(L,0,N).
 
length3([],N,N).
length3([H|L],Acc,N) :- Acc2 is Acc + 1, length3(L,Acc2,N). 

% Redirect query to avoid ugly list notation in output which might be fixed in the future.
q1 :- length([a,b,c],3).

query( q1 ).