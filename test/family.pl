parent(maria, erik).
parent(erik,katrien).
parent(katrien,liese).


anc1(X,Y) :- parent(X,Y).
anc1(X,Z) :- parent(X,Y),anc1(Y,Z). 

anc2(X,Z) :- parent(X,Y),anc2(Y,Z). 
anc2(X,Y) :- parent(X,Y).

%:- table anc3/2.
anc3(X,Z) :- anc3(Y,Z), parent(X,Y). 
anc3(X,Y) :- parent(X,Y).

%:- table anc4/2.
anc4(X,Z) :- anc4(Y,Z), parent(X,Y). 
anc4(X,Y) :- parent(X,Y).


%:- table anc5/2.
anc5(X,Z) :- anc4(X,Y), parent(Y,Z). 
anc5(X,Y) :- parent(X,Y).


%:- table anc6/2.
anc6(X,Z) :- anc6(X,Y), parent(Y,Z). 
anc6(X,Y) :- parent(X,Y).

anc6(X,Y) :- anc6(X,Z), parent(Z,Y).
anc6(X,Y) :- parent(X,Y).

% anc6(X,Y) :- parent(X,Y).
% anc6(X,Y) :- parent(X,Y), parent(Z,Y).
% anc6(X,Y) :- anc6(X,Z), parent(Z,Y).



% def anc6(X,Y)
%     clause anc(X,Z)
%         -> and
%             -> call anc6(X,Y)
%                 -> def anc6(X,Y)
%                     -> clause anc6(X,Z)
%                         -> and
%                             -> call anc6(X,Y)   => CYCLE
%                     -> clause anc6(X,Y)
%                         -> parent(X,Y)
%                         <- parent(erik, katrien)
%                     <- clause anc6(erik,katrien)
%                 <- def anc6(erik,katrien)
%             <- anc6(erik,katrien)
%             -> parent(katrien,Z)
%             <- parent(katrien,liese)
%         <- and
%     <- anc(katrien,liese)
