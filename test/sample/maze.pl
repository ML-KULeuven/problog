:-use_module(library(lists)).

% Define maze size
maze(4,4).


%%% RULES OF ANY MAZE %%%

% Derive cells and possible connections from maze specification
cell(X,Y) :- maze(W,H),
             between(1,W,X),
             between(1,H,Y).
connectable(c(X1,Y1),c(X2,Y2)) :-
    cell(X1,Y1),
    cell(X2,Y2),
    Distance is (abs(X1-X2) + abs(Y1-Y2)),
    Distance == 1.

% Specify the chance of cells being connected
0.5:: connected(X,Y) :- connectable(X,Y).
connected(X,Y) :- connected(Y,X).

% Add connections from start to end: to enter the maze
connected(c(0,1),c(1,1)).
connected(c(W,H),c(X,H)) :-   maze(W,H),
							  RightW is W+1,
							  X = RightW.

% Predicate to state that there is a path between certain cells
exists_path(X,Y) :- connected(X,Y).
exists_path(X,Y) :- connected(X,Z),
                    exists_path(Z,Y).

%%% PROPERTIES %%%

%% SOLVABILITY: State that the maze must be solvable (= there is a path between upperleft and bottomright)
solvable_maze :- maze(W,H),
                 exists_path(c(1,1),c(W,H)).

%% CONNECTEDNESS: Define the concept of a fully connected maze (= every cell is connected to the start cell )
unconnected_maze :- cell(X,Y),
					\+ exists_path(c(1,1), c(X,Y)).
fully_connected_maze :- \+ unconnected_maze.

%% SINGLE SOLUTION:
path(A,B,Path) :-
       travel(A,B,[A],Q),
       reverse(Q,Path).
travel(A,A,P,P).
travel(A,B,P,[B|P]) :-
       connected(A,B).
travel(A,B,Visited,Path) :-
       connected(A,C),
       C \== B,
       \+ member(C,Visited),
       travel(C,B,[C|Visited],Path).

multiple_paths :-
          cell(X1,Y1),
          cell(X2,Y2),
          (X1 \== X2 ; Y1 \== Y2 ),
          path(c(X1,Y1),c(X2,Y2), Path1),
				  path(c(X1,Y1),c(X2,Y2), Path2),
          %% writenl( - ),
          %% writenl(1:Path1),
          %% writenl(2:Path2),
				  Path1 \== Path2.
          %% writenl(dissimilar).
no_cycles :- \+ multiple_paths.


%% ENFORCED PROPERTIES: Give necessary properties of the maze to generate
evidence(solvable_maze).
evidence(fully_connected_maze).
%evidence(no_cycles).

% Find all connections when sampling
query(connected(_,_)).
query(path(c(1,1),c(1,2), _)).