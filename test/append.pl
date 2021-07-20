%Expected outcome:
% append([[1], [2, 3], [4]],[1, 2, 3, 4]) 1
% append([[], [1, 2, 3]],[1, 2, 3]) 1
% append([[1], [2, 3]],[1, 2, 3]) 1
% append([[1, 2], [3]],[1, 2, 3]) 1
% append([[1, 2, 3], []],[1, 2, 3]) 1
:-use_module(library(lists)).

query(append([[1],[2,3],[4]],[1,2,3,4])).
query(append([_,_], [1,2,3])).