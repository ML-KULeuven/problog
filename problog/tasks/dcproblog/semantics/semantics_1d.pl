normal(0,1)~x :- c(X,Y).
c(1,2). c(1,3). c(2,4). c(2,3).
X>0::test:- x~=X.
query(test).
%result: 15/16

%creates four densities for each grounding of c(_,_).

% 1: atom(identifier=(3, (1, 2) {{}}, 0), probability=normal(0,1), group=(3, (1, 2) {{}}), name=choice(3,0,x,1,2), source=choice(3,0,x,1,2), atype='density')
% 2: atom(identifier=(3, (1, 3) {{}}, 0), probability=normal(0,1), group=(3, (1, 3) {{}}), name=choice(3,0,x,1,3), source=choice(3,0,x,1,3), atype='density')
% 3: atom(identifier=(3, (2, 4) {{}}, 0), probability=normal(0,1), group=(3, (2, 4) {{}}), name=choice(3,0,x,2,4), source=choice(3,0,x,2,4), atype='density')
% 4: atom(identifier=(3, (2, 3) {{}}, 0), probability=normal(0,1), group=(3, (2, 3) {{}}), name=choice(3,0,x,2,3), source=choice(3,0,x,2,3), atype='density')
% 5: atom(identifier=(17, ((x,0,0),) {{}}, 0), probability=(x,0,0)>0, group=(17, ((x,0,0),) {{}}), name=choice(17,0,test,(x,0,0)), source=choice(17,0,test,(x,0,0)), atype='bool')
% 6: atom(identifier=(17, ((x,1,0),) {{}}, 0), probability=(x,1,0)>0, group=(17, ((x,1,0),) {{}}), name=choice(17,0,test,(x,1,0)), source=choice(17,0,test,(x,1,0)), atype='bool')
% 7: atom(identifier=(17, ((x,2,0),) {{}}, 0), probability=(x,2,0)>0, group=(17, ((x,2,0),) {{}}), name=choice(17,0,test,(x,2,0)), source=choice(17,0,test,(x,2,0)), atype='bool')
% 8: atom(identifier=(17, ((x,3,0),) {{}}, 0), probability=(x,3,0)>0, group=(17, ((x,3,0),) {{}}), name=choice(17,0,test,(x,3,0)), source=choice(17,0,test,(x,3,0)), atype='bool')
% 9: disj(children=(8, 5, 6, 7), name=test)
% Queries :
% * test : 9 [query]
