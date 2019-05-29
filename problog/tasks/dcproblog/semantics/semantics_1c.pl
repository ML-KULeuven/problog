normal(0,1)~x:- a(Y).
a(Y):- c(X,Y).
c(1,2). c(1,3). c(2,4). c(2,3).
X>0::test:- x~=X.
query(test).
%result: 7/8

%creates three densities: for  a(2), a(3) and a(4).


% 1: atom(identifier=(3, (2,) {{}}, 0), probability=normal(0,1), group=(3, (2,) {{}}), name=choice(3,0,x,2), source=choice(3,0,x,2), atype='density')
% 2: atom(identifier=(3, (3,) {{}}, 0), probability=normal(0,1), group=(3, (3,) {{}}), name=choice(3,0,x,3), source=choice(3,0,x,3), atype='density')
% 3: atom(identifier=(3, (4,) {{}}, 0), probability=normal(0,1), group=(3, (4,) {{}}), name=choice(3,0,x,4), source=choice(3,0,x,4), atype='density')
% 4: atom(identifier=(20, ((x,0,0),) {{}}, 0), probability=(x,0,0)>0, group=(20, ((x,0,0),) {{}}), name=choice(20,0,test,(x,0,0)), source=choice(20,0,test,(x,0,0)), atype='bool')
% 5: atom(identifier=(20, ((x,1,0),) {{}}, 0), probability=(x,1,0)>0, group=(20, ((x,1,0),) {{}}), name=choice(20,0,test,(x,1,0)), source=choice(20,0,test,(x,1,0)), atype='bool')
% 6: atom(identifier=(20, ((x,2,0),) {{}}, 0), probability=(x,2,0)>0, group=(20, ((x,2,0),) {{}}), name=choice(20,0,test,(x,2,0)), source=choice(20,0,test,(x,2,0)), atype='bool')
% 7: disj(children=(4, 5, 6), name=test)
% Queries :
% * test : 7 [query]
