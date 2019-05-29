normal(0,1)~x:- a(X).
a(X):- c(X,Y).
c(1,2). c(1,3). c(2,4). c(2,3).
X>0::test:- x~=X.
query(test).
%result: 3/4

%creates two densities: for  a(1) and a(2). These random variabels (different x)
%are again independent of each other given a(1), a(2) respectively


% 1: atom(identifier=(3, (1,) {{}}, 0), probability=normal(0,1), group=(3, (1,) {{}}), name=choice(3,0,x,1), source=choice(3,0,x,1), atype='density')
% 2: atom(identifier=(3, (2,) {{}}, 0), probability=normal(0,1), group=(3, (2,) {{}}), name=choice(3,0,x,2), source=choice(3,0,x,2), atype='density')
% 3: atom(identifier=(20, ((x,0,0),) {{}}, 0), probability=(x,0,0)>0, group=(20, ((x,0,0),) {{}}), name=choice(20,0,test,(x,0,0)), source=choice(20,0,test,(x,0,0)), atype='bool')
% 4: atom(identifier=(20, ((x,1,0),) {{}}, 0), probability=(x,1,0)>0, group=(20, ((x,1,0),) {{}}), name=choice(20,0,test,(x,1,0)), source=choice(20,0,test,(x,1,0)), atype='bool')
% 5: disj(children=(3, 4), name=test)
% Queries :
% * test : 5 [query]
