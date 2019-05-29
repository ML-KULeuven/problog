normal(0,1)~x:- a.
a:- c(X,Y).
c(1,2). c(1,3). c(2,4). c(2,3).
X>0::test:- x~=X.
query(test).
%result 1/2

%creates one density as X can only unify with one particular distribution given the body

% 1: atom(identifier=(3, () {{}}, 0), probability=normal(0,1), group=(3, () {{}}), name=choice(3,0,x), source=choice(3,0,x), atype='density')
% 2: atom(identifier=(20, ((x,0,0),) {{}}, 0), probability=(x,0,0)>0, group=(20, ((x,0,0),) {{}}), name=choice(20,0,test,(x,0,0)), source=test, atype='bool')
% Queries :
% * test : 2 [query]
