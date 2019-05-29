beta(0,1)~y.
beta(2,1)~y.
normal(0,1)~x:- y~=Y, conS(Y>1/2).

X>0::test:- x~=X.
query(test).
%result: 0.375

%create 2 densities for y and for each grounding of y a corresponding density for x.
%This results in the logic formual below

% 1: atom(identifier=3, probability=beta(0,1), group=None, name=y, source=y, atype='density')
% 2: atom(identifier=5, probability=beta(2,1), group=None, name=y, source=y, atype='density')
% 3: disj(children=(1, 2), name=None)
% 4: atom(identifier=8789526247183, probability=(y,0,0)>0.5, group=None, name=(y,0,0)>0.5, source=(y,0,0)>0.5, atype='bool')
% 5: atom(identifier=8789526272576, probability=(y,1,0)>0.5, group=None, name=(y,1,0)>0.5, source=(y,1,0)>0.5, atype='bool')
% 6: atom(identifier=(6, ((y,0,0),) {{}}, 0), probability=normal(0,1), group=(6, ((y,0,0),) {{}}), name=choice(6,0,x,(y,0,0)), source=choice(6,0,x,(y,0,0)), atype='density')
% 7: atom(identifier=(6, ((y,1,0),) {{}}, 0), probability=normal(0,1), group=(6, ((y,1,0),) {{}}), name=choice(6,0,x,(y,1,0)), source=choice(6,0,x,(y,1,0)), atype='density')
% 8: atom(identifier=(19, ((x,0,0),) {{}}, 0), probability=(x,0,0)>0, group=(19, ((x,0,0),) {{}}), name=choice(19,0,test,(x,0,0)), source=choice(19,0,test,(x,0,0)), atype='bool')
% 9: conj(children=(8, 4), name=None)
% 10: atom(identifier=(19, ((x,1,0),) {{}}, 0), probability=(x,1,0)>0, group=(19, ((x,1,0),) {{}}), name=choice(19,0,test,(x,1,0)), source=choice(19,0,test,(x,1,0)), atype='bool')
% 11: conj(children=(10, 5), name=None)
% 12: disj(children=(9, 11), name=test)
% Queries :
% * test : 12 [query]

%I think here we would like somehting else though! And only create one density for x?
