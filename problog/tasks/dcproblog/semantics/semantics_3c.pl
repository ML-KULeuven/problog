beta(0,1)~y.
beta(2,1)~y.
normal(Y,1)~x:- y~=Y, a.
a:-  y~=Y, conS(Y>1/2).

X>0::test:- x~=X.
query(test).
%here the logic formula is a different one from the ones of 3a and 3b!


% 1: atom(identifier=3, probability=beta(0,1), group=None, name=y, source=y, atype='density')
% 2: atom(identifier=5, probability=beta(2,1), group=None, name=y, source=y, atype='density')
% 3: disj(children=(1, 2), name=None)
% 4: atom(identifier=-9223363249560800319, probability=(y,0,0)>1/2, group=None, name=(y,0,0)>1/2, source=(y,0,0)>1/2, atype='bool')
% 5: atom(identifier=8787293975360, probability=(y,1,0)>1/2, group=None, name=(y,1,0)>1/2, source=(y,1,0)>1/2, atype='bool')
% 6: disj(children=(4, 5), name=None)
% 7: atom(identifier=(6, ((y,0,0),) {{}}, 0), probability=normal((y,0,0),1), group=(6, ((y,0,0),) {{}}), name=choice(6,0,x,(y,0,0)), source=choice(6,0,x,(y,0,0)), atype='density')
% 8: atom(identifier=(6, ((y,1,0),) {{}}, 0), probability=normal((y,1,0),1), group=(6, ((y,1,0),) {{}}), name=choice(6,0,x,(y,1,0)), source=choice(6,0,x,(y,1,0)), atype='density')
% 9: atom(identifier=(24, ((x,0,0),) {{}}, 0), probability=(x,0,0)>0, group=(24, ((x,0,0),) {{}}), name=choice(24,0,test,(x,0,0)), source=choice(24,0,test,(x,0,0)), atype='bool')
% 10: conj(children=(9, 6), name=None)
% 11: atom(identifier=(24, ((x,1,0),) {{}}, 0), probability=(x,1,0)>0, group=(24, ((x,1,0),) {{}}), name=choice(24,0,test,(x,1,0)), source=choice(24,0,test,(x,1,0)), atype='bool')
% 12: conj(children=(11, 6), name=None)
% 13: disj(children=(10, 12), name=test)
% Queries :
% * test : 13 [query]
