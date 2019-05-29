beta(0,1)~y.
beta(2,1)~y.
normal(Y,1)~x:- y~=Y, conS(Y>1/2).

X>0::test:- x~=X.
query(test).
%result: (d/dx)⁻¹[e^(-x²)](-⅟2^(3/2))·1/4·⅟√̅π+(d/dx)⁻¹[e^(-x²)](⅟2^(3/2))·⅟√̅π+-(d/dx)⁻¹[e^(-x²)](-⅟√̅2)·⅟√̅π+-(d/dx)⁻¹[e^(-x²)](⅟√̅2)·⅟√̅π+-1/4·⅟e^(1/8)·⅟√̅π·√̅2+3/4+⅟√̅2·⅟√̅e·⅟√̅π
%Same as 3a, only messier result because of hierarchy in densities



% 1: atom(identifier=3, probability=beta(0,1), group=None, name=y, source=y, atype='density')
% 2: atom(identifier=5, probability=beta(2,1), group=None, name=y, source=y, atype='density')
% 3: disj(children=(1, 2), name=None)
% 4: atom(identifier=8728380772342, probability=(y,0,0)>1/2, group=None, name=(y,0,0)>1/2, source=(y,0,0)>1/2, atype='bool')
% 5: atom(identifier=8728380772118, probability=(y,1,0)>1/2, group=None, name=(y,1,0)>1/2, source=(y,1,0)>1/2, atype='bool')
% 6: atom(identifier=(6, ((y,0,0),) {{}}, 0), probability=normal((y,0,0),1), group=(6, ((y,0,0),) {{}}), name=choice(6,0,x,(y,0,0)), source=choice(6,0,x,(y,0,0)), atype='density')
% 7: atom(identifier=(6, ((y,1,0),) {{}}, 0), probability=normal((y,1,0),1), group=(6, ((y,1,0),) {{}}), name=choice(6,0,x,(y,1,0)), source=choice(6,0,x,(y,1,0)), atype='density')
% 8: atom(identifier=(19, ((x,0,0),) {{}}, 0), probability=(x,0,0)>0, group=(19, ((x,0,0),) {{}}), name=choice(19,0,test,(x,0,0)), source=choice(19,0,test,(x,0,0)), atype='bool')
% 9: conj(children=(8, 4), name=None)
% 10: atom(identifier=(19, ((x,1,0),) {{}}, 0), probability=(x,1,0)>0, group=(19, ((x,1,0),) {{}}), name=choice(19,0,test,(x,1,0)), source=choice(19,0,test,(x,1,0)), atype='bool')
% 11: conj(children=(10, 5), name=None)
% 12: disj(children=(9, 11), name=test)
% Queries :
% * test : 12 [query]
