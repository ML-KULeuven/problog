%System test 12 - delayed AD grounding, two ways
%Expected outcome: 
% goes_to(alice,seaside,1) 0.37
% goes_to(alice,mountains,1) 0.315
% goes_to(alice,city,1) 0.315
% gt(alice,seaside,1) 0.37
% gt(alice,mountains,1) 0.315
% gt(alice,city,1) 0.315
person(alice).

destinations(seaside,mountains,city).
destinations(mountains,seaside,city).
destinations(city,seaside,mountains).

0.4::goes_to(P,seaside,0); 0.3::goes_to(P,mountains,0); 0.3::goes_to(P,city,0) <- person(P).

0.7::goes_to(X,D1,T); 0.15::goes_to(X,D2,T); 0.15::goes_to(X,D3,T) <- T>0, TPrev is T-1, destinations(D1,D2,D3), goes_to(X,D1,TPrev).

query(goes_to(alice,_,1)).

0.4::gt(P,seaside,0); 0.3::gt(P,mountains,0); 0.3::gt(P,city,0) <- person(P).

0.7::gt(X,D1,T); 0.15::gt(X,D2,T); 0.15::gt(X,D3,T) <- T>0, TPrev is T-1, gt(X,D1,TPrev), destinations(D1,D2,D3).

query(gt(alice,_,1)).
