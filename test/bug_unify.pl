%Expected outcome:
% property_compress(content(last,[X5])) 1

question(prob,content(last,[_])).

property_origin(X) :-
     question(_,X).   % => THIS CALL

property_compress(Why) :-
     property_origin(content(last,List)),
     Why = content(last,List).

query(property_compress(_)).