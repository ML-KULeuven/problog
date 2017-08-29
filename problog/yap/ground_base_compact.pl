
get_id(Goal,I,ID) :-
    (ground_id(Goal,I,ID) ->
        true
    ;
        new_id(ID),
        assert(ground_id(Goal,I,ID) )
    ).

get_id(Goal,ID) :-
    (ground_id(Goal,ID) ->
        true
    ;
        new_id(ID),
        assert( ground_id(Goal,ID) )
    ).

new_id(ID) :- 
    catch(nb_getval(factid,ID),_,ID=1),
    ID1 is ID + 1,
    nb_setval(factid,ID1).

write_fact(P,I-Goal) :-
    recorded(gstream,S,_),
    get_id(Goal,I,ID),
    format(S,'~q FACT ~q | ad~q_~q\n',[ID,P,I,Goal]).
write_fact(P,Goal) :-
    Goal \= ad-_,
    get_id(Goal,ID),
    recorded(gstream,S,_),
    format(S,'~q FACT ~q | ~q\n',[ID,P,Goal]).
    
%%%%%
% Write clause
%%%%%
write_clause(Goal,true) :-
    !,\+certainly_true(Goal),
    recorded(gstream,S,_),
    get_id(Goal,ID),
    format(S,'~q FACT 1 | ~q\n',[Goal]).
write_clause(Goal,Body) :-
    recorded(gstream,S,_),
    remove_certainly_true(Body,PBody),
    get_id(Goal,ID),
    (PBody == true -> 
        format(S,'~q FACT 1 | ~q\n',[ID, Goal])
    ;
        format(S,'~q AND ',[ID]),
        write_body(PBody),
        format(S,' | ~q\n',[Goal])
    ).
        
write_body((Atom,Rest)) :-
    !,
    recorded(gstream,S,_),
    write_body(Atom),
    format(S,' ',[]),
    write_body(Rest).
write_body(\+ Atom) :-
    !,write_body(not(Atom)).

write_body(not(Atom)) :-
    !,
    recorded(gstream,S,_),
    (Atom = R-P-Fact ->
        ground_id( Fact, R, ID ),
        format(S,'-~q',[ID])
    ;
        ground_id( Atom, ID ),
        format(S,'-~q',[ID])
    ).
write_body(Atom) :-
    recorded(gstream,S,_),
    (Atom = R-P-Fact ->
        ground_id( Fact, R, ID ),
        format(S,'~q',[ID])
    ;
        ground_id( Atom, ID ), !,
        format(S,'~q',[ID])
    ).
