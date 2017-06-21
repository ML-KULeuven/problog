% AUTHOR: Guy Van den Broeck <guy.vandenbroeck@cs.kuleuven.be>
% AUTHOR: Joris Renkens <joris.renkens@cs.kuleuven.be>

:- use_module(library(lists)).

:- op(900,xfx,['::']). % to support probabilistic facts
:- op(1149,xfx,['<-']). % to support annotated disjunctions

:- table possibly_true/1.
:- table certainly_true/1.
:- table write_fact/2.
:- table write_clause/2.

evidence(_, _) :- fail.
query(_) :- fail.
problog_ad(_, _) :- fail.
ground_id(_, _) :- fail.
ground_id(_, _, _) :- fail.

:- yap_flag(unknown, fail).

%%%%%%%%%%%%%%%%%%%%%%
% Used for grounding a ProbLog program. Use: yap -q -l ground.pl -g "main('input','grounding_output','evidence_output','queries_output')" 
% - Returns 0 on succes
% - Returns 1 when encountered illegal evidence
% - Returns 2 when encountered illegal query
% - Returns 3 when encountered illegal probabilistic fact
% - Returns 4 when encountered an error during grounding
%%%%%%%%%%%%%%%%%%%%%%

main :- 
    current_prolog_flag(argv, ARGS),
    (ARGS = [Input,Grounding,Evidence,Queries] ->
        main(Input, Grounding, Evidence, Queries)
    ;
        halt(5)
    ).

main(Input, Grounding, Evidence, Queries) :- 
    init(Input, Grounding, Evidence, Queries),
    catch(write_evidence,illegal_evidence,cleanup(0)),
    catch(write_queries,illegal_query,cleanup(0)),
    catch(catch(catch(write_grounding,illegal_fact,cleanup(0)),grounding_error,cleanup(0)),builtin_unsupported,cleanup(0)),
    cleanup(0).

init(Input, Grounding, Evidence, Queries) :-
    abolish_all_tables,
    eraseall(gstream),
    eraseall(estream),
    eraseall(qstream),
    open(Grounding,'write',G),
    recorda(gstream,G,_),
    open(Evidence,'write',E),
    recorda(estream,E,_),
    open(Queries,'write',Q),
    recorda(qstream,Q,_),
    source,
    consult(Input).

cleanup(Message) :-
    %abolish_all_tables,
    flush_output(user_error),
	 flush_output(user_output),
    recorded(gstream,GStream,GRef),
    flush_output(GStream),
    close(GStream),
    erase(GRef),
    recorded(estream,EStream,ERef),
    flush_output(EStream),
    close(EStream),
    erase(ERef),
    recorded(qstream,QStream,QRef),
    flush_output(QStream),
    close(QStream),
    erase(QRef),
    halt(Message).

%%%%%%%
% Writing evidence
%%%%%%%

write_evidence :-
    recorded(estream,Stream,_),
    evidence(Atom,Truth),
    valid_evidence(Atom,Truth),
    (Truth = true ->
        format(Stream,'~q ~q\n',[Atom,'t'])
    ;
        format(Stream,'~q ~q\n',[Atom,'f'])
    ),
    fail.
write_evidence.

valid_evidence(Atom,_) :- 
    \+ ground(Atom),
    format(user_error,'ERROR: The evidence atom ~q is non ground. Evidence atoms should be grounded in the body.\n',[Atom]),
    throw(illegal_evidence).
valid_evidence(Atom,Truth) :-
    (\+ ground(Truth) ; Truth \= true , Truth \= false),
    format(user_error,'ERROR: The truth value for the evidence atom ~q is ~q. The truth value should be equal to true or false.\n',[Atom,Truth]),
    throw(illegal_evidence).
valid_evidence(_,_).

%%%%%%
% Writing queries
%%%%%%

write_queries :-
    recorded(qstream,Stream,_),
    query(Atom),
%    valid_query(Atom),
    (ground(Atom) ->
		format(Stream,'~q\n', Atom)
	 ;
		findall( Atom, possibly_true(Atom), Grounded ),
		format_all(Stream,'~q\n', Grounded)
	 ),
    fail.
write_queries.

valid_query(Atom) :- 
    \+ ground(Atom),
    format(user_error,'ERROR: The query ~q is non ground. Queries should be grounded in the body.\n',[Atom]),
    throw(illegal_query).
valid_query(_).

format_all(Stream, Format, []) :- !.
format_all(Stream, Format, [H|T]) :-
	format(Stream, Format, [H]),
	format_all(Stream, Format, T).

%%%%%%
% Writing grounding
%%%%%%
write_grounding :-
      query(Goal),
      possibly_true(Goal),
		(certainly_true(Goal) ->
			write_fact(1.0,Goal)
		;
			true
		),
      fail.
write_grounding :-
    	evidence(Goal,_),
    	possibly_true(Goal),
      fail.
write_grounding.

%%%%%%
% Possibly true
%%%%%%

possibly_true(Goal) :-
    certainly_true(Goal).
possibly_true(\+ Goal) :-
    \+ certainly_true(Goal).
possibly_true(not(Goal)) :-
    \+ certainly_true(Goal).
possibly_true(R-P-Fact) :-
    write_fact(P,R-Fact).
possibly_true(Goal) :-
    clause((P::Goal),Body),    
    call(Body),
    valid_probfact(Goal,P),
    write_fact(P,Goal).
possibly_true(Goal) :-
	(	
	% findall((AD,Body), clause((AD <- Body), true), ALL),
    findall((AD,Body), problog_ad(AD, Body), ALL),
	nth1(I,ALL,(AD,Body)),
    % clause((AD <- Body),true,R),
    % nth_clause(_,I,R),
    ad_converter(Goal,AD,Body,I,Sum,ExtendedBody),
    (Sum>1.00000000001 ->
        format(user_error,'ERROR: Sum of annotated disjunction is larger than 1.0.~n~w <- ~w.~n',[AD, Body]),
        fail
        ;
        true
    ),
    possibly_true(ExtendedBody),
    write_clause(Goal,ExtendedBody)
	;
    catch(clause(Goal,Body),_,fail),
    possibly_true(Body),
    valid_for_grounding(Goal),
    write_clause(Goal,Body)
	).
possibly_true((Goal1,Goal2)) :-
    possibly_true(Goal1),
    possibly_true(Goal2).

ad_converter(Head,AD,Body,I,Sum,Result) :-
    ad_converter(Head,AD,Body,I,1,Sum,Result).
ad_converter(Head,[P::Head],Body,I,Dev,P,Result) :-
    Pnew is P / Dev,
    Result = (Body,I-Pnew-Head).
ad_converter(Head,([P::Head|_]),Body,I,Dev,P,Result) :-
    Pnew is P / Dev,
    Result = (Body,I-Pnew-Head).
ad_converter(Head,([P::Fact|Rec]),Body,I,Dev,NSum,Result) :-
    Pnew is P / Dev,
    DevRec is Dev - P,
    ad_converter(Head,Rec,Body,I,DevRec,Sum,ResultRec),
    NSum is Sum + P,
    Result = (ResultRec,\+ I-Pnew-Fact).

%%%%%%
% Certainly true
%%%%%%
certainly_true(\+ Goal) :-
    (
        % To backtrack over all possible ways it can be true
        possibly_true(Goal),
        fail
    ;
        \+ possibly_true(Goal)
    ).
certainly_true(not(Goal)) :-
    (
        % To backtrack over all possible ways it can be true
        possibly_true(Goal),
        fail
    ;
        \+ possibly_true(Goal)
    ).
certainly_true((Goal1,Goal2)) :- 
    certainly_true(Goal1),
    certainly_true(Goal2).
certainly_true((Goal1;Goal2)) :- %to handle disjunction in bodies
    certainly_true(Goal1)
    ;
    certainly_true(Goal2).
certainly_true(Goal) :-
	 \+ builtin_reused(Goal),
    predicate_property(Goal,built_in),
    (builtin_support(Goal) ->
    	call(Goal)
	 ;
    	format(user_error,'ERROR: The built in ~q is not supported by the system.\n',[Goal]),
    	throw(builtin_unsupported)
	 ).
certainly_true(Goal) :-
    catch(clause(Goal,Body),_,fail),
    certainly_true(Body),
    valid_for_grounding(Goal).


%%%%%%
% Valid probabilistic fact
%%%%%%
valid_probfact(Atom,_) :-
    \+ ground(Atom),
    format(user_error,'ERROR: The probabilistic fact ~q is non ground. Probabilistic facts should be grounded after calling.\n',[Atom]),
    throw(illegal_fact).
valid_probfact(Atom,Prob) :-
    \+ number(Prob),
    format(user_error,'ERROR: The probability for the probabilistic fact ~q is ~q. Probabilities should be a float between 0 and 1.\n',[Atom,Prob]),
    throw(illegal_fact).
valid_probfact(Atom,Prob) :-
    Prob < 0,
    format(user_error,'ERROR: The probability for the probabilistic fact ~q is ~q. Probabilities should not be lower than 0.\n',[Atom,Prob]),
    throw(illegal_fact).
valid_probfact(Atom,Prob) :-
    Prob > 1,
    format(user_error,'ERROR: The probability for the probabilistic fact ~q is ~q. Probabilities should not be greater than 1.\n',[Atom,Prob]),
    throw(illegal_fact).
valid_probfact(_,_).

%%%%%%
% Valid atom during grounding
%%%%%%
valid_for_grounding(Atom) :-
    \+ground(Atom),
    format(user_error,'ERROR: encountered an ungrounded atom ~q during grounding. Heads of rules should be ground after calling the rule.\n',[Atom]),
    throw(grounding_error).
valid_for_grounding(_).

%%%%%
% Write fact
%%%%%
write_fact(P,I-Goal) :-
    recorded(gstream,S,_),
    format(S,'~q::ad~q_~q.\n',[P,I,Goal]).
write_fact(P,Goal) :-
    Goal \= ad-_,
    recorded(gstream,S,_),
    format(S,'~q::~q.\n',[P,Goal]).

%%%%%
% Write clause
%%%%%
write_clause(Goal,true) :-
    !,\+certainly_true(Goal),
    recorded(gstream,S,_),
    format(S,'~q.\n',[Goal]).
write_clause(Goal,Body) :-
    recorded(gstream,S,_),
    remove_certainly_true(Body,PBody),
    format(S,'~q :- ',[Goal]),
    write_body(PBody),
    format(S,'.\n',[]).
write_body((Atom,Rest)) :-
    !,
    write_body(Atom),
    recorded(gstream,S,_),
    format(S,',',[]),
    write_body(Rest).
write_body(\+ Atom) :-
    !,write_body(not(Atom)).
write_body(not(Atom)) :-
    !,
    recorded(gstream,S,_),
    (Atom = R-P-Fact ->
        format(S,'\\+ ad~q_~q',[R,Fact])
    ;
        format(S,'\\+ (~q)',[Atom])
    ).
write_body(Atom) :-
    recorded(gstream,S,_),
    (Atom = R-P-Fact ->
        format(S,'ad~q_~q',[R,Fact])
    ;
        format(S,'~q',[Atom])
    ).

remove_certainly_true((Atom,Rest),Res) :-
    !,
    remove_certainly_true(Atom,P2Rest),
    remove_certainly_true(Rest,P1Rest),
    (P2Rest = true ->
      (P1Rest = true ->
         Res = true
      ;
         Res = P1Rest
      )
    ;
      (P1Rest = true ->
         Res = P2Rest
      ;
         Res = (P2Rest,P1Rest)
      )
    ).
remove_certainly_true(Atom,true) :-
    certainly_true(Atom),!.
remove_certainly_true(Atom,Atom).

%
% builtin_reused/1 defines the builtins that are reused by problog
%
builtin_reused((_,_)).
builtin_reused(not(_)).
builtin_reused(\+(_)).

%
% builtin_support/1 defines which builtins are supported
%
builtin_support('='(_,_)).
builtin_support(true).
builtin_support(is(_,_)).
builtin_support('@<'(_,_)).
builtin_support('@>'(_,_)).
builtin_support('@>='(_,_)).
builtin_support('>'(_,_)).
builtin_support('<'(_,_)).
builtin_support('=<'(_,_)).
builtin_support('>='(_,_)).
builtin_support('\\=='(_,_)).
builtin_support('\\='(_,_)).
builtin_support(write(_)).
builtin_support(writeln(_)).
builtin_support(format(_)).
builtin_support(format(_,_)).
builtin_support(between(_,_,_)).

builtin_support(_).  % SUPPORT ALL

