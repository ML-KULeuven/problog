%% Whirl ProbLog implementation
%%
%% From:
%% W. W. Cohen. Whirl: A word-based information representation language.
%% Artificial Intelligence, 118(1):163–196, 2000.
%%
%% W. W. Cohen. Data integration using similarity joins and a word-based
%% information representation language. ACM Transactions on Information
%% Systems (TOIS), 18(3):288–321, 2000.
%%
%% Recursion and negation:
%% WHIRL does not allow for recursion or negation. Additionally, only one
%% many/2 predicate is allowed in a query. These restrictions do not apply
%% to Problog because it knows how to handle such situations (e.g. the
%% inclusion-exclusion principle).
%%
%% Search strategy:
%% Problog does not use a find-best-substitution approach, whereas the WHIRL
%% system uses an A* search strategy. The default reasoning technique in
%% Problog finds all proofs.
%%
%% Author:
%% - Wannes Meert
%% - Anton Dries
%%

:- load_external('whirl.py').

P::similar(X,Y) :- call_external(similarity(X,Y), P).


% FACTS

listing('Roberts Theater Chatham', 'Brassed Off', '7:15 - 9:10').
listing('Berkeley Cinema', 'Hercules', '2:00 - 4:15 - 7:30').
listing('Sony Mountainside Theater', 'Men in Black', '7:40 - 8:40 - 9:30 - 10:10').

review('Men in Black, 1997', '(***) One of the summer s biggest hits, this ... a comedy about space aliens with Will Smith ...').
review('Face/Off, 1997',     '(**1/2) After a somewhat slow start, Cage and Travolta').
review('Space Balls, 1987',  '(*1/2) While not one of Mel Brooks better efforts, this Star Wars spoof ... a comedy about space').
review('Hercules',  'Animated Disney film').
review('The Lord of the Rings: The Fellowship of the Ring, 2001', 'An epic fantasy film directed by Peter Jackson based on the first volume of J. R. R. Tolkien The Lord of the Rings. It is the first installment in the The Lord of the Rings film series, and was followed by The Two Towers (2002) and The Return of the King (2003), based on the second and third volumes of The Lord of the Rings.').

academy_award('Best makeup').

winner('Men in Black', 'Best makeup', 1997).
winner('The Lord of the Rings: The Fellowship of the Ring', 'Best makeup', 2001).


% CONJUNCTIVE QUERIES

% Find reviews about comedies with space aliens
q1(Movie, Review) :- review(Movie, Review),
                     similar(Review, 'comedy with space aliens').
%query(q1(X,Y)).

% Soft database join to see times and reviews
q2(Cinema,Movie1,Times,Review) :- listing(Cinema, Movie1, Times),
                                  review(Movie2, Review),
                                  similar(Movie1,Movie2).
%query(q2(C,M,T,R)).

% Database join to see times and reviews
% Whill be empty if no exactly matching movie titles
q3(Cinema,Movie,Times,Review) :- listing(Cinema,Movie,Times),
                                 review(Movie,Review).
%query(q3(C,M,T,R)).

% See where the latest science fiction comedy is playing
q4(Movie1) :- listing(Cinema, Movie1, Times), review(Movie2, Review),
              similar(Movie1, Movie2),
              similar(Review,"comedy with space aliens").
%query(q4(M)).


% DISJUNCTIVE QUERIES

% Disjunctive version of Q4
% Finds cinemas that are playing either a science fiction comedy or an animated
% film produced by Disney
view1(Cinema) :- listing(Cinema, Movie1, Times), 
                 review(Movie2, Review),
                 similar(Movie1, Movie2),
                 similar(Review, "comedy with space aliens").
view1(Cinema) :- listing(Cinema, Movie1, Times),
                 review(Movie2, Review),
                 similar(Movie1, Movie2),
                 similar(Review, "animated Walt Disney film").
q4a(Cinema) :- view1(Cinema).
%query(q4a(C)).


% SOFT UNIVERSAL QUANTIFICATION

% Movie with Will Smith that won an award
q5a(Movie, Cat) :- review(Movie ,Review),
              similar(Review, "Will Smith"),
              academy_award(Cat), winner(Movie2, Cat),
              similar(Movie, Movie2).
%query(q5a(M,C)).

% Auxiliaries
%P::many(Template, Test) :- ...
% TODO: pass atoms
many_int(L, S) :-
    many_int(L, 0, 0, S).
many_int([], P, N, 0) :-
	T is P+N,
    T = 0.
many_int([], P, N, S) :-
	T is P+N,
	T > 0,
    S is P/T.
many_int([H|T], PA, NA, S) :-
    (  call(H), PAN is PA + 1, NAN is NA;      % Test: true
     \+call(H), PAN is PA,     NAN is NA + 1), % Test: false
    many_int(T, PAN, NAN, S).


% Movie that has most academy awards
S::many_awards(M) :-
	% many(Template, Test) as findall(Test, Template, L)
	findall(winner(M,C,Y), (academy_award(C),winner(_,C,Y)), L),
	many_int(L, S).

q5(M) :-
	listing(C,M,T),
	%many(academy_award(Y), winner(M,Y)).
	many_awards(M).
query(q5(M)).


