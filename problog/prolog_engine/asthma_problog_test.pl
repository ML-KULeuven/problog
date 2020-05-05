person(0). friend(0,2). friend(0,3). friend(0,6). friend(0,7). friend(0,11). friend(0,12). friend(0,13). person(1). friend(1,2). friend(1,3). friend(1,4). friend(1,5). friend(1,6). friend(1,7). friend(1,8). friend(1,11). friend(1,12). friend(1,13). friend(1,14). person(2). friend(2,3). friend(2,7). friend(2,10). friend(2,11). friend(2,14). person(3). friend(3,6). friend(3,8). friend(3,10). friend(3,11). friend(3,13). friend(3,14). person(4). friend(4,7). friend(4,8). friend(4,9). friend(4,12). friend(4,13). person(5). friend(5,8). friend(5,12). person(6). friend(6,7). friend(6,8). friend(6,9). friend(6,10). friend(6,12). friend(6,14). person(7). friend(7,8). friend(7,12). friend(7,13). person(8). friend(8,10). friend(8,11). friend(8,13). person(9). friend(9,12). friend(9,14). person(10). friend(10,13). friend(10,14). person(11). friend(11,12). friend(11,14). person(12). person(13). friend(13,14). person(14). 
0.3::stress(X) :- person(X). 
0.2::influences(X,Y) :- person(X), person(Y). 
smokes(X) :- stress(X). 
smokes(X) :- friend(X,Y), influences(Y,X), smokes(Y). 
0.4::asthma(X) :- smokes(X). 
query(asthma(_)). 
