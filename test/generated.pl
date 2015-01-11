%Expected outcome:
% klQuery1 0.15003609408123691
% klQuery10 0.13583364369744425
% klQuery11 0.2452640126357755
% klQuery12 0.007449743080399876
% klQuery13 0.051928428978961064
% klQuery14 0.20078532673721425
% klQuery15 0.1567183366817575
% klQuery16 0.09599541903441781
% klQuery17 0.12560484657280263
% klQuery18 0.14597320007109496
% klQuery19 0.2652706413012306
% klQuery2 0.11509323517359456
% klQuery20 0.006307405342666971
% klQuery21 0.10297622850463256
% klQuery22 0.16860181813926522
% klQuery23 0.10871269207155218
% klQuery24 0.1628653545723454
% klQuery25 0.2878490840707309
% klQuery26 0.33452731392003865
% klQuery27 0.6068245130992083
% klQuery28 0.015551884891561242
% klQuery29 0.11354291521319712
% klQuery3 0.25512357181849593
% klQuery30 0.5088334827775723
% klQuery31 0.24169197031374615
% klQuery32 0.3806844276770233
% klQuery33 0.3403254597512213
% klQuery34 0.41356939288282163
% klQuery35 0.7351228708034552
% klQuery36 0.018771981830587703
% klQuery37 0.16404751993316768
% klQuery38 0.5898473327008751
% klQuery39 0.3116545471410136
% klQuery4 0.01000575743633546
% klQuery40 0.44224030549302934
% klQuery41 0.10608374388845243
% klQuery42 0.13142122482047938
% klQuery43 0.23221896735398573
% klQuery44 0.005286001354946104
% klQuery45 0.051681081191063585
% klQuery46 0.18582388751786832
% klQuery47 0.09507323897418539
% klQuery48 0.14243172973474644
% klQuery49 0.26512932925483146
% klQuery5 0.037661997372230395
% klQuery50 0.2527137557161753
% klQuery51 0.2715780466438976
% klQuery52 0.6223763979907696
% klQuery53 0.753894852634043
% klQuery54 0.23750496870893176
% klQuery6 0.227467331882601
% klQuery7 0.10613127050070906
% klQuery8 0.15899805875412243
% klQuery9 0.11688011201873114




0.4625::node3.
0.9751::node15.
0.2176::node18.
0.4003::node19.
0.2699647::pfact1.
0.34486834::pfact2.
0.0809573::pfact3.
0.23828006::pfact4.
0.0334016::pfact5.
0.19677152::pfact6.
0.6361178::pfact7.
0.60130316::pfact8.
0.4038451718::pfact9.
0.55479861632::pfact10.
0.38306338196::pfact11.
0.542125685504::pfact12.
0.12692123425::pfact13.
0.0842738672::pfact14.
0.17010467035::pfact15.
0.14934269984::pfact16.
0.4674995692::pfact17.
0.69789214072::pfact18.
0.21768487264::pfact19.
0.129662732224::pfact20.
0.530786816808125::pfact21.
0.514973761647213::pfact22.
0.509399042778603::pfact23.
0.562249054089031::pfact24.
0.624668625685738::pfact25.
0.667600504902681::pfact26.
0.606858786299073::pfact27.
0.675113552927393::pfact28.
0.769669332448::pfact29.
0.786196069312::pfact30.
0.713256958672::pfact31.
0.758241023968::pfact32.
0.229902433045775::pfact33.
0.245244649378603::pfact34.
0.20852990334049::pfact35.
0.215523964180531::pfact36.
node5 :- pfact29,node3,node19.
node5 :- pfact30,not(node3),node19.
node5 :- pfact31,node3,not(node19).
node5 :- pfact32,not(node3),not(node19).
node6 :- pfact33,node3,node15.
node6 :- pfact34,not(node3),node15.
node6 :- pfact35,node3,not(node15).
node6 :- pfact36,not(node3),not(node15).
invented_symbol_31 :- not(node15),not(node18).
node1 :- invented_symbol_31,pfact6,node3.
node11 :- invented_symbol_31,pfact12,node19.
node12 :- invented_symbol_31,pfact20.
node2 :- invented_symbol_31,pfact27,node19.
invented_symbol_75 :- not(node15),node18.
node1 :- invented_symbol_75,pfact5,node3.
node11 :- invented_symbol_75,pfact10,node19.
node12 :- invented_symbol_75,pfact18.
node2 :- invented_symbol_75,pfact23,node19.
invented_symbol_140 :- node15,not(node18).
node1 :- invented_symbol_140,pfact2,node3.
node11 :- invented_symbol_140,pfact11,node19.
node12 :- invented_symbol_140,pfact19.
node2 :- invented_symbol_140,pfact25,node19.
invented_symbol_229 :- node15,node18.
node1 :- invented_symbol_229,pfact1,node3.
node11 :- invented_symbol_229,pfact9,node19.
node12 :- invented_symbol_229,pfact17.
node2 :- invented_symbol_229,pfact21,node19.
node1 :- invented_symbol_317,not(node3).
invented_symbol_317 :- invented_symbol_31,pfact8.
invented_symbol_317 :- invented_symbol_75,pfact7.
invented_symbol_317 :- invented_symbol_140,pfact4.
invented_symbol_317 :- invented_symbol_229,pfact3.
node11 :- invented_symbol_380,not(node19).
invented_symbol_380 :- invented_symbol_31,pfact16.
invented_symbol_380 :- invented_symbol_75,pfact14.
invented_symbol_380 :- invented_symbol_140,pfact15.
invented_symbol_380 :- invented_symbol_229,pfact13.
node2 :- invented_symbol_418,not(node19).
invented_symbol_418 :- invented_symbol_31,pfact28.
invented_symbol_418 :- invented_symbol_75,pfact24.
invented_symbol_418 :- invented_symbol_140,pfact26.
invented_symbol_418 :- invented_symbol_229,pfact22.
klQuery2 :- \+ node3, node1.
query(klQuery2).
klQuery1 :- node3, node1.
query(klQuery1).
klQuery3 :- node15, node1.
query(klQuery3).
klQuery4 :- not node15, node1.
query(klQuery4).
klQuery5 :- node18, node1.
query(klQuery5).
klQuery6 :- not node18, node1.
query(klQuery6).
klQuery7 :- node19, node1.
query(klQuery7).
klQuery8 :- not node19, node1.
query(klQuery8).
klQuery9 :- node3, node11.
query(klQuery9).
klQuery10 :- not node3, node11.
query(klQuery10).
klQuery11 :- node15, node11.
query(klQuery11).
klQuery12 :- not node15, node11.
query(klQuery12).
klQuery13 :- node18, node11.
query(klQuery13).
klQuery14 :- not node18, node11.
query(klQuery14).
klQuery15 :- node19, node11.
query(klQuery15).
klQuery16 :- not node19, node11.
query(klQuery16).
klQuery17 :- node3, node12.
query(klQuery17).
klQuery18 :- not node3, node12.
query(klQuery18).
klQuery19 :- node15, node12.
query(klQuery19).
klQuery20 :- not node15, node12.
query(klQuery20).
klQuery21 :- node18, node12.
query(klQuery21).
klQuery22 :- not node18, node12.
query(klQuery22).
klQuery23 :- node19, node12.
query(klQuery23).
klQuery24 :- not node19, node12.
query(klQuery24).
klQuery25 :- node3, node2.
query(klQuery25).
klQuery26 :- not node3, node2.
query(klQuery26).
klQuery27 :- node15, node2.
query(klQuery27).
klQuery28 :- not node15, node2.
query(klQuery28).
klQuery29 :- node18, node2.
query(klQuery29).
klQuery30 :- not node18, node2.
query(klQuery30).
klQuery31 :- node19, node2.
query(klQuery31).
klQuery32 :- not node19, node2.
query(klQuery32).
klQuery33 :- node3, node5.
query(klQuery33).
klQuery34 :- not node3, node5.
query(klQuery34).
klQuery35 :- node15, node5.
query(klQuery35).
klQuery36 :- not node15, node5.
query(klQuery36).
klQuery37 :- node18, node5.
query(klQuery37).
klQuery38 :- not node18, node5.
query(klQuery38).
klQuery39 :- node19, node5.
query(klQuery39).
klQuery40 :- not node19, node5.
query(klQuery40).
klQuery41 :- node3, node6.
query(klQuery41).
klQuery42 :- not node3, node6.
query(klQuery42).
klQuery43 :- node15, node6.
query(klQuery43).
klQuery44 :- not node15, node6.
query(klQuery44).
klQuery45 :- node18, node6.
query(klQuery45).
klQuery46 :- not node18, node6.
query(klQuery46).
klQuery47 :- node19, node6.
query(klQuery47).
klQuery48 :- not node19, node6.
query(klQuery48).
klQuery49 :- node1.
query(klQuery49).
klQuery50 :- node11.
query(klQuery50).
klQuery51 :- node12.
query(klQuery51).
klQuery52 :- node2.
query(klQuery52).
klQuery53 :- node5.
query(klQuery53).
klQuery54 :- node6.
query(klQuery54).

