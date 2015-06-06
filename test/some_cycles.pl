%%% Paste your program, query and evidence here

1.0::isVulnerable(X) <- hasVuln(X,Z), vuln(Z).
1.0::isVulnerable(X) <- influencedBy(X,Y), isVulnerable(Y).

0.2::influencedBy(zz,q).
0.8::influencedBy(w,zz).

0.4::influencedBy(q,zz).
0.9::influencedBy(zz,w).

0.7::influencedBy(d,q).
0.3::influencedBy(d,w).

1.0::hasVuln(w,vw).
1.0::hasVuln(q,vq).

0.5::vuln(vw).
0.7::vuln(vq).

query(isVulnerable(d)). % outcome: 0.598

%0.5833

%0.79
%0.3784*0.5
%0.7144*0.7