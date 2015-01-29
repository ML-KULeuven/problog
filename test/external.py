
def pysim(a,b):
    #print("a = {}, type = {}".format(a, type(a)))
    #print("b = {}, type = {}".format(b, type(b)))
    if not type(a) is str or not type(b) is str:
        return 0.0

    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    distance = current[n]
    return 1.0/(1.0+distance)


def pyedge(a):
    # This could be a call to a DB
    if a == "v1":
        return [("v2",0.3),("v3",0.5)]
    if a == "v2":
        return [("v1",0.1),("v3",0.8)]
    if a == "v3":
        return [("v1",0.2)]

