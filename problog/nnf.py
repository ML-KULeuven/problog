"""
Logic 

"""



# NNF == and-or graph
# OBDD : 
# d-DNNF : deterministic decomposable NNF
# sd-DNNF : smooth deterministic decomposable NNF
# SDD : structured decomposable + strong determinism

# OBDD < SDD < d-DNNF

# cycle-free: and-or-graph is a DAG 
# deterministic: disjuncts are logically disjoint
# decomposable: conjuncts do not share variables
# smoothness: all disjuncts mention the same set of variables
#   => smoothen => 
#    for each disjunction A = A1 \/ A2 \/ ... \/ An
#       if atoms(Ai) != atoms(A)
#           replace Ai with Ai /\ /\_A in S (A\/-A) where S = atoms(A)-atoms(Ai)       

# c2d => CNF -> d-DNNF
# dsharp => CNF -> sd-DNNF
# sdd => CNF/DNF -> sDD

# DIMACS format for CNF
#
# Command lines start with 'c'
# Problem line: 'p' 'cnf' #vars #clauses
# Clause: a b -c 0  where a,b,c are indexes of literals
# Clause always ends with 0   
#
#   and: self_lit -child_lit1 -child_lit2
#       -self_lit +child_lit1
#       -self_lit +child_lit2
#           
#   or:  -self_lit child_lit1 child_lit2
#        +self_lit -child_lit1
#        +self_lit -child_lit2
#
#
#   OR 1
#       AND 2
#           LIT3
#           LIT4
#       AND 5
#           LIT6
#           LIT7
#
# AND                   TAKE 3,4 = TRUE / 6,7 = FALSE
#   -1 OR 2 OR 5        TRUE
#       1 OR -2         1 = TRUE
#       1 OR -5         TRUE
#   2 OR -3 OR -4       For 2=TRUE => TRUE
#       -2 OR 3         TRUE
#       -2 OR 4         TRUE
#   5 OR -6 OR -7       TRUE
#       -5 OR 6         FALSE => 5 
#       -5 OR 7         FALSE => 5




class NNF(object) :
    
    def isDAG(self) :
        pass
        
    def isDeterministic(self) :
        pass
        
    def isDecomposable(self) :
        pass
        
    def isSmooth(self) :
        pass
                
    def getNode(self) :
        pass
        
    def getUsedFacts(self, index) :
        pass
        
    
