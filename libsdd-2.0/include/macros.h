/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#ifndef MACROS_H_
#define MACROS_H_

/****************************************************************************************
 * error checks and messages
 ****************************************************************************************/
 
#define ERR_MSG_GC "\nerror in %s: accessing sdd node that has been garbage collected\n"
#define ERR_MSG_DEREF "\nerror in %s: more dereferences than references to an sdd node\n"
#define ERR_MSG_INVALID_VAR "\nerror in %s: invalid var index\n"
#define ERR_MSG_VTREE "\nerror in %s: unrecognized vtree type\n"
#define ERR_MSG_INPUT_VTREE "\nerror in %s: must supply a vtree\n"
#define ERR_MSG_ONE_VAR "\nerror in %s: manager must have at least one variable\n"
#define ERR_MSG_TWO_VARS "\nerror in %s: manager must have at least two variables\n"
#define ERR_MSG_REM_VAR "\nerror in %s: removing a variable that is currently being used\n"
#define ERR_MSG_MOV_VAR "\nerror in %s: moving a variable that is currently being used\n"
#define ERR_MSG_NODE_ITR "\nerror in %s: argument not a decision node\n"
#define ERR_MSG_NODE_LIT "\nerror in %s: argument not a literal node\n"
#define ERR_MSG_WMC "\nerror in %s: WMC manager is no longer valid due to automatic SDD minimization\n"
#define ERR_MSG_FRG_N "\nerror in %s: fragment cannot be moved to the next state while in goto mode\n"
#define ERR_MSG_FRG_G "\nerror in %s: fragment cannot by moved to the given state while in next mode\n"
#define ERR_MSG_FRG_R "\nerror in %s: fragment cannot be rewinded while in goto mode\n"

//if condition C is met, print error message M that materialized in function F
#define CHECK_ERROR(C,M,F) if(C) { fprintf(stderr,M,F); exit(1); }

/****************************************************************************************
 * tests and lookups
 ****************************************************************************************/

//OP: constant, which is either CONJOIN or DISJOIN
//M: manager
#define ZERO(M,OP) (OP==CONJOIN? (M)->false_sdd: (M)->true_sdd)
#define ONE(M,OP) (OP==CONJOIN? (M)->true_sdd: (M)->false_sdd)

//OP: constant, which is either CONJOIN or DISJOIN
//N: node
#define IS_ZERO(N,OP) (OP==CONJOIN? (N)->type==FALSE: (N)->type==TRUE)
#define IS_ONE(N,OP) (OP==CONJOIN? (N)->type==TRUE: (N)->type==FALSE)

//OP: constant, which is either CONJOIN or DISJOIN
//N: sdd node
#define IS_FALSE(N) ((N)->type==FALSE)
#define IS_TRUE(N) ((N)->type==TRUE)
#define IS_LITERAL(N) ((N)->type==LITERAL)
#define IS_DECOMPOSITION(N) ((N)->type==DECOMPOSITION)

//checks that an sdd node is neither true nor false
//N: sdd node
#define NON_TRIVIAL(N) ((N)->type!=FALSE && (N)->type!=TRUE)
#define TRIVIAL(N) ((N)->type==FALSE || (N)->type==TRUE)

//checks whether node is on the gc_list (i.e., has been garbage collected)
//N: Sdd node
#define GC_NODE(N) (N->id==0)

//recovering fields of nodes
#define ELEMENTS_OF(N) (N)->alpha.elements
#define LITERAL_OF(N) (N)->alpha.literal
#define VAR_OF(N) labs((N)->alpha.literal)

//checking whether an sdd node is live
//terminal sdds are always live
#define LIVE(N) ((N)->type!=DECOMPOSITION || (N)->ref_count)
#define DEAD(N) ((N)->type==DECOMPOSITION && (N)->ref_count==0)

//tests whether a variable index is valid
//V: var index
//M: manager
#define VALID_VAR(V,M) (1 <= V && V <= M->var_count)

//a function is provided for this in the sddapi, but a macro is used internally for efficiency
#define LEAF(V) ((V)->left==NULL)
#define INTERNAL(V) ((V)->left)

//checks whether an sdd node N could be a valid prime/sub for vtree V
#define OK_PRIME(P,V) (NON_TRIVIAL(P) && sdd_vtree_is_sub((P)->vtree,(V)->left))
#define OK_SUB(S,V) (TRIVIAL(S) || sdd_vtree_is_sub((S)->vtree,(V)->right))

#define Assert(C)\
if(!(C)) {\
  void print_trace ();\
  print_trace();\
  assert(C);\
}

/****************************************************************************************
 * macro for constructing sdd nodes (see basic/partitions.c)
 ****************************************************************************************/

//sets N to sdd node with a compression of the elements added in code B
//if sdd node has form T.n, set N to n (trimming rule 1)
//if sdd node has form n.T + ~n.F, set N to n (trimming rule 2)
//
//the resulting node will be normalized to vtree V unless one of the trimming rules applies
//
//N: node
//V: vtree
//M: manager
//B: code (uses DECLARE_element(prime,sub,manager) to add elements)
//
//this macro may invoke apply
#define GET_node_from_partition(N,V,M,B) {\
  void START_partition(struct sdd_manager_t* manager);\
  void DECLARE_element(struct sdd_node_t* prime, struct sdd_node_t* sub, struct vtree_t* vtree, struct sdd_manager_t* manager);\
  SddNode* GET_node_of_partition(struct vtree_t* vtree, struct sdd_manager_t* manager, int limited);\
  START_partition(M);\
  B; /* execute code */\
  N = GET_node_of_partition(V,M,0);\
  assert(N);\
}

//this is used ONLY inside GET_node_from_partition_limited
//C: test for exceeding limit and aborting GET_node_from_partition_limited
#define ABORT_partition_if(C) if(C) goto abort

//T: time limit
//this macro may invoke apply
#define GET_node_from_partition_limited(N,V,M,T,B) {\
  void START_partition(struct sdd_manager_t* manager);\
  void DECLARE_element(struct sdd_node_t* prime, struct sdd_node_t* sub, struct vtree_t* vtree, struct sdd_manager_t* manager);\
  void ABORT_partition(struct sdd_manager_t* manager);\
  SddNode* GET_node_of_partition(struct vtree_t* vtree, struct sdd_manager_t* manager, int limited);\
  START_partition(M);\
  B; /* execute code: may jump to abort */\
  N = GET_node_of_partition(V,M,T); /* N may be NULL */\
  goto done;\
  abort:\
  N = NULL;\
  ABORT_partition(M);\
  done:\
  /* N may or may not be NULL */\
  ;\
}

//sets N to sdd node with elements declared in code B
//assumes elements already compressed, cannot be trimmed and normalized for vtree
//
//N: node
//V: vtree
//M: manager
//B: code (uses DECLARE_compressed_element(prime,sub,manager) to add elements)
//
//this macro will not invoke apply
#define GET_node_from_compressed_partition(N,V,M,B) {\
  void START_partition(struct sdd_manager_t* manager);\
  void DECLARE_compressed_element(struct sdd_node_t* prime, struct sdd_node_t* sub, struct vtree_t* vtree, struct sdd_manager_t* manager);\
  SddNode* GET_node_of_compressed_partition(struct vtree_t* vtree, struct sdd_manager_t* manager);\
  START_partition(M);\
  B; /* execute code */\
  N = GET_node_of_compressed_partition(V,M);\
}

/****************************************************************************************
 * macro for auto gc and search
 ****************************************************************************************/

//ensures that code B of manager M will run without ever invoking vtree search
//
//M: manager
//B: code
#define WITH_no_auto_mode(M,B) {\
  int _mode = M->auto_gc_and_search_on; /* save auto-search mode */\
  M->auto_gc_and_search_on = 0; /* deactivate auto-search */\
  B; /* execute code */\
  M->auto_gc_and_search_on = _mode; /* recover auto-search mode */\
}

//ensures that if code B of manager M invokes gc or vtree search, then gc/search
//will only take place locally (and not globally)
//
//M: manager
//B: code
#define WITH_local_auto_mode(M,B) {\
  int _mode = M->auto_local_gc_and_search_on;\
  M->auto_local_gc_and_search_on = 1;\
  B; /* execute code */\
  M->auto_local_gc_and_search_on = _mode;\
}

/****************************************************************************************
 * macro for timing code
 ****************************************************************************************/

//C: variable that accumulates time
//B: code
#define WITH_timing(C,B) {\
  clock_t start_time = clock();\
  B; /* execute code */\
  C += (clock()-start_time); /* accumulate time*/\
}

/****************************************************************************************
 * memory allocation, with error catching
 ****************************************************************************************/
 
//catching malloc and calloc errors
#define MALLOC(variable,type,message) {\
  variable = (type*) malloc(sizeof(type));\
  if(variable==NULL) {\
    fprintf(stderr,"\nmalloc failed in %s\n",message);\
    exit(1);\
  }\
}
  
#define CALLOC(variable,type,count,message) {\
  if(count==0) variable = NULL;\
  else {\
    if((variable=(type*) calloc(count,sizeof(type)))==NULL) {\
      fprintf(stderr,"\ncalloc failed in %s\n",message);\
      exit(1);\
    }\
  }\
}

#define REALLOC(variable,type,count,message) {\
  variable = (type*) realloc(variable,(count)*sizeof(type));\
  if(variable==NULL) {\
    fprintf(stderr,"\nrealloc failed in %s\n",message);\
    exit(1);\
  }\
}


/****************************************************************************************
 * general utilities
 ****************************************************************************************/

//compute max of A and B
//
//A: number
//B: number
#define MAX(A,B) (((A)>(B))?(A):(B))

//compute min of A and B
//
//A: number
//B: number
#define MIN(A,B) (((A)<(B))?(A):(B))

//swapping the values of two variables
//
//T: type of vars
//V1: first var
//V2: second var
#define SWAP(T,V1,V2) {	T V = V1; V1 = V2; V2 = V; }

//compute the space in megabytes for N members of type T
//
//N: number of elements of type T 
//T: type
#define TYPE2MB(N,T)  (((float)(N)*sizeof(T))/(1024*1024))

//add value to array whose elements are of type and has size
#define ADD_TO_ARRAY(type,value,array,size) {\
  ++(size);\
  REALLOC(array,type,size,"ADD_TO_ARRAY");\
  array[(size)-1] = value;\
}

#endif // MACROS_H_

/****************************************************************************************
 * end
 ****************************************************************************************/
