/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#ifndef ITERATORS_H_
#define ITERATORS_H_

/****************************************************************************************
 * iterator over ALL vtree nodes (linked using ->next, ->prev, ->first and ->last)
 ****************************************************************************************/

//iterates over each node V in a vtree rooted at R, executing code B for each node
//
//V: vtree node
//R: vtree root
//B: code
#define FOR_each_vtree_node(V,R,B) {\
  Vtree* V    = (R)->first; /* head of linked list */\
  Vtree* _out = (R)->last->next; /* node following last node of vtree (can be NULL) */\
  while(V!=_out) {\
    assert(V);\
	B; /* execute code */\
	V = V->next; /* next node of vtree */\
  }\
}

/****************************************************************************************
 * iterator over INTERNAL vtree nodes (linked using ->next, ->prev, ->first and ->last)
 *
 * the nodes of a vtree linked list alternate between leaf and internal nodes, with
 * the first and last nodes being leaves
 ****************************************************************************************/

//iterates over each internal node V in a vtree rooted at R, executing code B for each node
//
//V: vtree node
//R: vtree root
//B: code
#define FOR_each_internal_vtree_node(V,R,B) {\
  Vtree* V          = (R)->first; /* first leaf node of vtree */\
  Vtree* _last_leaf = (R)->last; /* last leaf node of vtree */\
  while(V!=_last_leaf) {\
    V = V->next; /* V is internal */\
    assert(INTERNAL(V));\
	B; /* execute code */\
	V = V->next; /* V is leaf */\
    assert(LEAF(V));\
  }\
}

/****************************************************************************************
 * iterator over LEAF vtree nodes (linked using ->next, ->prev, ->first and ->last)
 *
 * the nodes of a vtree linked list alternate between leaf and internal nodes, with
 * the first and last nodes being leaves
 ****************************************************************************************/

//iterates over each leaf node V in a vtree rooted at R, executing code B for each node
//
//V: vtree node
//R: vtree root
//B: code
#define FOR_each_leaf_vtree_node(V,R,B) {\
  Vtree* V          = (R)->first; /* first leaf node of vtree */\
  Vtree* _last_leaf = (R)->last; /* last leaf node of vtree */\
  while(V!=_last_leaf) {\
    assert(LEAF(V));\
	B; /* execute code */\
	V = V->next->next; /* V is leaf */\
  }\
  assert(LEAF(V));\
  B; /* execute code on last leaf */\
}

/****************************************************************************************
 * iterator over ANCESTOR vtree nodes (visiting children before parents)
 ****************************************************************************************/

//iterates over each ancestor V in a vtree rooted at R, executing code B for each node
//
//V: vtree node
//R: vtree root
//B: code
#define FOR_each_ancestral_vtree_node(V,R,B) {\
  Vtree* V = (R)->parent;\
  while(V) {\
	B; /* execute code */\
	V = V->parent;\
  }\
}


/****************************************************************************************
 * iterator over nodes normalized for a vtree 
 * (linked using the ->vtree_next and ->vtree_prev fields)
 ****************************************************************************************/

//iterate over the sdd nodes normalized for vtree node V, executing code B for each node
//this iterator allows freeing sdd nodes or changing their ->vtree_next and ->vtree_prev
//fields in code B
//
//N: node
//V: vtree
//B: code
#define FOR_each_sdd_node_normalized_for(N,V,B) {\
  SddNode* N = (V)->nodes; /* head of linked list */\
  while(N) {\
     /* save N->vtree_next in case N is freed or N->vtree_next is changed in code B */\
	SddNode* _next = N->vtree_next;\
	/* execute code */\
	B;\
	/* advance to next node */\
	N = _next;\
  }\
}

/****************************************************************************************
 * iterator over sdd nodes normalized for nodes in vtree 
 ****************************************************************************************/

//N: node
//V: vtree
//B: code
#define FOR_each_decomposition_in(N,V,B) FOR_each_internal_vtree_node(v,V,FOR_each_sdd_node_normalized_for(N,v,B))
#define FOR_each_decomposition_above(N,V,B) FOR_each_ancestral_vtree_node(v,V,FOR_each_sdd_node_normalized_for(N,v,B))
#define FOR_each_literal_in(N,V,B) FOR_each_leaf_vtree_node(v,V,FOR_each_sdd_node_normalized_for(N,v,B))

/****************************************************************************************
 * iterator for sdd nodes (linked using the ->next field)
 ****************************************************************************************/

//this iterator allows freeing nodes or changing their ->next field in code B
//
//N: node
//L: head of linked list (node)
//B: code
#define FOR_each_linked_node(N,L,B) {\
  SddNode* N = L; /* head of linked list */\
  while(N) {\
     /* save N->next in case N is freed or N->next is changed in code B */\
	SddNode* _next = N->next;\
	/* execute code */\
	B;\
	/* advance to next node */\
	N = _next;\
  }\
}

/****************************************************************************************
 * iterator for unique nodes
 ****************************************************************************************/

//iterates over unique nodes of manager, executing a bit of code for each node
//this iterator allows freeing nodes or changing their ->next field in code B
//
//N: node
//M: manager
//B: code to be executed for each node (references the variable E)
#define FOR_each_unique_node(N,M,B) {\
  SddHash* H = M->unique_nodes;\
  if(H->count) {\
    SddSize _size = H->size;\
    SddNode** _clists = H->clists;\
    while(_size--) {\
      SddNode* N = *_clists++;\
	  while(N) {\
	    /* save next in case N is freed or N->next is changed in code B */\
	    SddNode* _next = N->next;\
	    /* execute code */\
	    B;\
	    /* advance to next entry in hash table */\
	    N = _next;\
	  }\
    }\
  }\
}

/****************************************************************************************
 * macros for iterating over primes, subs and elements of an sdd node
 ****************************************************************************************/

//iterate over elements, executing code B for each element
//
//P: prime
//S: sub
//C: element count
//ES: elements array
//B: code
#define FOR_each_prime_sub_of_elements(P,S,C,ES,B) {\
  for(SddElement* _e=ES; _e<(ES)+(C); _e++) {\
	SddNode* P = _e->prime;\
    SddNode* S = _e->sub;\
    /* execute code */\
	B;\
  }\
}

//iterate over primes and subs P/S of sdd node N, executing code B for each P/S pair
//
//P: prime
//S: sub
//N: sdd node
//B: code
#define FOR_each_prime_sub_of_node(P,S,N,B) {\
  assert(IS_DECOMPOSITION(N));\
  FOR_each_prime_sub_of_elements(P,S,(N)->size,ELEMENTS_OF(N),B)\
}

//iterate over primes P of sdd node N, executing code B for each prime
//
//P: prime
//N: sdd node
//B: code
#define FOR_each_prime_of_node(P,N,B) {\
  assert(IS_DECOMPOSITION(N));\
  for(SddElement* _e=ELEMENTS_OF(N); _e<ELEMENTS_OF(N)+(N)->size; _e++) {\
	SddNode* P = _e->prime;\
    /* execute code */\
	B;\
  }\
}

//iterate over subs S of sdd node N, executing code B for each sub
//
//S: sub
//N: sdd node
//B: code
#define FOR_each_sub_of_node(S,N,B) {\
  assert(IS_DECOMPOSITION(N));\
  for(SddElement* _e=ELEMENTS_OF(N); _e<ELEMENTS_OF(N)+(N)->size; _e++) {\
	SddNode* S = _e->sub;\
    /* execute code */\
	B;\
  }\
}

#endif // ITERATORS_H_

/****************************************************************************************
 * end
 ****************************************************************************************/
