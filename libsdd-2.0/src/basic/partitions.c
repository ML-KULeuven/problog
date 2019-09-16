/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/hash.c
SddNode* lookup_sdd_node(SddElement* elements, SddNodeSize size, SddHash* hash, SddManager* manager);

//basic/memory.c
SddElement* new_elements(SddNodeSize size, SddManager* manager);

//basic/nodes.c
SddNode* construct_decomposition_sdd_node(SddNodeSize size, SddElement* elements, Vtree* vtree, SddManager* manager);

//sdds/apply.c
SddNode* apply(SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager, int limited);

/****************************************************************************************
 * a utility for constructing a compressed, trimmed partition from an uncompressed one:
 *
 * --START_partition: is called before uncompressed elements are declared
 *
 * --DECLARE_element: is called to declare elements
 *
 * --GET_node_of_partition: is called after all elements has been declared
 *   and returns an sdd node corresponding to the compressed and trimmed partition
 *
 * --GET_elements_of_partition: is called after all elements has been 
 *   declared returns the compressed elements (assumes no trimming is possible).
 *
 * the macro GET_node_from_partition provides a shortcut for this usage, returning
 * a unique node corresponding to the compressed and trimmed elements
 *
 * the above utility in this file is also used for constructing cartesian products
 *
 * one can also get nodes corresponding to a compressed partition using START_partition(),
 * DECLARE_compressed_element() and GET_node_of_compressed_partition()
 * the macro GET_node_from_compressed_partition provides a shortcut for this
 ****************************************************************************************/

/****************************************************************************************
 * macros for popping/pushing elements from compression and elements stack, while
 * ensuring protection of elements
 *
 * protection is needed in case compression invokes vtree search
 ****************************************************************************************/
 
//reference primes/subs when pushing elements on compression_stack
//dereference primes/subs when popping elements from compression_stack
#define uPUSH_ELM(P,S,M) { PUSH_ELM(P,S,compression_stack,M); if(M->auto_gc_and_search_on) { sdd_ref(P,M);   sdd_ref(S,M);   } }
#define  uPOP_ELM(P,S,M) { POP_ELM(P,S,compression_stack,M);  if(M->auto_gc_and_search_on) { sdd_deref(P,M); sdd_deref(S,M); } }
//#define uPUSH_ELM(P,S,M) { PUSH_ELM(P,S,compression_stack,M); }
//#define  uPOP_ELM(P,S,M) { POP_ELM(P,S,compression_stack,M); }

//reference primes when pushing elements on element_stack
//dereference primes when popping elements from element_stack
//NOTE: compression will not gc subs
#define cPUSH_ELM(P,S,M) { PUSH_ELM(P,S,element_stack,M); if(M->auto_gc_and_search_on) sdd_ref(P,M); }
#define  cPOP_ELM(P,S,M) { POP_ELM(P,S,element_stack,M);  if(M->auto_gc_and_search_on) sdd_deref(P,M); }
//#define cPUSH_ELM(P,S,M) { PUSH_ELM(P,S,element_stack,M); }
//#define  cPOP_ELM(P,S,M) { POP_ELM(P,S,element_stack,M);  }

/****************************************************************************************
 * openning a partition
 ****************************************************************************************/
 
//saves number of elements currently on compression_stack
void START_partition(SddManager* manager) {
  SddSize cur_element_count = manager->top_compression_stack - manager->start_compression_stack;
  PUSH_STACK(cur_element_count,SddSize,meta_compression_stack,manager);
}

/****************************************************************************************
 * declaring elements
 ****************************************************************************************/

//push element on compression stack
//used for declaring elements that may need to be compressed
void DECLARE_element(SddNode* prime, SddNode* sub, Vtree* vtree, SddManager *manager) {
  assert(!IS_FALSE(prime));
  assert(IS_TRUE(prime) || sdd_vtree_is_sub(prime->vtree,vtree->left));
  assert(TRIVIAL(sub) || sdd_vtree_is_sub(sub->vtree,vtree->right));
  
  uPUSH_ELM(prime,sub,manager);
}

//push element on compression stack
//used for declaring elements that are compressed
void DECLARE_compressed_element(SddNode* prime, SddNode* sub, Vtree* vtree, SddManager *manager) {
  assert(!IS_FALSE(prime));
  assert(IS_TRUE(prime) || sdd_vtree_is_sub(prime->vtree,vtree->left));
  assert(TRIVIAL(sub) || sdd_vtree_is_sub(sub->vtree,vtree->right));
  
  PUSH_ELM(prime,sub,compression_stack,manager); //no referencing
}

/****************************************************************************************
 * aborting a node partition
 ****************************************************************************************/
 
//this is called when limits are exceeded before the partition is closed
//it removes the elements declared thus far from compression_stack
void ABORT_partition(SddManager *manager) {
 
  SddSize prev_elem_count  = POP_STACK(meta_compression_stack,manager);
  SddSize total_elem_count = manager->top_compression_stack - manager->start_compression_stack;
  SddSize cur_elm_count    = total_elem_count-prev_elem_count;
 
  SddNode* p; SddNode* s;
  while(cur_elm_count--) { uPOP_ELM(p,s,manager); }
}

/****************************************************************************************
 * reseting compression_stack and element_stack to their initial state (before
 * a partition was opened).
 ****************************************************************************************/

//compressed_count: number of elements pushed on the elements stack (by compression)
//uncompressed_count: number of elements put on the compression stack (by declare element)
//processed_uncompressed_count: number of elements poped from the compression stack
//
//Note: popping does not really remove elements from the stack; it just adjusts the
//pointer to the stack top (elements continue to reside on the stack even after popping)
static inline
void reset_stacks(SddSize compressed_count, SddSize uncompressed_count, SddSize processed_uncompressed_count, SddManager* manager) {
  SddNode* p; SddNode* s;
  
  //reset compressed elements stack
  while(compressed_count--) { cPOP_ELM(p,s,manager); }
  
  //reset uncompressed elements stack
  while(processed_uncompressed_count++ < uncompressed_count) { uPOP_ELM(p,s,manager); }
}

/****************************************************************************************
 * compressing and trimming declared elements of partition
 ****************************************************************************************/
 
//if node!=NULL is returned, then trimming took place and node is the result
//if node==NULL, then size and elements describe the compressed elements
static
int compress_and_trim(SddNodeSize* size, SddElement** elements, SddNode** node, Vtree* vtree, SddManager* manager, int limited) {

  //count of elements on the compression stack before this partition was opened
  SddSize prev_elements_count       = POP_STACK(meta_compression_stack,manager);
  //count and array of elements to be compressed
  SddElement* uncompressed_elements = manager->start_compression_stack + prev_elements_count;
  SddSize uncompressed_count        = manager->top_compression_stack - uncompressed_elements;
  
  //at least one element must have been declared
  assert(uncompressed_count >= 1);
  
  //keeping track of largest uncompressed decomposition
  manager->stats.max_uncompressed_decomposition_size = MAX(uncompressed_count,manager->stats.max_uncompressed_decomposition_size);

  //order elements by sub id: order will be reversed when elements are pushed on elements stack
  sort_uncompressed_elements(uncompressed_count,(SddElement*)uncompressed_elements);
  //all equal subs appear consecutively now, with false subs before true subs
  
  SddNode* first_sub = uncompressed_elements[0].sub;
  SddNode* last_sub  = uncompressed_elements[uncompressed_count-1].sub;
  //uncompressed_elements is not valid after calling apply (stacks may be resized and moved)
  
  SddSize processed_uncompressed_count = 0;
  SddNodeSize compressed_count         = 0;
  SddNode* trimmed                     = NULL;
  
  //FIRST trimming rule: node has form T.sub, return sub
  //this avoids disjoining the primes as the result must be true
  if(first_sub==last_sub) { //only one distinct sub, prime must be true
    trimmed = first_sub;
    goto success;
  }
  assert(uncompressed_count >= 2); //since we have at least two distinct subs
  
  #ifndef NDEBUG 
  Vtree** vtree_loc = sdd_vtree_location(vtree,manager); //does not change
  Vtree* right = vtree->right; //does not change
  #endif
  
  //SECOND trimming rule: node has form prime.T + ~prime.F, return prime
  //the following test assumes that (id of false) < (id of true); since elements
  //with true subs are popped first from the stack
  if(IS_FALSE(first_sub) && IS_TRUE(last_sub)) { //only two distinct subs: true and false
    //disjoin only the primes of the true sub
    SddNode* prime = manager->false_sdd; //will contain the disjunction of primes for the true sub
    SddNode* p; SddNode* s;
    uPOP_ELM(p,s,manager); ++processed_uncompressed_count;
    while(IS_TRUE(s)) { //compress
      assert(NON_TRIVIAL(p));
	  assert(sdd_vtree_is_sub(p->vtree,vtree->left));
      //vtree->left may be changed by auto search (use vtree->left instead of saving into left)
      WITH_local_auto_mode(manager,
        prime = apply(p,prime,DISJOIN,manager,limited)); //only vtree->left may change
      if(prime==NULL) goto failure; //closing partition failed due to time limits
      assert(NON_TRIVIAL(prime));
	  assert(sdd_vtree_is_sub(prime->vtree,vtree->left));
      assert(vtree==*vtree_loc); //apply does not change vtree
      assert(right==vtree->right); //apply does not change right subtree
      uPOP_ELM(p,s,manager); ++processed_uncompressed_count;
    }
    trimmed = prime;
    goto success;
  }
  
  //no trimming

  //pop uncompressed elements, compressing and placing compressed elements on element_stack
  SddNode* c_prime; SddNode* c_sub;
  uPOP_ELM(c_prime,c_sub,manager); ++processed_uncompressed_count;
  assert(OK_PRIME(c_prime,vtree));
	  
  while(processed_uncompressed_count < uncompressed_count) { 
    SddNode* prime; SddNode* sub;
    uPOP_ELM(prime,sub,manager); ++processed_uncompressed_count;
    assert(OK_PRIME(prime,vtree));
    assert(OK_SUB(sub,vtree));
	if(sub==c_sub) { //compress (apply will not gc sub)
      //vtree->left may be changed by auto search (use vtree->left instead of saving into left)
      WITH_local_auto_mode(manager,
        c_prime = apply(prime,c_prime,DISJOIN,manager,limited)); //only vtree->left may change
	  if(c_prime==NULL) goto failure; //closing partition failed due to time limit
	  assert(OK_PRIME(c_prime,vtree));
      assert(vtree==*vtree_loc); //apply does not change vtree
      assert(right==vtree->right); //apply does not change right subtree
	}
	else { //just popped a new element
	  //place previous compressed element on element_stack
	  assert(OK_PRIME(c_prime,vtree));
	  assert(OK_SUB(c_sub,vtree));
	  cPUSH_ELM(c_prime,c_sub,manager);
	  ++compressed_count;
      c_prime = prime;
	  c_sub = sub;
	}
  }
  
  cPUSH_ELM(c_prime,c_sub,manager);
  ++compressed_count;
  
  //keeping track of largest decomposition
  manager->stats.max_decomposition_size = MAX(compressed_count,manager->stats.max_decomposition_size);
  
  success:
    reset_stacks(compressed_count,uncompressed_count,processed_uncompressed_count,manager);
    *size     = compressed_count;
    *elements = manager->top_element_stack;
    *node     = trimmed;
    return 1;
  
  failure:
    reset_stacks(compressed_count,uncompressed_count,processed_uncompressed_count,manager);
    return 0;
}

/****************************************************************************************
 * constructing sdd nodes or looking them from a unique table
 ****************************************************************************************/
 
static
SddNode* lookup_or_construct_sdd_node(SddNodeSize size, SddElement* elements, Vtree* vtree, SddManager* manager) {
 
  //lookup from unique table
  SddHash* hash = manager->unique_nodes;
  SddNode* node = lookup_sdd_node(elements,size,hash,manager);

  if(node==NULL) node = construct_decomposition_sdd_node(size,elements,vtree,manager);
  
  return node;
}

/****************************************************************************************
 * ending a partition:
 *
 * --GET_node_of_partition: returns a node corresponding to the partition after it has
 *   been compressed and trimmed
 * --GET_elements_of_partition: returns the partition itself after it has been trimmed
 * --GET_node_of_compressed_partition: returns a node corresponding to the declared
 *   compressed partition (no compression or trimming)
 *
 ****************************************************************************************/

//returns 0 if closure is not successful
//otherwise, returns 1 and compressed partition (in size and elements)
//NOTE: elements array is newly allocated and can be claimed by the calling function
int GET_elements_of_partition(SddNodeSize* size, SddElement** elements, Vtree* vtree, SddManager* manager, int limited) {
  
  SddElement* buffer;
  SddNode* trim;
    
  int success = compress_and_trim(size,&buffer,&trim,vtree,manager,limited);
  assert(success==0 || trim==NULL);
  
  if(success) {
    *elements = new_elements(*size,manager);
    memcpy(*elements,buffer,*size*sizeof(SddElement));
    
    assert(elements_sorted_and_compressed(*size,*elements));
    assert(vtree==lca_of_compressed_elements(*size,*elements,manager));
  }
  
  return success;
}

//CALLED ONLY by macros GET_node_from_partition() and GET_node_from_partition_limited()
//
//returns NULL if closure is not successful
//otherwise, returns a node corresponding to the compressed partition (existing or newly created)
SddNode* GET_node_of_partition(Vtree* vtree, SddManager* manager, int limited) {
  
  SddNodeSize size;
  SddElement* buffer;
  SddNode* trim;
  
  int success = compress_and_trim(&size,&buffer,&trim,vtree,manager,limited);
  
  if(success==0) return NULL;
  else if(trim) return trim; //trimming
  else return lookup_or_construct_sdd_node(size,buffer,vtree,manager);
}


//CALLED ONLY by macro GET_node_from_compressed_partition()
//
//always succeeds
SddNode* GET_node_of_compressed_partition(Vtree* vtree, SddManager* manager) {
  
  //count of elements on the compression stack before this partition was opened
  SddSize prev_count   = POP_STACK(meta_compression_stack,manager);
  
  //count and array of compressed elements
  SddElement* elements = manager->start_compression_stack + prev_count;
  SddSize count        = manager->top_compression_stack - elements;
  
  //reset stack top (effectively pop elements)
  manager->top_compression_stack = elements;
  
  sort_compressed_elements(count,elements);
  
  assert(elements_sorted_and_compressed(count,elements));
  assert(vtree==lca_of_compressed_elements(count,elements,manager));
    
  return lookup_or_construct_sdd_node(count,elements,vtree,manager);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
