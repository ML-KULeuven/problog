/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/multiply.c
int multiply_decompositions(SddElement* elements1, SddNodeSize size1, SddElement* elements2, SddNodeSize size2, 
 BoolOp op, Vtree* vtree, SddManager* manager, int limited, void fn(SddNode* prime, SddNode* sub, Vtree* vtree, SddManager* manager));

//vtree_operations/limits.c
void start_apply_limits(SddManager* manager);
void end_apply_limits(SddManager* manager);
int apply_aborted(SddManager* manager);
int exceeded_limits(SddManager* manager);

//vtree_search/auto.c
void try_auto_gc_and_minimize(Vtree* vtree, SddManager* manager);

//vtrees/compare.c
char cmp_vtrees(Vtree** lca, Vtree* vtree1, Vtree* vtree2);

//local declarations
SddNode* apply(SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager, int limited);

/****************************************************************************************
 * macros for protecting nodes
 ****************************************************************************************/
 
#define c_ref(N,M)   if(M->auto_gc_and_search_on) sdd_ref(N,M)
#define c_deref(N,M) if(M->auto_gc_and_search_on) sdd_deref(N,M)
//#define c_ref(N,M)   
//#define c_deref(N,M) 

/****************************************************************************************
 * apply
 *
 * suppose node is a decomposition normalized for vtree
 *
 * invariants: 
 * --node cannot have true prime (trimming)
 * --node cannot have all its subs as either true or false (trimming)
 * --all variables of node appear in vtree
 * --no subtree of vtree satisfies above property
 *
 ****************************************************************************************/

//node1 and node2 are normalized for arbitrary vtrees
//no time limits
SddNode* sdd_apply(SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager) {
  assert(node1!=NULL && node2!=NULL);
  CHECK_ERROR(GC_NODE(node1),ERR_MSG_GC,"sdd_apply");
  CHECK_ERROR(GC_NODE(node2),ERR_MSG_GC,"sdd_apply");
  
  SddNode* node = apply(node1,node2,op,manager,0);
  assert(node!=NULL);
  return node;  
}

/****************************************************************************************
 * negate
 ****************************************************************************************/

//node is normalized for an arbitrary vtree
SddNode* sdd_negate(SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_negate");

  SddNode* negation = node->negation;
  if(negation!=NULL) {
    assert(!GC_NODE(negation));
    assert(node->vtree==negation->vtree);
    assert(node==negation->negation);
    return negation;
  }

  assert(node->type==DECOMPOSITION);
  Vtree* vtree = node->vtree;
  
  //Note: compression is not possible here
  GET_node_from_compressed_partition(negation,vtree,manager,{
    FOR_each_prime_sub_of_node(prime,sub,node,{
	  SddNode* sub_neg = sdd_negate(sub,manager);
	  DECLARE_compressed_element(prime,sub_neg,vtree,manager);
	});
  });
  assert(negation);
  assert(node->vtree==negation->vtree);
  
  //cache negations
  node->negation = negation;
  negation->negation = node;

  return negation;
}

/****************************************************************************************
 * apply case 1: node1->vtree = node2->vtree
 ****************************************************************************************/

//node1 and node2 are normalized for vtree (node1->vtree = vtree and node2->vtree = vtree)
//result is normalized for vtree unless it was trimmed
static
SddNode* sdd_apply_equal(SddNode* node1, SddNode* node2, BoolOp op, Vtree* vtree, SddManager* manager, int limited) {
  assert(node1!=NULL && node2!=NULL);
  assert(NON_TRIVIAL(node1) && NON_TRIVIAL(node2));
  assert(node1->vtree==vtree);
  assert(node1->vtree==vtree);
  
  c_ref(node1,manager); c_ref(node2,manager); //protect
  
  SddNode* node;
  //NOTE: compression is possible here (i.e., apply may be called)
  GET_node_from_partition_limited(node,vtree,manager,limited,{
    int success =
      multiply_decompositions(
        ELEMENTS_OF(node1),node1->size,
        ELEMENTS_OF(node2),node2->size,
        op,vtree,manager,limited,
        DECLARE_element);
    ABORT_partition_if(success==0);
  });
  
  c_deref(node1,manager); c_deref(node2,manager); //release
    
  return node; //could be NULL
}

/****************************************************************************************
 * apply case 2: node1->vtree in node2->vtree->left
 ****************************************************************************************/
 
//node2 is normalized for vtree (node2->vtree = vtree)
//node1 is normalized for a sub_vtree of vtree->left
//result is normalized for vtree unless it was trimmed
static
SddNode* sdd_apply_left(SddNode* node1, SddNode* node2, BoolOp op, Vtree* vtree, SddManager* manager, int limited) {
  assert(node1!=NULL && node2!=NULL);
  assert(NON_TRIVIAL(node1) && NON_TRIVIAL(node2));
  assert(node1->vtree->position < node2->vtree->position);
  assert(node2->vtree==vtree);
  assert(sdd_vtree_is_sub(node1->vtree,vtree->left));

  SddNode* node1_neg = sdd_negate(node1,manager);
  SddNode* n         = (op==CONJOIN? node1: node1_neg);
  
  c_ref(n,manager); c_ref(node2,manager); //protect
  
  SddNode* node;
  GET_node_from_partition_limited(node,vtree,manager,limited,{
    DECLARE_element(n->negation,ZERO(manager,op),vtree,manager);
    FOR_each_prime_sub_of_node(prime,sub,node2,{
      //vtree will not change, but vtree->left may change
      SddNode* new_prime = apply(prime,n,CONJOIN,manager,limited); //could be NULL
      ABORT_partition_if(new_prime==NULL);
      if(!IS_FALSE(new_prime)) DECLARE_element(new_prime,sub,vtree,manager);
    });
  });

  c_deref(n,manager); c_deref(node2,manager); //release
    
  return node; //could be NULL
}

/****************************************************************************************
 * apply case 3: node2->vtree in node1->vtree->right
 ****************************************************************************************/
 
//node1 is normalized for vtree (node1->vtree = vtree)
//node2 is normalized for a sub_vtree of vtree->right
//result is normalized for vtree unless it was trimmed
static
SddNode* sdd_apply_right(SddNode* node1, SddNode* node2, BoolOp op, Vtree* vtree, SddManager* manager, int limited) {
  assert(node1!=NULL && node2!=NULL);
  assert(NON_TRIVIAL(node1) && NON_TRIVIAL(node2));
  assert(node1->vtree->position < node2->vtree->position);
  assert(node1->vtree==vtree);
  assert(sdd_vtree_is_sub(node2->vtree,vtree->right));

  c_ref(node1,manager); c_ref(node2,manager); //protect
    
  SddNode* node;
  //NOTE: compression is possible here (i.e., apply may be called)
  GET_node_from_partition_limited(node,vtree,manager,limited,{
    FOR_each_prime_sub_of_node(prime,sub,node1,{
      //vtree will not change, but vtree->right may change
      SddNode* new_sub = apply(sub,node2,op,manager,limited); //could be NULL
      ABORT_partition_if(new_sub==NULL);
      DECLARE_element(prime,new_sub,vtree,manager);
    });
  });
    
  c_deref(node1,manager); c_deref(node2,manager); //release
    
  return node; //could be NULL
}

/****************************************************************************************
 * apply case 4: node1->vtree and node2->vtree incomparable
 ****************************************************************************************/
 
//node1 and node2 are normalized for incomparable vtree nodes
//node1 is before node2 in the vtree inorder
//vtree is lca of node1->vtree and node2->vtree
//result is normalized for vtree
static
SddNode* sdd_apply_incomparable(SddNode* node1, SddNode* node2, BoolOp op, Vtree* vtree, SddManager* manager, int limited) {
  assert(node1!=NULL && node2!=NULL);
  assert(NON_TRIVIAL(node1) && NON_TRIVIAL(node2));
  assert(node1->vtree->position < node2->vtree->position);
  assert(sdd_vtree_is_sub(node1->vtree,vtree));
  assert(sdd_vtree_is_sub(node2->vtree,vtree));
  assert(!sdd_vtree_is_sub(node1->vtree,node2->vtree));
  assert(!sdd_vtree_is_sub(node2->vtree,node1->vtree));

  SddNode* node1_neg = sdd_negate(node1,manager);
  
  SddNode* node1_sub = apply(node2,manager->true_sdd,op,manager,limited);
  assert(node1_sub!=NULL);
  
  SddNode* node1_neg_sub = apply(node2,manager->false_sdd,op,manager,limited);
  assert(node1_neg_sub!=NULL);

  SddNode* node;
  //NOTE: no compression is possible here (i.e., apply will not be called)
  GET_node_from_compressed_partition(node,vtree,manager,{
    DECLARE_compressed_element(node1,node1_sub,vtree,manager);
	DECLARE_compressed_element(node1_neg,node1_neg_sub,vtree,manager);
  });

  assert(node!=NULL);
  return node;
}

/****************************************************************************************
 * apply master: channel to one of the above cases
 ****************************************************************************************/

//checks whether the current apply call has no parent apply calls
int root_apply(SddManager* manager) {
  return manager->apply_depth==1;
}

//checks whether the current apply call has no limited, parent apply calls
static inline
int root_limited_apply(SddManager* manager) {
  return manager->limited_apply_depth==1;
}

//book keeping when starting an apply that could invoke vtree search
static inline
void prepare_for_vtree_search(Vtree* lca, SddManager* manager) {
  manager->auto_apply_vtree              = lca;
  manager->auto_apply_outside_live_size  = (manager->sdd_size-manager->dead_sdd_size);
  manager->auto_apply_outside_live_count = (manager->node_count-manager->dead_node_count);
  manager->auto_apply_outside_dead_count = manager->dead_node_count;
  FOR_each_internal_vtree_node(v,lca,{
    manager->auto_apply_outside_live_size  -= (v->sdd_size-v->dead_sdd_size);
    manager->auto_apply_outside_live_count -= (v->node_count-v->dead_node_count);
    manager->auto_apply_outside_dead_count -= v->dead_node_count;
  });
}

//not limited: vtree search possible
static
SddNode* u_apply(char apply_type, Vtree* lca, SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager) {
  if(manager->auto_gc_and_search_on && root_apply(manager)) prepare_for_vtree_search(lca,manager);
  SddNode* node = NULL;
  switch(apply_type) {
    case 'e': node = sdd_apply_equal(node1,node2,op,lca,manager,0); break;
    case 'l': node = sdd_apply_left(node1,node2,op,lca,manager,0); break;
    case 'r': node = sdd_apply_right(node1,node2,op,lca,manager,0); break;
    case 'i': node = sdd_apply_incomparable(node1,node2,op,lca,manager,0); break;
    default: assert(0); 
  }
  assert(node);  
  cache_computation(node1,node2,node,op,manager);
  if(manager->auto_gc_and_search_on && lca->var_count > 1) {
    sdd_ref(node,manager); //same as c_ref
    try_auto_gc_and_minimize(lca,manager);
    sdd_deref(node,manager); //same as c_deref
  }
  return node;
}

//limited: vtree search not possible
static
SddNode* l_apply(char apply_type, Vtree* lca, SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager) {
  ++manager->limited_apply_depth;
  if(root_limited_apply(manager)) start_apply_limits(manager);
  SddNode* node = NULL;
  switch(apply_type) {
    case 'e': node = sdd_apply_equal(node1,node2,op,lca,manager,1); break;
    case 'l': node = sdd_apply_left(node1,node2,op,lca,manager,1); break;
    case 'r': node = sdd_apply_right(node1,node2,op,lca,manager,1); break;
    case 'i': node = sdd_apply_incomparable(node1,node2,op,lca,manager,1); break;
    default: assert(0); 
  }
  if(node) {
    cache_computation(node1,node2,node,op,manager);
    //only place where limits are checked (but see search.c where result of check are examined)
    if(exceeded_limits(manager)) node = NULL; //failed
  }
  if(root_limited_apply(manager)) end_apply_limits(manager);
  --manager->limited_apply_depth;  
  return node;
}

//node1 and node2 are normalized for arbitrary vtrees
//node1->vtree and node2->vtree are sub_vtrees of vtree
//result is normalized for a sub_vtree of vtree
SddNode* apply(SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager, int limited) {
  assert(!apply_aborted(manager));
  assert(node1!=NULL && node2!=NULL);
  assert(!GC_NODE(node1));
  assert(!GC_NODE(node2));
       
  //base cases
  if(node1==node2) return node1;
  if(node1==node2->negation) return ZERO(manager,op); //same as ZERO(node2->vtree,op);
  if(IS_ZERO(node1,op) || IS_ZERO(node2,op)) return ZERO(manager,op);
  if(IS_ONE(node1,op)) return node2;
  if(IS_ONE(node2,op)) return node1;

  //check cache
  SddNode* node = lookup_computation(node1,node2,op,manager);
  if(node!=NULL) return node; //cache hit
  
  //cache miss: must recurse
  ++manager->apply_depth;
  
  ++manager->stats.apply_count; //all recursed applies
  if(root_apply(manager)) ++manager->stats.apply_count_top; //top-level recursed applies
             
  //order is required by cmp_vtrees and recursive apply calls
  if(node1->vtree->position > node2->vtree->position) SWAP(SddNode*,node1,node2);
  Vtree* lca      = NULL; //lowest common ancestor
  char apply_type = cmp_vtrees(&lca,node1->vtree,node2->vtree);
  //(ab) + (a~b) = a, which is why node->vtree!=lca in general

  node = limited? l_apply(apply_type,lca,node1,node2,op,manager):
                  u_apply(apply_type,lca,node1,node2,op,manager);
  
  --manager->apply_depth;
  return node;
}

/****************************************************************************************
 * special case of sdd_apply used by left/right rotations
 *
 * could be handled by sdd_apply but this saves some overhead as it is called a lot
 * (it will also not call vtree search)
 *
 * assumes:
 * --node1->vtree and node2->vtree are incomparable
 * --node1->vtree->position < node2->vtree->position 
 * --vtree contains node1->vtree and node2->vtree
 ****************************************************************************************/
 
//not limited
SddNode* sdd_conjoin_lr(SddNode* node1, SddNode* node2, Vtree* lca, SddManager* manager) {
  assert(!apply_aborted(manager));
  assert(node1!=NULL && node2!=NULL);
  assert(!GC_NODE(node1));
  assert(!GC_NODE(node2));
        
  if(IS_FALSE(node1) || IS_FALSE(node2)) return manager->false_sdd;
  if(IS_TRUE(node1)) return node2;
  if(IS_TRUE(node2)) return node1;
  
  assert(INTERNAL(lca));
  assert(sdd_vtree_is_sub(node1->vtree,lca->left));
  assert(sdd_vtree_is_sub(node2->vtree,lca->right));
  
  ++manager->apply_depth;
  ++manager->stats.apply_count; //this is considered a recursed apply
    
  SddNode* node = lookup_computation(node1,node2,CONJOIN,manager);
  if(node==NULL) { //no compression or trimming possible
    GET_node_from_compressed_partition(node,lca,manager,{
      DECLARE_compressed_element(node1,node2,lca,manager);
	  DECLARE_compressed_element(sdd_negate(node1,manager),manager->false_sdd,lca,manager);
    });
    cache_computation(node1,node2,node,CONJOIN,manager);
  }
  
  assert(node);
  --manager->apply_depth;
  return node;
}  


/****************************************************************************************
 * end
 ****************************************************************************************/
