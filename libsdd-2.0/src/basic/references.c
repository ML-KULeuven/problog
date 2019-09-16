/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

/****************************************************************************************
 * updating counts and sizes
 ****************************************************************************************/
  
static inline
void update_counts_and_sizes_after_livelihood_change(SddNode* node, SddManager* manager) {
  assert(node->ref_count==0 || node->ref_count==1);
  assert(node->type==DECOMPOSITION);
  if(node->in_unique_table==0) return;
  
  Vtree* vtree = node->vtree;
  SddSize size = node->size;
  int inc      = node->ref_count==1? -1: 1;
  //live->dead: inc= +1
  //dead->live: inc= -1
  
  //only dead counts and sizes need to be updated
  manager->dead_node_count += inc;
  manager->dead_sdd_size   += inc*size;
  vtree->dead_node_count   += inc;
  vtree->dead_sdd_size     += inc*size;
}

/****************************************************************************************
 * high level
 ****************************************************************************************/

//ref_count of terminal sdds is 0
SddRefCount sdd_ref_count(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"reference_count");
  if(IS_DECOMPOSITION(node)) return node->ref_count;
  else {
    assert(node->ref_count==0);
    return 0; 
  }
}

//returns node
SddNode* sdd_ref(SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_ref");
  
  if(IS_DECOMPOSITION(node) && ++node->ref_count==1) { //node was dead and became live
    update_counts_and_sizes_after_livelihood_change(node,manager);
    FOR_each_prime_sub_of_node(prime,sub,node,{
      sdd_ref(prime,manager);
      sdd_ref(sub,manager);
    });
  }
  
  return node;
}

//returns node
SddNode* sdd_deref(SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_deref");
  CHECK_ERROR(IS_DECOMPOSITION(node) && node->ref_count==0,ERR_MSG_DEREF,"sdd_deref");

  if(IS_DECOMPOSITION(node) && --node->ref_count==0) { //node was live and became dead
    update_counts_and_sizes_after_livelihood_change(node,manager);
    FOR_each_prime_sub_of_node(prime,sub,node,{
      sdd_deref(prime,manager);
      sdd_deref(sub,manager);
    });
  }
 
  return node;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
