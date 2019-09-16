/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/memory.c
SddNode* new_sdd_node(SddNodeType type, SddNodeSize size, Vtree* vtree, SddManager* manager);
void free_elements(SddNodeSize size, SddElement* elements, SddManager* manager);

//basic/hash.c
void insert_sdd_node(SddNode* node, SddHash* hash, SddManager* manager);
void remove_sdd_node(SddNode* node, SddHash* hash, SddManager* manager);

/****************************************************************************************
 * freeing an sdd node structure
 ****************************************************************************************/
 
void free_sdd_node(SddNode* node, SddManager* manager) {
  if(node->type==DECOMPOSITION) {
    free_elements(node->size,ELEMENTS_OF(node),manager);
    free(node);
  }
  else free(node); //terminal sdd node
}

/****************************************************************************************
 * updating counts and sizes
 ****************************************************************************************/

//node was just added or removed from unique table 
static inline
void update_counts_and_sizes_after_unique_table_change(SddNode* node, SddManager* manager) {
  assert(node->type==DECOMPOSITION);
    
  int inc = node->in_unique_table? 1: -1;
  //added node:   inc= +1
  //removed node: inc= -1

  Vtree* vtree = node->vtree;
  SddSize size = node->size;
  
  //total counts and sizes
  manager->node_count += inc;
  manager->sdd_size   += inc*size;
  vtree->node_count   += inc;
  vtree->sdd_size     += inc*size;
  
  //dead counts and sizes
  //live sizes are not maintained: computed from total/dead
  if(node->ref_count==0) { //dead node
    manager->dead_node_count += inc;
    manager->dead_sdd_size   += inc*size;
    vtree->dead_node_count   += inc;
    vtree->dead_sdd_size     += inc*size;
  }
}

/****************************************************************************************
 * maintaining parent counts for sdd nodes
 *
 * sdd node n is a parent of sdd node m iff m is a prime or sub of node n
 ****************************************************************************************/

void declare_acquired_parent(SddNode* node, SddManager* manager) {
  assert(IS_DECOMPOSITION(node));
  FOR_each_prime_sub_of_node(prime,sub,node,{
    ++prime->parent_count;
    ++sub->parent_count;
  });
}

void declare_lost_parent(SddNode* node, SddManager* manager) {
  assert(IS_DECOMPOSITION(node));
  FOR_each_prime_sub_of_node(prime,sub,node,{
    assert(prime->parent_count && sub->parent_count);
    --prime->parent_count;
    --sub->parent_count;
  });
}

/****************************************************************************************
 * inserting and removing sdd nodes from the unique table
 ****************************************************************************************/

void insert_in_unique_table(SddNode* node, SddManager* manager) {
  assert(node->type==DECOMPOSITION);
  assert(node->in_unique_table==0);
  
  //insert in hash table
  insert_sdd_node(node,manager->unique_nodes,manager);
  //insert in nodes of vtree
  Vtree* vtree = node->vtree;
  if(vtree->nodes) vtree->nodes->vtree_prev = &(node->vtree_next);
  node->vtree_next = vtree->nodes;
  node->vtree_prev = &(vtree->nodes);
  vtree->nodes = node;
  //update counts and sizes
  node->in_unique_table = 1; //before updating counts and sizes
  update_counts_and_sizes_after_unique_table_change(node,manager);
}

void remove_from_unique_table(SddNode* node, SddManager* manager) {
  assert(node->type==DECOMPOSITION);
  assert(node->in_unique_table==1);
  
  //remove from hash table
  remove_sdd_node(node,manager->unique_nodes,manager);
  //remove from nodes of vtree
  if(node->vtree_next) node->vtree_next->vtree_prev = node->vtree_prev;
  *(node->vtree_prev) = node->vtree_next;
  //update counts and sizes
  node->in_unique_table = 0; //before updating counts and sizes
  update_counts_and_sizes_after_unique_table_change(node,manager);
}

/****************************************************************************************
 * constructing decomposition sdd nodes
 ****************************************************************************************/
 
//the elements array is not used by the newly constructed node: new array is created
SddNode* construct_decomposition_sdd_node(SddNodeSize size, SddElement* elements, Vtree* vtree, SddManager* manager) {
  assert(elements_sorted_and_compressed(size,elements));
  assert(vtree==lca_of_compressed_elements(size,elements,manager));
  
  //allocate node (and its elements)
  SddNode* node = new_sdd_node(DECOMPOSITION,size,vtree,manager);
  //copy elements into node
  memcpy(ELEMENTS_OF(node),elements,size*sizeof(SddElement));
  //insert in unique table
  insert_in_unique_table(node,manager);
  //primes and subs of node have acquired a new parent
  declare_acquired_parent(node,manager);

  return node;
}


/****************************************************************************************
 * constructing literal sdd nodes
 * 
 * there are two literal sdds for each variable
 ****************************************************************************************/

//this is called only when constructing manager
SddNode* construct_literal_sdd_node(SddLiteral literal, Vtree* vtree, SddManager* manager) {
  SddNode* node = new_sdd_node(LITERAL,0,vtree,manager);
  LITERAL_OF(node) = literal;
  return node;
}

/****************************************************************************************
 * constructing true and false sdd nodes for manager
 *
 * there is one true sdd and one false sdd per manager
 ****************************************************************************************/
 
//this is called only when constructing manager
SddNode* construct_true_sdd_node(SddManager* manager) {
  SddNode* node = new_sdd_node(TRUE,0,NULL,manager);
  return node;
}

//this is called only when constructing manager
SddNode* construct_false_sdd_node(SddManager* manager) {
  SddNode* node = new_sdd_node(FALSE,0,NULL,manager);
  return node;
}

/*****************************************************************************************
 * end
 ****************************************************************************************/
