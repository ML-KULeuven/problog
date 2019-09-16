/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/memory.c
void free_elements(SddNodeSize size, SddElement* elements, SddManager* manager);

//basic/nodes.c
void declare_acquired_parent(SddNode* node, SddManager* manager);
void declare_lost_parent(SddNode* node, SddManager* manager);

/****************************************************************************************
 * replaces the elements and vtree of a (live) decomposition node 
 *
 * does not deal with the insertion/deletion of the node into a unique table
 ****************************************************************************************/

//assumes node is live
//if replacement is reversible, then save current elements (instead of freeing them)
//if replacement is irreversible, then free current elements
void replace_node(int reversible, SddNode* node, SddNodeSize new_size, SddElement* new_elements, Vtree* new_vtree, SddManager* manager) {
  assert(node->ref_count); //live
  assert(node->type==DECOMPOSITION); //not terminal
  
  SddSize cur_size         = node->size;
  SddElement* cur_elements = ELEMENTS_OF(node);
  
  SddRefCount ref_count = node->ref_count; //save ref_count
  while(node->ref_count) sdd_deref(node,manager); //dereference node
  //node is now dead
  
  declare_lost_parent(node,manager); //cur_elements lost node as a parent
      
  //replace vtree, size and elements
  node->vtree       = new_vtree;
  node->size        = new_size;
  ELEMENTS_OF(node) = new_elements;
  sort_compressed_elements(new_size,new_elements); //elements must be sorted
  
  declare_acquired_parent(node,manager); //new_elements acquired node as a parent
  
  while(ref_count--) sdd_ref(node,manager); //re-establish references
  //node is now live again
   
  //save or free current elements
  if(reversible) {
    node->replaced       = 1; //replacement reversible
    node->saved_size     = cur_size;
    node->saved_elements = cur_elements;
  }
  else {
    node->replaced       = 0; //replacement is irreversible
    node->saved_size     = 0;
    node->saved_elements = NULL;
    free_elements(cur_size,cur_elements,manager);
  }
}

//confirms a reversible replacement: free saved elements and insert in unique table
//once replacement is confirmed, it cannot be reversed
void confirm_node_replacement(SddNode* node, SddManager* manager) {
  assert(node->replaced); //was replaced
  assert(node->saved_elements); //has saved elements
  assert(node->ref_count); //live
  assert(node->type==DECOMPOSITION); //not terminal

  node->replaced = 0; //no longer reversible
  free_elements(node->saved_size,node->saved_elements,manager);

  node->saved_size     = 0;
  node->saved_elements = NULL;
}

//reverse a replacement: recover saved elements and insert in unique table
void reverse_node_replacement(SddNode* node, Vtree* vtree, SddManager* manager) {
  assert(node->replaced); //was replaced
  assert(node->saved_elements); //has saved elements
  assert(node->ref_count); //live
  assert(node->type==DECOMPOSITION); //not terminal
 
  int reversible = 0; //replacement is not reversible (current elements will be freed)
  replace_node(reversible,node,node->saved_size,node->saved_elements,vtree,manager);
}

/*****************************************************************************************
 * end
 ****************************************************************************************/
