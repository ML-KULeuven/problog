/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//vtree nodes use zero-based indexing
//vtree variables use one-based indexing

/****************************************************************************************
 * constructing vtree nodes
 ****************************************************************************************/

#define INITIALIZE_VTREE_FIELDS(vtree,left,right) {\
  vtree->parent = NULL;\
  vtree->nodes  = NULL;\
  /* count and size */\
  vtree->sdd_size        = 0;\
  vtree->dead_sdd_size   = 0;\
  vtree->node_count      = 0;\
  vtree->dead_node_count = 0;\
  /* bits */\
  vtree->some_X_constrained_vars = 0;\
  vtree->all_vars_in_sdd = 0;\
  vtree->no_var_in_sdd   = 0;\
  vtree->bit             = 0;\
  /* user fields */\
  vtree->user_data         = NULL;\
  vtree->user_search_state = NULL;\
  vtree->user_bit          = 0;\
  /* auto minimize mode */\
  vtree->auto_last_search_live_size = 0;\
  /* vtree search state (library)*/\
  VtreeSearchState* state = (VtreeSearchState*)malloc(sizeof(VtreeSearchState));\
  state->previous_left       = left;\
  state->previous_right      = right;\
  state->previous_size       = 0;\
  state->previous_count      = 0;\
  state->fold                = 0;\
  vtree->search_state = state;\
}

//create a new leaf vtree node corresponding to var
Vtree* new_leaf_vtree(SddLiteral var) {

  Vtree* leaf;
  MALLOC(leaf,Vtree,"new_leaf_vtree");
  
  //vars
  leaf->var       = var;
  leaf->var_count = 1;
  //children
  leaf->left  = NULL;
  leaf->right = NULL;
  //other fields
  INITIALIZE_VTREE_FIELDS(leaf,NULL,NULL);
  
  return leaf;
}  

//create a new internal vtree node with given children
Vtree* new_internal_vtree(Vtree* left_child, Vtree* right_child) {

  Vtree* internal;
  MALLOC(internal,Vtree,"new_internal_vtree");
  
  //vars
  internal->var_count = left_child->var_count+right_child->var_count;
  //children and parents
  internal->left      = left_child;
  internal->right     = right_child;
  left_child->parent  = internal;
  right_child->parent = internal;
  //other fields
  INITIALIZE_VTREE_FIELDS(internal,left_child,right_child);

  return internal;
}

/****************************************************************************************
 * freeing vtrees
 ****************************************************************************************/

//freeing vtree and associated structures
void sdd_vtree_free(Vtree* vtree) {
  if(INTERNAL(vtree)) {
    sdd_vtree_free(vtree->left);
    sdd_vtree_free(vtree->right);
  }
  //free vnode
  free(vtree->search_state);
  free(vtree);
}

/****************************************************************************************
 * basic tests and lookups
 ****************************************************************************************/

int sdd_vtree_is_leaf(const Vtree* vtree) {
  return LEAF(vtree);
}

//returns the leaf vtree which hosts var
//var is an integer > 0
Vtree* sdd_manager_vtree_of_var(const SddLiteral var, const SddManager* manager) {
  return manager->leaf_vtrees[var];
}

Vtree* sibling(Vtree* vtree) {
  assert(vtree->parent!=NULL);
  if(vtree==vtree->parent->left) return vtree->parent->right;
  else return vtree->parent->left;
}

//left-most leaf vtree 
Vtree* first_leaf_vtree(Vtree* vtree) {
  while(INTERNAL(vtree)) vtree = vtree->left;
  return vtree;
}

//right-most leaf vtree 
Vtree* last_leaf_vtree(Vtree* vtree) {
  while(INTERNAL(vtree)) vtree = vtree->right;
  return vtree;
}

//the location of the pointer for vtree
Vtree** sdd_vtree_location(Vtree* vtree, SddManager* manager) {
  Vtree* parent = vtree->parent;
  if(parent!=NULL) return (vtree==parent->left? &(parent->left): &(parent->right));
  else return &(manager->vtree);
}
 

/****************************************************************************************
 * adding/removing/moving variables will change positions and variable counts
 ****************************************************************************************/

//sets the positions and var count of each vtree node
void set_vtree_properties(Vtree* vtree) {
  void set_sub_vtree_properties(Vtree* vtree, SddLiteral start_position);
  set_sub_vtree_properties(vtree,0); 
  assert(!FULL_DEBUG || verify_vtree_properties(vtree));
}

void set_sub_vtree_properties(Vtree* vtree, SddLiteral start_position) {
  if(LEAF(vtree)) {
    vtree->var_count = 1;
    vtree->next      = vtree->prev = NULL;
    vtree->first     = vtree->last = vtree;
    vtree->position  = start_position;
  }
  else {
    Vtree* left  = vtree->left;
    Vtree* right = vtree->right;
    
    set_sub_vtree_properties(left,start_position);
    set_sub_vtree_properties(right,2+left->last->position);
    
    //linked list
    vtree->next  = right->first;
    vtree->prev  = left->last;
    left->last->next   = vtree;
    right->first->prev = vtree;
    vtree->first = left->first;
    vtree->last  = right->last;
    
    //var count and position
    vtree->var_count = left->var_count+right->var_count;
    vtree->position  = 1+left->last->position;
  }
}

/****************************************************************************************
 * maintaining vtree positions after its children have been swapped
 ****************************************************************************************/
 
void update_positions_after_swap(Vtree* vtree) {
  assert(INTERNAL(vtree));
  Vtree* left  = vtree->left;
  Vtree* right = vtree->right;
  SddLiteral start_position = right->first->position; //was left->first->position before swap
  //positions
  vtree->position    = start_position+(2*left->var_count-1);
  SddLiteral loffset = start_position-left->first->position;
  SddLiteral roffset = 1+vtree->position-right->first->position;
  //update
  FOR_each_vtree_node(v,left,v->position += loffset);
  FOR_each_vtree_node(v,right,v->position += roffset);
  assert(!FULL_DEBUG || verify_vtree_properties(vtree));
}

/****************************************************************************************
 * end
 ****************************************************************************************/
