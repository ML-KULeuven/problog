/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 1.1.1, January 31, 2014
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sddapi.h"
#include "compiler.h"

/****************************************************************************************
 * The dynamic vtree search algorithms tries to be selective on which parts of a
 * vtree it applies itself to. For this, it tries to identify parts of the vtree that
 * were "unchanged" since the last search (iteration) and avoids these parts.
 *
 * The definition of "change" is heuristic and is based on three factors: the size of
 * sdd nodes normalized for the vtree node, in addition to the identity of its left and 
 * right children. If none of these have changed since the last search iteration, the 
 * vtree node is considered unchanged.
 *
 * To implement this strategy, once must maintain a "state" with each vtree node to keep
 * track of the previous size, left child, and right child. This is done using the
 * VtreeSearchState structure in the code below. The field vtree->fold is set to 1 iff
 * all the nodes in vtree are unchanged. 
 *
 * Note: a leaf vtree node is considered unchanged iff its variable is being used 
 * by some decomposition sdd node.
 *
 * The search algorithm implements two strategies based on the notion of change:
 *
 * 1. The algorithm will apply itself to the smallest subtree with a changed vtree node
 * 2. The algorithm will treat a subtree as a (virtual) leaf when all nodes in the
 *    vtree are unchanged
 *
 * The state of a vtree node is updated after each pass of the search algorithm. 
 ****************************************************************************************/
 
 
// local search state, stored at each vtree node

typedef struct {
  Vtree* previous_left;
  Vtree* previous_right;
  SddSize previous_size;
  unsigned fold:1;
} VtreeSearchState;

/****************************************************************************************
 * vtree state: initialization and freeing
 ****************************************************************************************/

// function for initializing vtree state at a node
void* mip_initialize_node_search_state(Vtree* vtree) {
  VtreeSearchState* state = (VtreeSearchState*)malloc(sizeof(VtreeSearchState));
  state->previous_left  = NULL;
  state->previous_right = NULL;
  state->previous_size  = 0;
  state->fold           = 0;
  return (void*)state;
}

// recursively initialize the search state at each vtree node
void mip_initialize_vtree_search_state(Vtree* vtree) {
  void* search_state = mip_initialize_node_search_state(vtree);
  sdd_vtree_set_search_state(search_state,vtree);
  if(!sdd_vtree_is_leaf(vtree)) {
    mip_initialize_vtree_search_state(sdd_vtree_left(vtree));
    mip_initialize_vtree_search_state(sdd_vtree_right(vtree));
  }
}

void mip_initialize_manager_search_state(SddManager* manager) {
  mip_initialize_vtree_search_state(sdd_manager_vtree(manager));
}

// recursively free the search state at each vtree node
void mip_free_vtree_search_state(Vtree* vtree) {
  void* search_state = sdd_vtree_search_state(vtree);
  free(search_state);
  sdd_vtree_set_search_state(NULL,vtree);
  if(!sdd_vtree_is_leaf(vtree)) {
    mip_free_vtree_search_state(sdd_vtree_left(vtree));
    mip_free_vtree_search_state(sdd_vtree_right(vtree));
  }
}

void mip_free_manager_search_state(SddManager* manager) {
  mip_free_vtree_search_state(sdd_manager_vtree(manager));
}


/****************************************************************************************
 * if there was no change in a vtree, treat it as a leaf node as far as search is
 * concerned (i.e., do not try to change the vtree structure) 
 ****************************************************************************************/
 
int mip_is_virtual_leaf_vtree(Vtree* vtree) {
  VtreeSearchState* state = sdd_vtree_search_state(vtree);
  return sdd_vtree_is_leaf(vtree) || state->fold;
}

/****************************************************************************************
 * returns the largest subtree which involved a change 
 * 
 * as a side effect:
 * --updates the "unchanged" field for the search state
 * --updates the "state" of each vtree node (size, left child, right child)
 ****************************************************************************************/

Vtree* mip_update_vtree_change(Vtree* vtree, SddManager* manager) {
  VtreeSearchState* state = sdd_vtree_search_state(vtree);
  
  if(sdd_vtree_is_leaf(vtree)) {
    SddLiteral var = sdd_vtree_var(vtree);
    state->fold    = sdd_manager_is_var_used(var,manager);
    return vtree;
  }
  else {
    Vtree* left  = sdd_vtree_left(vtree);
    Vtree* right = sdd_vtree_right(vtree);

    Vtree* pruned_left  = mip_update_vtree_change(left,manager);
    Vtree* pruned_right = mip_update_vtree_change(right,manager);
   
    VtreeSearchState* left_state = sdd_vtree_search_state(left);
    VtreeSearchState* right_state = sdd_vtree_search_state(right);
    
    SddSize cur_size  = sdd_vtree_live_size_at(vtree);
    
    int unchanged = (cur_size==state->previous_size) && 
                    (left==state->previous_left) &&
                    (right==state->previous_right);
    
    if(!unchanged) {
      //save state of vtree node
      state->previous_size  = cur_size;
      state->previous_left  = left;
      state->previous_right = right;
    }
    
    state->fold = (unchanged && left_state->fold && right_state->fold);
    
    if(state->fold) return NULL; //this vtree did not change
    else if(unchanged==0) return vtree; //this vtree changed at least locally
    else if(left_state->fold && !right_state->fold) return pruned_right; //only right changed
    else if(!left_state->fold && right_state->fold) return pruned_left; //only left changed
    else return vtree; //both left and right changed
  }
}


//same as above function, except that it also updates the state of vtree's parent
Vtree* mip_update_vtree_change_p(Vtree* vtree, SddManager* manager) {
  //the parent of vtree may have a changed child (when vtree is a new root)
  Vtree* parent = sdd_vtree_parent(vtree);
  if(parent!=NULL) {
    VtreeSearchState* state = sdd_vtree_search_state(parent);
    if(sdd_vtree_left(parent)==vtree) state->previous_left = vtree;
    else state->previous_right = vtree;
  }
  return mip_update_vtree_change(vtree,manager);
}


/****************************************************************************************
 * end
 ****************************************************************************************/
