/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

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
 
/****************************************************************************************
 * if there was no change in a vtree, treat it as a leaf node as far as search is
 * concerned (i.e., do not try to change the vtree structure) 
 ****************************************************************************************/
 
int is_virtual_leaf_vtree(Vtree* vtree) {
  VtreeSearchState* state = vtree->search_state;
  return LEAF(vtree) || state->fold;
}
  
/****************************************************************************************
 * returns the largest subtree which involved a change 
 * 
 * as a side effect:
 * --updates the "fold" field for the search state
 * --updates the "state" of each vtree node (size, parent, left child, right child)
 *
 ****************************************************************************************/

//vtree with no local changes will become a virtual leaf only when all its variables are used
    
Vtree* update_vtree_change(Vtree* vtree, SddManager* manager) {
  VtreeSearchState* state = vtree->search_state;

  if(LEAF(vtree)) {
    state->fold = sdd_manager_is_var_used(vtree->var,manager);
    //when fold==1, does not matter what is returned (vtree or NULL)
    //when fold==0:
    // --passing  NULL: a vtree with no local changes will be pruned only if it has at most one fold=0 var
    // --passing vtree: a vtree with no local changes will be pruned only if it has no fold=0 vars 
    //                  (i.e., when all its variables are used)
    return vtree;
  }
    
  //internal vtree node
  Vtree* left   = sdd_vtree_left(vtree);
  Vtree* right  = sdd_vtree_right(vtree);
  
  Vtree* pruned_left  = update_vtree_change(left,manager);
  Vtree* pruned_right = update_vtree_change(right,manager);

  VtreeSearchState* left_state  = left->search_state;
  VtreeSearchState* right_state = right->search_state;
    
  SddSize cur_size   = sdd_vtree_live_size_at(vtree);
  SddSize cur_count  = sdd_vtree_live_count_at(vtree);
   
  int unchanged = (cur_size==state->previous_size) &&
                  (cur_count==state->previous_count) &&
                  (left==state->previous_left) && 
                  (right==state->previous_right);                           

  if(!unchanged) {
    //state has changed, save new one
    state->previous_size   = cur_size;
    state->previous_count  = cur_count;
    state->previous_left   = left;
    state->previous_right  = right;
  }    
  
  state->fold  = (unchanged && left_state->fold && right_state->fold);
 
  //--folded   vtree: NULL (prune it)
  //--unfolded vtree:
  //  --local change: vtree (don't prune it)
  //  --no local change: 
  //    --folded   left, unfolded right: pruned_right (can be NULL)
  //    --unfolded left, folded   right: pruned_left  (can be NULL)
  //    --unfolded left, unfolded right: vtree
  //    --folded   left, folded   right: (impossible case)
  //
  if(state->fold) return NULL;
  else if(!unchanged) return vtree; 
  else if(left_state->fold && !right_state->fold) return pruned_right; 
  else if(!left_state->fold && right_state->fold) return pruned_left;
  else return vtree;
}

//same as above function, except that it also updates the state of vtree's parent
Vtree* update_vtree_change_p(Vtree* vtree, SddManager* manager) {
  Vtree* parent = sdd_vtree_parent(vtree);
  if(parent!=NULL) {
    VtreeSearchState* state = parent->search_state;
    if(sdd_vtree_left(parent)==vtree) state->previous_left = vtree;
    else state->previous_right = vtree;
  }
  return update_vtree_change(vtree,manager);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
