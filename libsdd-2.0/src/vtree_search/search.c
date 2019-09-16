/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//vtree_operations/limits.c
void start_search_limits(SddManager* manager);
void end_search_limits(SddManager* manager);
int search_aborted(SddManager* manager);
void start_fragment_limits(SddManager* manager);
void end_fragment_limits(SddManager* manager);
int fragment_aborted(SddManager* manager);
void sdd_manager_init_vtree_size_limit(Vtree* vtree, SddManager* manager);
void sdd_manager_update_vtree_size_limit(SddManager* manager);

//search_vtree/state.c
int is_virtual_leaf_vtree(Vtree* vtree);
Vtree* update_vtree_change(Vtree* vtree, SddManager* manager);
Vtree* update_vtree_change_p(Vtree* vtree, SddManager* manager);

/****************************************************************************************
 * Local search algorithm for finding a good vtree: one with a minimal sdd size.
 *
 * See fragments/construct.c for the definition of fragments (left-linear & right-linear).
 *
 * The algorithm works as follows:
 *
 * --It makes multiple bottom-up passes through the vtree, until the reduction in the 
 *   sdd size is small enough (see include/parameter.h).
 *
 * --In each pass, at internal node n, it considers both left-linear and right-linear 
 *   fragments with node n as root (when such fragments exit).
 *
 * --It then considers all 12 states of each existing fragment, whenever this is feasible
 *   within the given time and size limits (see include/paramater.h).
 *
 * --It then moves the vtree to the best state found (potentially, 24 states will be
 *   considered). The best state is the one with the smallest sdd size (see tie breakers).
 * 
 * The algorithm is guaranteed to improve the sdd size monotonically through its passes.
 *
 ***************************************************************************************/

/****************************************************************************************
 * quality measures
 ****************************************************************************************/

SddSize best_size;
SddSize best_count;
SddSize best_balance;
  
static inline
SddSize balance(Vtree* vtree, SddManager* manager) {
  SddLiteral left_count  = sdd_vtree_var_count(sdd_vtree_left(vtree));
  SddLiteral right_count = sdd_vtree_var_count(sdd_vtree_right(vtree));
  return labs(left_count - right_count);
}

/****************************************************************************************
 * Finding the best state of a vtree fragment: potentially considering 12 states.
 *
 * The fragment could be left-linear r=(c=(. .) .) or right-linear r=(. c=(. .))
 ****************************************************************************************/

//updates index (0...11) and direction ('f' 'b') of best fragment state that beats current best
static
void best_fragment_state(int* best_state, char* best_direction, VtreeFragment* fragment, int limited) {
  
  SddManager* manager = fragment->manager;
  
  if(limited) start_fragment_limits(manager);
  
  char cur_direction      = 'f'; //start by moving forward through states
  int visited_state_count = 0; //number of fragment states examined

  //handling constrained vtrees
  Vtree* X_node = NULL;
  if(fragment->root->some_X_constrained_vars) { //otherwise all states are ok
    Vtree* v = fragment->root->right;
    while(INTERNAL(v) && v->some_X_constrained_vars) v = v->right;
    if(v->some_X_constrained_vars==0) {
      if(fragment->type=='l') X_node = fragment->root->right;
      else if(v==fragment->child) return; //don't search this fragment
      else X_node = fragment->child->right;
    }
  }
  
  while(visited_state_count < 12) { //try cycling through all 12 states
  
    //go to neighboring state
    if(vtree_fragment_next(cur_direction,fragment,limited)) { //at neighbor
      ++visited_state_count;
      
      //handling constrained vtrees
      if(X_node && ((fragment->type=='l' && fragment->root->right  != X_node) ||
                    (fragment->type=='r' && fragment->child->right != X_node))) continue;

      //see if current state is better than best so far  
      Vtree* cur_root     = vtree_fragment_root(fragment);
      SddSize cur_size    = sdd_manager_live_size(manager);
      SddSize cur_count   = sdd_manager_live_count(manager);
      SddSize cur_balance = balance(cur_root,manager);
  
      int better = (cur_size<best_size) || 
                   (cur_size==best_size && (cur_count<best_count || cur_balance<best_balance));
                   
      if(better) {
        //save found state and its details
        best_size       = cur_size;
        best_count      = cur_count;
        best_balance    = cur_balance;
        *best_state     = vtree_fragment_state(fragment);
        *best_direction = cur_direction;
        if(limited) sdd_manager_update_vtree_size_limit(manager); //new baseline for size limits
      }
    }
    else { //failed moving to neighbor
      assert(limited);
      if(FRAGMENT_SEARCH_BACKWARD && cur_direction=='f' && !fragment_aborted(manager) && !search_aborted(manager)) { 
        //failure due to aborting a vtree operation, so try searching backward
        vtree_fragment_rewind(fragment); //back to initial state 0 
        cur_direction = 'b'; //try searching backward
      }
      else break; //done with this fragment
    }
  
  } 
  
  if(vtree_fragment_state(fragment)!=0) vtree_fragment_rewind(fragment); //back to initial state

  ++manager->fragment_count;
  if(*best_state!=-1) { //found a better state
    ++manager->successful_fragment_count; 
    if(*best_direction=='b') ++manager->backward_successful_fragment_count;
  }
  
  if(visited_state_count==12) { //visited all 12 states
    ++manager->completed_fragment_count;
    if(*best_state!=-1) ++manager->successful_completed_fragment_count;
    if(cur_direction=='b') ++manager->backward_completed_fragment_count;
  }
  
  if(limited) end_fragment_limits(manager);
}

/****************************************************************************************
 * Determining whether left-linear and right-linear fragments exist
 *
 * A left-linear fragment has the form  r=(c=(. .) .) 
 * A right-linear fragment has the form r=(. c=(. .))
 ****************************************************************************************/

//there is a left-linear fragment with vtree as root in case vtree->left is to be
//treated as an internal node
static inline
int is_ll_fragment(Vtree* vtree) {
  return !is_virtual_leaf_vtree(vtree->left);
}

//there is a right-linear fragment with vtree as root in case vtree->right is to be
//treated as an internal node
static inline
int is_rl_fragment(Vtree* vtree) {
  return !is_virtual_leaf_vtree(vtree->right);
}

/****************************************************************************************
 * Finding the best state of fragments rooted at vtree: potentially examining 24 state
 * (12 for each fragment type)
 ****************************************************************************************/

//returns root of best vtree found
static
Vtree* best_local_state(Vtree* root, SddManager* manager, int limited) {

  if(limited && search_aborted(manager)) return root; //search aborted
   
  //initialize quality measures (global variables)
  best_size    = sdd_manager_live_size(manager);
  best_count   = sdd_manager_live_count(manager);
  best_balance = balance(root,manager);
  
  //find best state of rl fragment (if any)
  VtreeFragment* fragment_rl = NULL;
  int best_state_rl          = -1; //none
  char best_direction_rl     = ' '; //undefined
 
  //find best state of ll fragment (if any)
  VtreeFragment* fragment_ll = NULL;
  int best_state_ll          = -1; //none
  char best_direction_ll     = ' '; //undefined
     
  if(is_rl_fragment(root)) {
    fragment_rl = vtree_fragment_new(root,root->right,manager);
    best_fragment_state(&best_state_rl,&best_direction_rl,fragment_rl,limited);  
    if(limited && search_aborted(manager)) {
      vtree_fragment_free(fragment_rl);
      return root; //search aborted
    }
  }
   
  if(is_ll_fragment(root)) {
    fragment_ll = vtree_fragment_new(root,root->left,manager);
    best_fragment_state(&best_state_ll,&best_direction_ll,fragment_ll,limited);
    if(limited && search_aborted(manager)) {
      vtree_fragment_free(fragment_ll);
      if(fragment_rl) vtree_fragment_free(fragment_rl);
      return root; //search aborted
    }
  }
 
  //goto the best state found (following order is CRITICAL)
  if(best_state_ll!=-1) {
    root = vtree_fragment_goto(best_state_ll,best_direction_ll,fragment_ll);
  }
  else if(best_state_rl!=-1) {
    root = vtree_fragment_goto(best_state_rl,best_direction_rl,fragment_rl);
  }
  
  if(fragment_ll) vtree_fragment_free(fragment_ll);
  if(fragment_rl) vtree_fragment_free(fragment_rl);
  
  assert(best_size==sdd_manager_live_size(manager));
  assert(best_count==sdd_manager_live_count(manager));
  assert(best_balance==balance(root,manager));
  
  return root;
}

/****************************************************************************************
 * applying a bottom-up pass of the local search algorithm
 ****************************************************************************************/

//returns root of best vtree found
static
Vtree* local_search_pass(Vtree* vtree, SddManager* manager, int limited) {
  if(is_virtual_leaf_vtree(vtree)) return vtree;
  
  local_search_pass(vtree->left,manager,limited);
  local_search_pass(vtree->right,manager,limited);
    
  return best_local_state(vtree,manager,limited);
}

/****************************************************************************************
 * local vtree search algorithm
 ****************************************************************************************/

//returns percentage reduction in size
static
float size_reduction(SddSize prev_size, SddSize cur_size) {
  assert(cur_size <= prev_size);
  if(prev_size==0) {
    assert(cur_size==0);
    return 0;
  }
  else return 100.0*(prev_size-cur_size)/prev_size;
}

//returns new root
static
Vtree* sdd_vtree_minimize_limited_flag(Vtree* vtree, SddManager* manager, int limited) {
  assert(manager->auto_vtree_search_on==0);
  if(LEAF(vtree)) return vtree;

  manager->auto_vtree_search_on = 1;
  
  sdd_vtree_garbage_collect(vtree,manager); //local garbage collection 

  //identify the subtree to minimize (heuristic)
  Vtree* subtree = update_vtree_change(vtree,manager); //after gc
  if(subtree==NULL) {
    manager->auto_vtree_search_on = 0;
    return vtree;
  }
  
  Vtree** root_location = sdd_vtree_location(vtree,manager); //remember root location
  SddSize init_size     = sdd_vtree_live_size(subtree);
  SddSize out_size      = sdd_manager_live_size(manager)-init_size;
  float threshold       = manager->vtree_ops.convergence_threshold;
  int iterations        = 0;
  float reduction;
    
  if(limited) {
    start_search_limits(manager);
    sdd_manager_init_vtree_size_limit(subtree,manager); //baseline for size limits (subtree better than vtree)
  }

  assert(verify_X_constrained(manager->vtree));
  
  do { //pass
    SddSize prev_size        = sdd_vtree_live_size(subtree);    
    subtree                  = local_search_pass(subtree,manager,limited); //possibly new root
    SddSize cur_size         = sdd_vtree_live_size(subtree);
    reduction                = size_reduction(prev_size,cur_size);
    subtree                  = update_vtree_change_p(subtree,manager); //identify new subtree to minimize next
    //update_vtree_change_p also updates parent's state
    ++iterations;
  }
  while(!(limited && search_aborted(manager)) && subtree!=NULL && reduction>threshold);
  
  assert(verify_X_constrained(manager->vtree));

  if(manager->auto_gc_and_search_on) {
    SddSize final_size    = sdd_manager_live_size(manager)-out_size;
    float total_reduction = size_reduction(init_size,final_size);
    manager->auto_search_iteration_count += iterations;
    manager->auto_search_reduction_sum   += total_reduction;
  }
  
  manager->auto_vtree_search_on = 0;
  if(limited) end_search_limits(manager);
  assert(!FULL_DEBUG || verify_gc(*root_location,manager));
  return *root_location; //root may have changed due to rotations
}

//unlimited
Vtree* sdd_vtree_minimize(Vtree* vtree, SddManager* manager) {
  return sdd_vtree_minimize_limited_flag(vtree,manager,0);
}

//limited
Vtree* sdd_vtree_minimize_limited(Vtree* vtree, SddManager* manager) {
  return sdd_vtree_minimize_limited_flag(vtree,manager,1);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
