/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//sdds/apply.c
int root_apply(SddManager* manager);

//local 
static int try_auto_minimize_top(Vtree* vtree, SddManager* manager);
static int try_auto_minimize_recursive(Vtree* vtree, SddManager* manager);
static void save_size(Vtree* vtree);
    
/****************************************************************************************
 * invoking vtree search and/or gc in auto mode
 *
 * only vtree will be changed in case this came from a recursive apply call; otherwise,
 * the whole vtree may change
 *
 * deciding an auto trigger is all about identifying the right:
 * --FEATURES to consider in designing an auto test; and
 * --the corresponding THRESHOLDS for these features
 ****************************************************************************************/

/****************************************************************************************
 * parameters
 ****************************************************************************************/

#define GLOBAL_GROWTH 2
#define LOCAL_GROWTH  1.15

//apply growth >= recursive growth, otherwise it is ineffective
#define APPLY_GROWTH     2
#define RECURSIVE_GROWTH 2

//for triggering gc after any apply
#define DEAD_APPLY_GROWTH .5

/****************************************************************************************
 * search 
 ****************************************************************************************/

void try_auto_gc_and_minimize(Vtree* vtree, SddManager* manager) {
  assert(manager->auto_gc_and_search_on);
  
  int top_level_apply = root_apply(manager);
  int searched = top_level_apply? try_auto_minimize_top(vtree,manager): 
                                  try_auto_minimize_recursive(vtree,manager);  
  
  //search invokes gc
  if(top_level_apply && !searched) { //try gc
    SddSize dead  = sdd_manager_dead_count(manager)-manager->auto_apply_outside_dead_count; //apply vtree
    SddSize live  = sdd_manager_live_count(manager)-manager->auto_apply_outside_live_count; //apply vtree
    SddSize all   = dead+live;
    if(dead > all*DEAD_APPLY_GROWTH) {
      ++manager->auto_gc_invocation_count;
      sdd_vtree_garbage_collect(vtree,manager);  //local only
    }
  }
}

static
Vtree* search(Vtree* vtree, SddManager* manager) {
  clock_t search_time = 0;
  WITH_timing(search_time,{
    if(manager->vtree_search_function!=NULL) vtree = (*((SddVtreeSearchFunc*)manager->vtree_search_function))(vtree,manager);
    else vtree = sdd_vtree_minimize_limited(vtree,manager);
  });
  manager->stats.auto_search_time    += search_time;
  manager->stats.auto_max_search_time = MAX(search_time,manager->stats.auto_max_search_time);
  return vtree;
}

/****************************************************************************************
 * searching after a top-level apply call has finished
 *
 * may change the whole vtree
 ****************************************************************************************/

//called from a top-level apply
//vtree is the lca of the apply arguments
static
int try_auto_minimize_top(Vtree* vtree, SddManager* manager) {
  assert(root_apply(manager)); //top-level apply
     
  //sizes of manager vtree
  SddSize cur_size  = sdd_manager_live_size(manager); //now
  SddSize last_size = manager->vtree->auto_last_search_live_size; //since last search
  
  if(cur_size < last_size) return 0;
  
  //sizes of apply vtree
  SddSize out_apply_size  = manager->auto_apply_outside_live_size; //size outside apply vtree
  SddSize cur_apply_size  = sdd_manager_live_size(manager)-out_apply_size; //size of apply vtree, now
  SddSize last_apply_size = vtree->auto_last_search_live_size; //size of apply vtree, after last search
  
  int global = !out_apply_size && cur_size >= GLOBAL_GROWTH*last_size; //manager vtree grew enough
  int local  = out_apply_size  && cur_apply_size >= LOCAL_GROWTH*last_apply_size; //apply vtree grew enough
   
  Vtree* root = (out_apply_size && manager->auto_local_gc_and_search_on==0)? manager->vtree: vtree;
  
  if(global || local) { 
    ++manager->auto_search_invocation_count;
    if(out_apply_size) ++manager->auto_search_invocation_count_global;
    else ++manager->auto_search_invocation_count_local;
    root = search(root,manager); //root may have changed
    save_size(root); //establish baseline for next search
    return 1;
  }
  else return 0;
}

/****************************************************************************************
 * searching after a recursive apply call has finished
 *
 * will only change the vtree of recursive apply call
 ****************************************************************************************/

//called from a recursive apply
static
int try_auto_minimize_recursive(Vtree* vtree, SddManager* manager) {
  assert(!root_apply(manager)); //recursive apply
  
  //apply vtree
  Vtree* apply_vtree      = manager->auto_apply_vtree;
  SddSize cur_apply_size  = sdd_manager_live_size(manager)-manager->auto_apply_outside_live_size; //now
  SddSize last_apply_size = apply_vtree->auto_last_search_live_size; //since last search inside top-level apply

  if(cur_apply_size < APPLY_GROWTH*last_apply_size) return 0; //apply vtree did not grow enough
  
  //this vtree   
  SddSize cur_vtree_size   = sdd_vtree_live_size(vtree);
//  SddSize last_vtree_size  = vtree->auto_last_search_live_size;
  int recursive = cur_vtree_size==0 || //balancing
                  cur_vtree_size >= RECURSIVE_GROWTH*last_apply_size; //vtree grew enough
  
  if(recursive) {
    //automatic vtree search
    ++manager->auto_search_invocation_count;
    ++manager->auto_search_invocation_count_recursive;
    vtree = search(vtree,manager); //only this vtree
    save_size(vtree); //establish baseline for next search
    return 1;
  }
  else return 0;
}
    
/****************************************************************************************
 * saving sizes after search
 ****************************************************************************************/

static
SddSize save_size_down(Vtree* vtree) {
  if(LEAF(vtree)) return 0;
  else {
    return vtree->auto_last_search_live_size = 
      sdd_vtree_live_size_at(vtree)+ 
      save_size_down(vtree->left)+
      save_size_down(vtree->right);
  }
}

static 
void save_size_up(Vtree* vtree) {
  while(vtree) {
    vtree->auto_last_search_live_size = 
      sdd_vtree_live_size_at(vtree)+ 
      vtree->left->auto_last_search_live_size+
      vtree->right->auto_last_search_live_size;
    vtree = vtree->parent;
  }
}

//saves sizes of descendants of vtree, and its ancestors
static
void save_size(Vtree* vtree) {
  save_size_down(vtree);
  save_size_up(vtree->parent);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
