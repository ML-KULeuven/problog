/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

/****************************************************************************************
 * a limit of 0 means no limit
 *
 * types of limits:
 *
 * --time: 
 *   limits the time spent by a function call
 *   applicable to vtree search, fragment search, vtree operations, and apply
 *
 * --size: 
 *   limits the growth in size of an sdd normalized for a given vtree
 *   applicable to vtree operations
 *   requires one to specify the vtree whose sdd size is to be limited
 *
 * --memory: 
 *   limits the growth of memory used by the manager's sdd (memory of nodes and elements)
 *   applicable to vtree operations
 *
 * vtree search calls fragment search, which calls vtree operations, which call apply;
 * any of these four stages can be subjected to limits
 *
 * fragment search and vtree operations can also be called directly by the user 
 * (perhaps when implementing their own vtree search)
 * user can also choose to apply limits in this case
 *
 ****************************************************************************************/

//
//limiting vtree search
//
void start_search_limits(SddManager* manager) { //called upon starting vtree search
  assert(manager->vtree_ops.search_aborted==0);
  manager->vtree_ops.search_time_stamp = clock();
}
void end_search_limits(SddManager* manager) { //called upon finishing vtree search
  manager->vtree_ops.search_time_stamp = 0; //apply limit only when stamp is not 0
  manager->vtree_ops.search_aborted    = 0;
}
int search_aborted(SddManager* manager) { //called to check whether vtree search has been aborted
  return manager->vtree_ops.search_aborted;
}

//
//limiting fragment search
//
void start_fragment_limits(SddManager* manager) { //called upon starting fragment search
  assert(manager->vtree_ops.fragment_aborted==0);
  manager->vtree_ops.fragment_time_stamp = clock();
}
void end_fragment_limits(SddManager* manager) { //called upon finishing fragement search
  manager->vtree_ops.fragment_time_stamp = 0; //apply limit only when stamp is not 0
  manager->vtree_ops.fragment_aborted    = 0;
}
int fragment_aborted(SddManager* manager) { //called to check whether a fragment search has been aborted
  return manager->vtree_ops.fragment_aborted;
}

//
//limiting vtree ops (left-rotate, right-rotate or swap)
//
void start_op_limits(SddManager* manager) { //called upon starting a vtree op
  assert(manager->vtree_ops.op_aborted==0);
  manager->vtree_ops.op_time_stamp   = clock();
  manager->vtree_ops.op_memory_stamp = TYPE2MB(manager->node_count,SddNode)+TYPE2MB(manager->sdd_size,SddElement);
}
void end_op_limits(SddManager* manager) { //called up finishing a vtree op
  manager->vtree_ops.op_time_stamp   = 0; //apply limit only when stamp is not 0
  manager->vtree_ops.op_memory_stamp = 0; //apply limit only when stamp is not 0
  manager->vtree_ops.op_aborted      = 0;
}
int op_aborted(SddManager* manager) { //not used
  return manager->vtree_ops.op_aborted;
}

//
//limiting apply
//
void start_apply_limits(SddManager* manager) { //called upon starting a root, limited apply
  assert(manager->vtree_ops.apply_aborted==0);
  manager->vtree_ops.apply_time_stamp = clock();
}
void end_apply_limits(SddManager* manager) { //called upon finishing a root, limited apply
  manager->vtree_ops.apply_time_stamp = 0; //apply limit only when stamp is not 0
  manager->vtree_ops.apply_aborted    = 0;
}
int apply_aborted(SddManager* manager) { //called only in assertions
  return manager->vtree_ops.apply_aborted;
}

/****************************************************************************************
 * time and memory limits: checked only within a limited apply, 
 *                         which is called only by a limited vtree op
 *
 * memory limits are only applicable to vtree ops
 ****************************************************************************************/

static inline
int exceeded_op_memory_limit(SddManager* manager) {
   
  float memory_limit = manager->vtree_ops.op_memory_limit;
  if(memory_limit==0) return 0;
  
  float init_memory   = VTREE_OP_MEMORY_MIN+manager->vtree_ops.op_memory_stamp;
  float cur_memory    = TYPE2MB(manager->node_count,SddNode)+TYPE2MB(manager->sdd_size,SddElement);
  int memory_exceeded = cur_memory > init_memory*memory_limit;
  
  if(memory_exceeded) {
    switch(manager->vtree_ops.current_op) {
      case 'l': ++manager->vtree_ops.failed_lr_count_memory; break;
      case 'r': ++manager->vtree_ops.failed_rr_count_memory; break;
      case 's': ++manager->vtree_ops.failed_sw_count_memory; break;
    }
  }
  
  return memory_exceeded;
}

//--first checks memory limit for vtree op, then checks time limits in this order (when applicable):
//  search, fragment, op, apply
//--records first reason for exceeding limits (if any)

int exceeded_limits(SddManager* manager) {

  //limits are checked only in a limited apply, which is called only by a (limited) vtree op
  assert(manager->vtree_ops.current_op != ' '); //we must be in the context of a vtree op
  
  //clock() is expensive, so check limits only after that many applies
  if(manager->stats.apply_count%LIMITS_CHECK_FREQUENCY) return 0;
  
  //check should not be made again if already succeeded (that is, some limit has been exceeded)
  assert(manager->vtree_ops.search_aborted==0);
  assert(manager->vtree_ops.fragment_aborted==0);
  assert(manager->vtree_ops.op_aborted==0);
  assert(manager->vtree_ops.apply_aborted==0);
        
  //check memory limit first
  if(exceeded_op_memory_limit(manager)) return manager->vtree_ops.op_aborted = 1;
  
  //check time limits next
  clock_t cur_time = clock(); //equality may hold in assertions below
  assert(cur_time >= manager->vtree_ops.search_time_stamp); 
  assert(cur_time >= manager->vtree_ops.fragment_time_stamp); 
  assert(cur_time >= manager->vtree_ops.op_time_stamp);
  assert(cur_time >= manager->vtree_ops.apply_time_stamp); 
 
  if(manager->vtree_ops.search_time_limit && manager->vtree_ops.search_time_stamp &&
     cur_time > manager->vtree_ops.search_time_limit+manager->vtree_ops.search_time_stamp) {
    ++manager->auto_search_invocation_count_aborted_search;
    manager->vtree_ops.search_aborted = 1;
  }
  else if(manager->vtree_ops.fragment_time_limit && manager->vtree_ops.fragment_time_stamp &&
          cur_time > manager->vtree_ops.fragment_time_limit+manager->vtree_ops.fragment_time_stamp) {
    ++manager->auto_search_invocation_count_aborted_fragment;
    manager->vtree_ops.fragment_aborted = 1;
  }
  else if(manager->vtree_ops.op_time_limit && manager->vtree_ops.op_time_stamp &&
          cur_time > manager->vtree_ops.op_time_limit+manager->vtree_ops.op_time_stamp) {
    ++manager->auto_search_invocation_count_aborted_operation;
    manager->vtree_ops.op_aborted = 1;
  }
  else if(manager->vtree_ops.apply_time_limit && manager->vtree_ops.apply_time_stamp &&
          cur_time > manager->vtree_ops.apply_time_limit+manager->vtree_ops.apply_time_stamp) {
    ++manager->auto_search_invocation_count_aborted_apply;
    manager->vtree_ops.apply_aborted = 1;
  } 
  else return 0;
    
  switch(manager->vtree_ops.current_op) {
    case 'l': ++manager->vtree_ops.failed_lr_count_time; break;
    case 'r': ++manager->vtree_ops.failed_rr_count_time; break;
    case 's': ++manager->vtree_ops.failed_sw_count_time; break;
  }

  return 1; //time out
}

/****************************************************************************************
 * size limit: checked only within a vtree op (left-rotate, right-rotate & swap)
 *
 * to use this limit, one must first call sdd_manager_init_vtree_size_limit() to set
 * the reference size for imposing the size limit. this reference size can be updated
 * efficiently using a call to sdd_manager_update_vtree_size_limit()
 ****************************************************************************************/

//declares the baseline size for enforcing a size limit: the current size of vtree
void sdd_manager_init_vtree_size_limit(Vtree* vtree, SddManager* manager) {
 manager->vtree_ops.op_size_stamp = sdd_vtree_live_size(vtree);
 //outside size is constant and needed for efficiently computing current size
 manager->vtree_ops.outside_size = sdd_manager_live_size(manager)-manager->vtree_ops.op_size_stamp; 
}

//updates the baseline size for enforcing size limits
//avoid recomputing the size of the vtree whose size is being limited
void sdd_manager_update_vtree_size_limit(SddManager* manager) {
  manager->vtree_ops.op_size_stamp = sdd_manager_live_size(manager)-manager->vtree_ops.outside_size;
}

//--this function is called after each node has been processed by a vtree operation
//
//--nodes to be processed by vtree operations are outside the unique table and, hence,
//  their sizes are not accounted for in sizes maintained by the manager and vtree nodes
//--offset_size represent the current size of such nodes: it must be added to the
//  current manager size to get the correct size

int exceeded_size_limit(SddSize offset_size, SddManager* manager) {
 
  float size_limit = manager->vtree_ops.op_size_limit;
  if(size_limit==0) return 0;
   
  SddSize cur_size  = (offset_size+sdd_manager_live_size(manager))-manager->vtree_ops.outside_size;
  if(cur_size <= VTREE_OP_SIZE_MIN) return 0; //don't enforce limit for small enough sizes
  
  SddSize init_size = manager->vtree_ops.op_size_stamp;
  int size_exceeded = cur_size > size_limit*init_size;
  
  if(size_exceeded) {
    switch(manager->vtree_ops.current_op) {
      case 'l': ++manager->vtree_ops.failed_lr_count_size; break;
      case 'r': ++manager->vtree_ops.failed_rr_count_size; break;
      case 's': ++manager->vtree_ops.failed_sw_count_size; break;
    }
  }
  
  return size_exceeded;
}

/****************************************************************************************
 * END
 ****************************************************************************************/
