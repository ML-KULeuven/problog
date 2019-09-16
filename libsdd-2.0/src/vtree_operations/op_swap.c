/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"
#include "vtree_ops.h"

//local declarations
static int try_swapping_partition(SddNodeSize* size, SddElement** elements, SddNode* node, Vtree* vtree, SddManager* manager, int limited);

/****************************************************************************************
 * swapping a vtree node and all its associated sdd nodes
 ****************************************************************************************/
 
//v = (a b) ===swap v===> v = (b a)
//
//return 1 if swapping is done within limits, otherwise return 0
//0 means no limit
int sdd_vtree_swap(Vtree* v, SddManager* manager, int limited) {
  
  if(limited) start_op_limits(manager);
   
  //stats
  manager->vtree_ops.current_op = 's';
  manager->vtree_ops.current_vtree = v->position;
  ++manager->vtree_ops.sw_count;
  
  assert(!FULL_DEBUG || verify_gc(v,manager));

  //remove nodes of v from unique table and collect them in a linked list;
  SddSize init_size  = sdd_manager_live_size(manager);
  SddSize count      = v->node_count; //before splitting
  SddNode* n_list    = split_nodes_for_swap(v,manager);
  //swapped nodes are removed and reinserted into hash table since their
  //hash keys may changed due to swapping (hash keys depend on primes/subs)
    
  //swap vtree structure
  swap_vtree_children(v,manager);
  if(count==0) return 1; //optimization: no nodes to swap
  
  //swap sdd nodes
  SddSize offset_size = init_size-sdd_manager_live_size(manager); //size of nodes currently outside the unique table (all live)
  SddNodeSize new_size; //size of swapped elements
  SddElement* new_elements; //swapped elements
  int success = 1;

  FOR_each_linked_node(n,n_list,{
    WITH_no_auto_mode(manager,
      success=try_swapping_partition(&new_size,&new_elements,n,v,manager,limited));
    if(success) { //node gets new elements and size
      offset_size -= n->size; //in two steps to avoid underflow
      offset_size += new_size;
      replace_node(1,n,new_size,new_elements,v,manager); //reversible
    }
    if(!success || (limited && exceeded_size_limit(offset_size,manager))) { //rollback
      swap_vtree_children(v,manager); //reverse vtree edit
      rollback_vtree_op(n_list,NULL,v,manager); //recover elements and move n_list back to v
      success = 0; //swap may have succeeded, but limit exceeded
      goto done;
    } 

  });
  
  assert(success);
  //confirm successful swapping of n_list and move back into v
  finalize_vtree_op(n_list,NULL,v,manager); 
  
  done:    
  
  //when sdd_vtree_swap() is called, it assumes no dead nodes above vtree v, and 
  //will not create dead nodes above v; hence, we only need to gc in v 
  garbage_collect_in(v,manager);
  assert(!FULL_DEBUG || verify_gc(v,manager));
  assert(!FULL_DEBUG || verify_counts_and_sizes(manager));
  manager->vtree_ops.current_op = ' ';
  
  if(limited) end_op_limits(manager);
  return success;
}

/****************************************************************************************
 * swaps the partition of an sdd node
 *
 * if successful, the swapped partition is returned (size, elements)
 ****************************************************************************************/

//v = (a b) ===swap===> v = (b a)
//node normalized for v 
//this function is called after vtree v has been swapped
//returns 1 if swapping succeeds (done within limits), 0 otherwise
int try_swapping_partition(SddNodeSize* size, SddElement** elements, SddNode* node, Vtree* v, SddManager* manager, int limited) {
 
  open_cartesian_product(manager);
  
    FOR_each_prime_sub_of_node(a,b,node,{
      open_partition(manager);
      
	    SddNode* neg_b = sdd_negate(b,manager);
        if(!IS_FALSE(b))     declare_element_of_partition(b,a,v,manager);
        if(!IS_FALSE(neg_b)) declare_element_of_partition(neg_b,manager->false_sdd,v,manager);
	  
	  if(!close_partition(0,v,manager,limited)) return 0; //with no compression
    });
    
  return close_cartesian_product(0,size,elements,v,manager,limited); //with no compression
}

/****************************************************************************************
 * END
 ****************************************************************************************/
