/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"
#include "vtree_ops.h"

//local declarations
static int try_rotating_partition_left(SddNodeSize* size, SddElement** elements, SddNode* node, Vtree* x, SddManager* manager, int limited);

/****************************************************************************************
 * left rotating a vtree node and its associated sdd nodes
 ****************************************************************************************/

//w = (a x=(b c)) ===left rotation of x===> x = (w=(a b) c)
//
//returns 1 if rotation is done within limits, otherwise returns 0
//0 means no limit
int sdd_vtree_rotate_left(Vtree* x, SddManager* manager, int limited) {
  
  if(limited) start_op_limits(manager);
  
  //stats
  manager->vtree_ops.current_op = 'l';
  manager->vtree_ops.current_vtree = x->position;
  ++manager->vtree_ops.lr_count; 
    
  Vtree* w = x->parent;
  assert(!FULL_DEBUG || verify_gc(w,manager));
  
  //unique nodes at vtree x continue to be normalized for x after left rotating x
  //unique nodes at w fall into three groups:
  //--w(a,bc): must be rotated and moved to x (this is the bc_list)
  //--w(a,c) : must be moved to x (this is the c_list)
  //--w(a,b) : stays at w (this is the ab_list)
  //rotation may create or lookup nodes at w, so the ab_list must be at w during rotation
  
//  SddSize init_count = manager->node_count;
  SddSize init_size  = sdd_manager_live_size(manager);
  SddSize bc_count; SddNode* bc_list; SddNode* c_list;
  split_nodes_for_left_rotate(&bc_count,&bc_list,&c_list,w,x,manager); //must be done before rotating vtree
   
  //rotate vtree structure
  rotate_vtree_left(x,manager); 
  Vtree* new_root = x;
  
  //rotate sdd nodes
//  SddSize offset_count = init_count-manager->node_count; //count of nodes currently outside the unique table (all live)
  SddSize offset_size  = init_size-sdd_manager_live_size(manager); //size of nodes currently outside the unique table (all live)
  SddNodeSize new_size; //size of rotated elements
  SddElement* new_elements; //rotated elements
  int success = 1;
  
  FOR_each_linked_node(n,bc_list,{
    WITH_no_auto_mode(manager,
      success=try_rotating_partition_left(&new_size,&new_elements,n,x,manager,limited));
    if(success) { //node gets new elements and size
      offset_size -= n->size; //in two steps to avoid underflow
      offset_size += new_size;
      replace_node(1,n,new_size,new_elements,x,manager); //reversible
    }
    if(!success || (limited && exceeded_size_limit(offset_size,manager))) { //rollback
      rotate_vtree_right(x,manager); //reverse vtree edit
      //recover elements and move bc_list back to w
      //move c_list back to w
      rollback_vtree_op(bc_list,c_list,w,manager);
      new_root = w; //back to original root
      success = 0; //rotate may have succeeded, but limit exceeded
      goto done;
    }
        
  });

  assert(success);
  //confirm successful rotation of bc_list and moved to x
  //move c_list to x
  finalize_vtree_op(bc_list,c_list,x,manager);
  
  done:  
  
  //when sdd_vtree_rotate_left() is called, it assumes no dead nodes above vtree w, 
  //and will not create dead nodes above new_root; hence, we only need to gc in new_root 
  garbage_collect_in(new_root,manager);
  assert(!FULL_DEBUG || verify_gc(new_root,manager));
  assert(!FULL_DEBUG || verify_counts_and_sizes(manager));
  manager->vtree_ops.current_op = ' ';
  
  if(limited) end_op_limits(manager);
  return success; 
}

/****************************************************************************************
 * left-rotates the partition of an sdd node
 *
 * if successful, the rotated partition is returned (size, elements)
 ****************************************************************************************/

//w = (a x=(b c)) ===left rotation of x===> x = (w=(a b) c)
//node depends on a, b and c
//node normalized for vtree w (before rotation)
//this function is called after left rotation of vtree x
//returns 1 if rotation succeeds (done within time limit), 0 otherwise
static
int try_rotating_partition_left(SddNodeSize* size, SddElement** elements, SddNode* node, Vtree* x, SddManager* manager, int limited) {
  Vtree* w = x->left;
    
  START_partition(manager);
  
    FOR_each_prime_sub_of_node(a,bc,node,{
      //bc can be true, false, a literal or a decomposition
      if(TRIVIAL(bc)) DECLARE_element(a,bc,x,manager);
      else if(bc->vtree==x) { //bc is a decomposition, normalized for vtree x
        FOR_each_prime_sub_of_node(b,c,bc,{
      	  //prime ab cannot be false since neither a nor b can be false
		  SddNode* ab = sdd_conjoin_lr(a,b,w,manager); //not limited
		  assert(ab!=NULL);
		  DECLARE_element(ab,c,x,manager);
	    });
	  } 
	  //bc is literal or decomposition, normalized for a vtree in b or a vtree in c
      else if(bc->vtree->position > x->position) { //bc is normalized for a vtree in c
        //view bc as a decomposition true.bc normalized for vtree x
        DECLARE_element(a,bc,x,manager);
      }
      else { //bc is normalized for a vtree in b
        //view bc as a decomposition (bc.true + ~bc.false) normalized for vtree x
        SddNode* ab = sdd_conjoin_lr(a,bc,w,manager); //not limited
        DECLARE_element(ab,manager->true_sdd,x,manager);
        SddNode* bc_neg = sdd_negate(bc,manager);
        ab = sdd_conjoin_lr(a,bc_neg,w,manager); //not limited
        assert(ab!=NULL);
        DECLARE_element(ab,manager->false_sdd,x,manager);
      }
    });

  return GET_elements_of_partition(size,elements,x,manager,limited);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
