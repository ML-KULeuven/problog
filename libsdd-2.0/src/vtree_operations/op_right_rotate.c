/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"
#include "vtree_ops.h"

//local declarations
static int try_rotating_partition_right(SddNodeSize* size, SddElement** elements, SddNode* node, Vtree* w, SddManager* manager, int limited);

/****************************************************************************************
 * right rotating a vtree node and its associated sdd nodes
 ****************************************************************************************/

//x = (w=(a b) c) ===right rotation of x===> w = (a x=(b c))
//
//returns 1 if rotation is done within limits, otherwise returns 0
//0 means no limit
int sdd_vtree_rotate_right(Vtree* x, SddManager* manager, int limited) {
 
  if(limited) start_op_limits(manager);
   
  //stats
  manager->vtree_ops.current_op = 'r';
  manager->vtree_ops.current_vtree = x->position;
  ++manager->vtree_ops.rr_count;
   
  Vtree* w = x->left;
  assert(!FULL_DEBUG || verify_gc(x,manager));

  //unique nodes at vtree w continue to be normalized for w after right rotating x
  //unique nodes at x fall into three groups:
  //--x(ab,c): must be rotated and moved to w (this is the ab_list)
  //--x(a,c) : must be moved to w (this is the a_list)
  //--x(b,c) : stays at x (this is the bc_list)
  //rotation may create or lookup nodes at x, so the bc_list must be at x during rotation
    
//  SddSize init_count = manager->node_count;
  SddSize init_size  = sdd_manager_live_size(manager);
  SddSize ab_count; SddNode* ab_list; SddNode* a_list;
  split_nodes_for_right_rotate(&ab_count,&ab_list,&a_list,x,w,manager); //must be done before rotating vtree
  
  //rotate vtree structure
  rotate_vtree_right(x,manager);
  Vtree* new_root = w;
  
  //rotate sdd nodes
//  SddSize offset_count = init_count-manager->node_count; //count of nodes currently outside the unique table (all live)
  SddSize offset_size  = init_size-sdd_manager_live_size(manager); //size of nodes currently outside the unique table (all live)
  SddNodeSize new_size; //size of rotated elements
  SddElement* new_elements; //rotated elements
  int success = 1;
  
  FOR_each_linked_node(n,ab_list,{
    WITH_no_auto_mode(manager,
      success=try_rotating_partition_right(&new_size,&new_elements,n,w,manager,limited));
    if(success) { //node gets new elements and size
      offset_size -= n->size; //in two steps to avoid underflow
      offset_size += new_size;
      replace_node(1,n,new_size,new_elements,w,manager); //reversible
    }
    if(!success || (limited && exceeded_size_limit(offset_size,manager))) { //rollback
      rotate_vtree_left(x,manager); //reverse vtree edit
      //recover elements and move ab_list back to x
      //move a_list back to x
      rollback_vtree_op(ab_list,a_list,x,manager);
      new_root = x; //back to original root
      success = 0; //rotate may have succeeded, but limit exceeded
      goto done;
    }  
  
  });
  
  assert(success);
  //confirm successful rotation of ab_list and moved to w
  //move a_list to w
  finalize_vtree_op(ab_list,a_list,w,manager);
  
  done:
  
  //when sdd_vtree_rotate_right() is called, it assumes no dead nodes above vtree x, 
  //and will not create dead nodes above new_root; hence, we only need to gc in new_root 
  garbage_collect_in(new_root,manager);
  assert(!FULL_DEBUG || verify_gc(new_root,manager));
  assert(!FULL_DEBUG || verify_counts_and_sizes(manager));
  manager->vtree_ops.current_op = ' ';
  
  if(limited) end_op_limits(manager);
  return success;
}

/****************************************************************************************
 * right-rotates the partition of an sdd node
 *
 * if successful, the rotated partition is returned (size, elements)
 ****************************************************************************************/

//x = (w=(a b) c) ===right rotation of x===> w = (a x=(b c))
//nodes depends on a b and c
//node normalized for vtree x (before rotation)
//this function is called after right rotation of vtree x
//returns 1 if rotation succeeds (done within limits), 0 otherwise
static
int try_rotating_partition_right(SddNodeSize* size, SddElement** elements, SddNode* node, Vtree* w, SddManager* manager, int limited) {  
  Vtree* x = w->right;
  
  open_cartesian_product(manager);
    
    FOR_each_prime_sub_of_node(ab,c,node,{
      //ab is either a literal or a decomposition
      //ab cannot be trivial
      open_partition(manager);
      
        if(ab->vtree==w) { //ab is a decomposition, normalized for vtree w
	      FOR_each_prime_sub_of_node(a,b,ab,{
	        SddNode* bc = sdd_conjoin_lr(b,c,x,manager); //not limited
	        assert(bc!=NULL);
	        declare_element_of_partition(a,bc,w,manager);
	      });
	    }
	    //ab is normalized for a vtree in a or a vtree in b
        else if(sdd_vtree_is_sub(ab->vtree,w->right)) { //ab is normalized for a vtree in b
          //view ab as a decomposition true.ab normalized for vtree w 
          SddNode* a  = manager->true_sdd;
          SddNode* bc = sdd_conjoin_lr(ab,c,x,manager); //not limited
          assert(bc!=NULL);
          declare_element_of_partition(a,bc,w,manager);
        }
        else { //ab is normalized for a vtree in a
          //view ab as a decomposition (ab.true + ~ab.false) normalized for vtree w 
          SddNode* a  = ab;
          SddNode* bc = c;
          declare_element_of_partition(a,bc,w,manager);
          a  = sdd_negate(ab,manager);
          bc = manager->false_sdd;
          declare_element_of_partition(a,bc,w,manager);
        }
        
	  if(!close_partition(1,w,manager,limited)) return 0; //with compression   
    });
    
  return close_cartesian_product(1,size,elements,w,manager,limited); //with compression
}


/****************************************************************************************
 * end
 ****************************************************************************************/
