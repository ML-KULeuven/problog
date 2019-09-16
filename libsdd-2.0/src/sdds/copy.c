/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//local declarations
static void sdd_copy_aux(SddNode* node, SddNode** start, SddNode*** node_copies_loc, Vtree* org_vtree, Vtree* dest_vtree, SddManager* dest_manager);

/****************************************************************************************
 * copy an sdd from one manager to another
 *
 * will not do auto gc/minmize as its computations are done in no-auto mode
 ****************************************************************************************/

//node is currently in some manager, call it org_manager
//create a copy of node in the dest_manager, which may be different than org_manager
//assumes that org_manager->vtree and dest_manager->vtree have the same structure
//note: if org_manager=dest_manager, then node will be returned
SddNode* sdd_copy(SddNode* node, SddManager* dest_manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_copy");
  assert(!GC_NODE(node));
  
  //trivial nodes are special
  if(node->type==FALSE) return dest_manager->false_sdd;
  if(node->type==TRUE) return dest_manager->true_sdd;
  
  //node is not trivial
  
  //find how many nodes
  SddSize size = sdd_all_node_count_leave_bits_1(node);
  //all nodes are marked 1 now

  //create array to hold copies of sdd nodes
  SddNode** node_copies;
  CALLOC(node_copies,SddNode*,size,"sdd_copy");
  
  //find root of original vtree
  Vtree* org_root = node->vtree;
  while(org_root->parent!=NULL) org_root = org_root->parent;
  
  //copy nodes
  SddNode* node_copy;
  WITH_no_auto_mode(dest_manager,{
    sdd_copy_aux(node,node_copies,&node_copies,org_root,dest_manager->vtree,dest_manager);
    //all nodes are maked 0 now
    node_copies -= size;
    //node_copies is again pointing to first cell in allocated array
    node_copy = node_copies[node->index];
  });
  
  free(node_copies);
	
  return node_copy;
}


//org_vtree and dest_vtree are isomorphic
//node_vtree appears in org_vtree
//returns the node in dest_vtree which corresponds to node_vtree
Vtree* find_vtree_copy(Vtree* node_vtree, Vtree* org_vtree, Vtree* dest_vtree) {
  
  if(node_vtree==org_vtree) return dest_vtree;
  else if(sdd_vtree_is_sub(node_vtree,org_vtree->left)) 
    return find_vtree_copy(node_vtree,org_vtree->left,dest_vtree->left);
  else //sdd_vtree_is_sub(node_vtree,org_vtree->right)
    return find_vtree_copy(node_vtree,org_vtree->right,dest_vtree->right);
  
}


//compute node copies and store them in associated array
//org_vtree and dest_vtree are isomorphic
//node->vtree is a subvtree of org_vtree
void sdd_copy_aux(SddNode* node, SddNode** start, SddNode*** node_copies_loc, Vtree* org_vtree, Vtree* dest_vtree, SddManager* dest_manager) {	

  if(node->bit==0) return; //node has been visited before (i.e., already copied)
  else node->bit=0; //this is the first visit to this node

  SddNode* node_copy;
  
  if(node->type==FALSE) node_copy = dest_manager->false_sdd;
  else if(node->type==TRUE) node_copy = dest_manager->true_sdd;
  else if(node->type==LITERAL) node_copy = sdd_manager_literal(LITERAL_OF(node),dest_manager);
  else { //decomposition
    Vtree* vtree_copy = find_vtree_copy(node->vtree,org_vtree,dest_vtree);
    FOR_each_prime_sub_of_node(prime,sub,node,{
      //recursive calls for copying descendants
      sdd_copy_aux(prime,start,node_copies_loc,node->vtree,vtree_copy,dest_manager);
	  sdd_copy_aux(sub,start,node_copies_loc,node->vtree,vtree_copy,dest_manager);
	});
	//copy node
    GET_node_from_partition(node_copy,vtree_copy,dest_manager,{
	  FOR_each_prime_sub_of_node(prime,sub,node,{
	    SddNode* prime_copy = start[prime->index];
	    SddNode* sub_copy = start[sub->index];
	    DECLARE_element(prime_copy,sub_copy,vtree_copy,dest_manager);
	  });
	});
  }
	
  //saving node copy
  **node_copies_loc = node_copy; 
  
  //location of saved node copy
  node->index = *node_copies_loc-start;
   
  //advance to next cell in array
  (*node_copies_loc)++; 
  
}

/****************************************************************************************
 * end
 ****************************************************************************************/
