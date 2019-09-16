/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//local declarations
static SddNode* sdd_condition_aux(SddNode* node, SddNode* literal, SddNode** start, SddNode*** cond_nodes, SddManager* manager);

/****************************************************************************************
 * conditioning an sdd 
 *
 * will not do auto gc/minmize as its computations are done in no-auto mode
 ****************************************************************************************/
 
//condition sdd node on literal
SddNode* sdd_condition(SddLiteral lit, SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_condition");
    
  if(node->type==FALSE || node->type==TRUE) return node;
    
  SddNode* literal = sdd_manager_literal(lit,manager); //literal being conditioned on 
  
  //find how many sdd nodes have multiple parents
  //count only nodes that are normalized for vnodes on the path from node->vtree to leaf of literal
  SddSize count = sdd_count_multiple_parent_nodes_to_leaf(node,literal->vtree);
  //all relevant nodes are marked 1 now

  //create array to hold conditioned sdd nodes
  SddNode** cond_nodes;
  CALLOC(cond_nodes,SddNode*,count,"sdd_condition");

  SddNode* cond_node;
  WITH_no_auto_mode(manager,{
    cond_node = sdd_condition_aux(node,literal,cond_nodes,&cond_nodes,manager);
    //all relevant nodes are maked 0 now
  });
  
  //cond_nodes is now pointing to last cell in allocated array
  free(cond_nodes-count+1);
	
  return cond_node;
}

//conditions node and some of its descendants on literal (if node's bit is 1)
//stores conditionings in the cond_nodes array, children before parents
//returns the conditioning of node
SddNode* sdd_condition_aux(SddNode* node, SddNode* literal, SddNode** start, SddNode*** cond_nodes, SddManager* manager) {
  //leaf_position is position of the leaf vnode that hosts the variable of literal
  if(node->bit==0) { //node has been visited before, lookup its cached conditioning
    --(*cond_nodes);
	return start[node->index];
  }
  //this is the first visit to this node
  node->bit=0;

  SddNode* cond_node;
  Vtree* vtree = node->vtree; //could be NULL
  Vtree* leaf_vtree = literal->vtree;
  
  if(node->type==TRUE) cond_node = node;
  else if(node->type==FALSE) cond_node = node;
  else if(node->type==LITERAL) {
    if(leaf_vtree==vtree) { //two literals over same variable
      cond_node = (node==literal? manager->true_sdd: manager->false_sdd);
    }
    else cond_node = node;
  }
  else { //decomposition node
    if(sdd_vtree_is_sub(leaf_vtree,vtree->left)) { //condition primes
	  GET_node_from_partition(cond_node,vtree,manager,{
	    FOR_each_prime_sub_of_node(prime,sub,node,{
	      //prime cannot be true or false
		  SddNode* cond_prime = sdd_condition_aux(prime,literal,start,cond_nodes,manager);
		  (*cond_nodes)++;
		  if(!IS_FALSE(cond_prime)) DECLARE_element(cond_prime,sub,vtree,manager);
	    });
	  });
    }
    else if(sdd_vtree_is_sub(leaf_vtree,vtree->right)) { //condition subs
	  GET_node_from_partition(cond_node,vtree,manager,{
	    FOR_each_prime_sub_of_node(prime,sub,node,{
		  SddNode* cond_sub = sdd_condition_aux(sub,literal,start,cond_nodes,manager);
   		  (*cond_nodes)++;
		  DECLARE_element(prime,cond_sub,vtree,manager);
	    });
	  });
    }
    else cond_node = node; //node and literal normalized for incomparable nodes
  }
	
  if(node->cit==1) { //node has multiple parents: save its conditioning as it will be visited again
    node->cit=0;
	**cond_nodes = cond_node; //saving conditioning
	node->index = *cond_nodes-start; //location of saved conditioning
  }
  else --(*cond_nodes); //node will not be visited again, do not save conditioning

  return cond_node;
}


/****************************************************************************************
 * end
 ****************************************************************************************/
