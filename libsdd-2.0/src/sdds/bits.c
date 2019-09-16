/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

/****************************************************************************************
 * each sdd node has two associated bits: bit and cit
 *
 * initially all bits/cits are cleared 
 * every algorithm that uses bits and cits must maintain this invariant after being done 
 ****************************************************************************************/

//clear bits and cits of sdd nodes 
void sdd_clear_node_bits(SddNode* node) {
  if(node->bit==0) return; //cit must be 0
  node->bit=0;
  node->cit=0; //cit may have been 0 or 1
  if(IS_DECOMPOSITION(node)) {
    FOR_each_prime_sub_of_node(prime,sub,node,{
      sdd_clear_node_bits(prime);
	  sdd_clear_node_bits(sub);
    });
  }
}


/****************************************************************************************
 * constructs an array of all nodes in the sdd, with children appearing before parents
 *
 * sets the index of each node to its location in the array
 ****************************************************************************************/

SddNode** sdd_topological_sort(SddNode* node, SddSize* size) {
 
  void sdd_topological_sort_aux(SddNode* node, SddNode** start, SddNode*** array);
   
  //count number of nodes in sdd
  *size = sdd_all_node_count_leave_bits_1(node);
  //all nodes are marked 1 now
  
  //allocate array to hold sdd nodes
  SddNode** array;
  CALLOC(array,SddNode*,*size,"sdd_topological_sort");
  
  //fill array
  sdd_topological_sort_aux(node,array,&array);
  //all nodes are marked 0 now
  //array is now pointing to last cell in allocated array
  
  return array-*size+1;
}


void sdd_topological_sort_aux(SddNode* node, SddNode** start, SddNode*** array) {	

  if(node->bit==0) { //node has been visited before
    --(*array);
	return;
  }
  //this is the first visit to this node
  node->bit = 0;

  if(IS_DECOMPOSITION(node)) {
	FOR_each_prime_sub_of_node(prime,sub,node,{
	  sdd_topological_sort_aux(prime,start,array);
	  (*array)++;
	  sdd_topological_sort_aux(sub,start,array);
	  (*array)++;
	});
  }
	
  **array = node; //save node
  node->index = *array-start; //save location of node
}


/****************************************************************************************
 * count of sdd nodes having more than one parent
 *
 * LEAVES bits of counted nodes set to 1
 ****************************************************************************************/

//count the number of nodes in an sdd that have more than one parent but leaves bits set to 1
//
//we have the following cases:
//bit cit
//0   0   node has not been visited (initial state)
//1   0   node visited only once before
//1   1   node visited at least twice before (has more than one parent)
//0   1   not used
SddSize sdd_count_multiple_parent_nodes(SddNode* node) {
  if (node->bit==1) { //node visited before, it has multiple parents
    if(node->cit==0) { //this is the second visit to this node
      node->cit=1;
	  return 1; //count this node
	}
    else { //this has been visited at least twice before
	  return 0; //do not count this node
    }
  }
  //this is the very first visit for this node
  node->bit=1;

  SddSize count=0;
  if(IS_DECOMPOSITION(node)) {
    FOR_each_prime_sub_of_node(prime,sub,node,{
      count += sdd_count_multiple_parent_nodes(prime);
	  count += sdd_count_multiple_parent_nodes(sub);
	});
  }
  return count;
}

/****************************************************************************************
 * count of sdd nodes having more than one parent, visiting only sdd nodes that are
 * normalized for vnodes on the path to leaf
 *
 * LEAVES bits of counted nodes set to 1
 ****************************************************************************************/

SddSize sdd_count_multiple_parent_nodes_to_leaf(SddNode* node, Vtree* leaf) {
  if(node->bit==1) { //node visited before, it has multiple parents
    if(node->cit==0) { //this is the second visit to this node
	  node->cit=1;
	  return 1; //count this node
	}
	else { //this has been visited at least twice before
	  return 0; //do not count this node
	}
  }
  //this is the very first visit for this node
  node->bit=1;

  SddSize count=0;
  if(IS_DECOMPOSITION(node)) {
    if(sdd_vtree_is_sub(leaf,node->vtree->left)) { //visit primes
	  FOR_each_prime_of_node(prime,node,{
	    count += sdd_count_multiple_parent_nodes_to_leaf(prime,leaf);
	  });
	}
	else if(sdd_vtree_is_sub(leaf,node->vtree->right)) { //visit subs
	  FOR_each_sub_of_node(sub,node,{
	    count += sdd_count_multiple_parent_nodes_to_leaf(sub,leaf);
	  });
	}
  }
  
  return count;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
