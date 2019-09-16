/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"


//local declarations
SddSize sdd_node_count_leave_bits_1(SddNode* node);

/****************************************************************************************
 * count of all nodes in an sdd 
 ****************************************************************************************/

//count the number of ALL nodes (decomposition and terminal) in an sdd
SddSize sdd_all_node_count(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_all_node_count");
  assert(!GC_NODE(node));
  
  SddSize count = sdd_all_node_count_leave_bits_1(node); 
  sdd_clear_node_bits(node);
  return count;
}

//count the number of all nodes in an sdd but leaves bits set to 1
//this is more efficient than sdd_node_count which has to clear bits
SddSize sdd_all_node_count_leave_bits_1(SddNode* node) {
  if (node->bit) return 0;
  node->bit = 1;

  SddSize count = 1;
  if(node->type==DECOMPOSITION) {
    FOR_each_prime_sub_of_node(prime,sub,node,{
	  count += sdd_all_node_count_leave_bits_1(prime);
	  count += sdd_all_node_count_leave_bits_1(sub);
	});
  }
  return count;
}


/****************************************************************************************
 * count of decomposition nodes in an sdd 
 ****************************************************************************************/
 
//count the number of DECOMPOSITION nodes in an sdd
SddSize sdd_count(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_count");
  assert(!GC_NODE(node));
  
  SddSize count = sdd_node_count_leave_bits_1(node); 
  sdd_clear_node_bits(node);
  return count;
}

//count the number of decomposition nodes in an sdd but leaves bits set to 1
//this is more efficient than sdd_node_count which has to clear bits
SddSize sdd_node_count_leave_bits_1(SddNode* node) {
  if (node->bit) return 0;
  node->bit = 1;

  SddSize count = 0;
  if(node->type==DECOMPOSITION) {
    ++count;
    FOR_each_prime_sub_of_node(prime,sub,node,{
	  count += sdd_node_count_leave_bits_1(prime);
	  count += sdd_node_count_leave_bits_1(sub);
	});
  }
  return count;
}


/****************************************************************************************
 * end
 ****************************************************************************************/
