/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//local declarations
static SddSize sdd_size_leave_bits_1(SddNode* node);

/****************************************************************************************
 * size of sdd 
 ****************************************************************************************/

//compute size of the sdd (sum of its decomposition sizes)
SddSize sdd_size(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_size");
  assert(!GC_NODE(node));
  
  SddSize size = sdd_size_leave_bits_1(node); 
  sdd_clear_node_bits(node);
  return size;
}

SddSize sdd_size_leave_bits_1(SddNode* node) {
  if (node->bit) return 0;
  node->bit = 1;

  SddNodeSize size = 0;
  if(IS_DECOMPOSITION(node)) {
    size = node->size;
	FOR_each_prime_sub_of_node(prime,sub,node,{
	  size += sdd_size_leave_bits_1(prime);
	  size += sdd_size_leave_bits_1(sub);
	});
  }
  return size;
}


/****************************************************************************************
 * size of multi-rooted sdd 
 ****************************************************************************************/

//compute size of the sdd (sum of its decomposition sizes)
SddSize sdd_shared_size(SddNode** nodes, SddSize count) {

  SddSize size = 0;
  
  for(SddSize i=0; i<count; i++) {
    SddNode* node = nodes[i];
    CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_shared_size");
    assert(!GC_NODE(node));
    size += sdd_size_leave_bits_1(node);
  }
  
  for(SddSize i=0; i<count; i++) sdd_clear_node_bits(nodes[i]);
 
  return size; 
}

/****************************************************************************************
 * end
 ****************************************************************************************/
