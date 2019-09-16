/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"


/****************************************************************************************
 * universally quantify a variable out of an sdd 
 ****************************************************************************************/

//universally quantify a variable out of an sdd
SddNode* sdd_forall(SddLiteral var, SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_forall");
  
  //condition will not do auto gc/minimize
  SddNode* p_cond = sdd_condition(var,node,manager);
  SddNode* n_cond = sdd_condition(-var,node,manager);
  
  //apply will not gc its arguments
  return sdd_apply(p_cond,n_cond,CONJOIN,manager);    
}

/****************************************************************************************
 * end
 ****************************************************************************************/
