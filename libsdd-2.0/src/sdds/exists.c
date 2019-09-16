/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"


/****************************************************************************************
 * existentially quantifying an sdd 
 ****************************************************************************************/

//existentially quantify a variable out of an sdd
SddNode* sdd_exists(SddLiteral var, SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_exists");
  
  //condition will not do auto gc/minimize
  SddNode* p_cond = sdd_condition(var,node,manager);
  SddNode* n_cond = sdd_condition(-var,node,manager);
  
  //apply will not gc its arguments
  return sdd_apply(p_cond,n_cond,DISJOIN,manager);  
}


/****************************************************************************************
 * end
 ****************************************************************************************/
