/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"


/****************************************************************************************
 * convert vtree into a left-linear one
 * return new root 
 ****************************************************************************************/

Vtree* left_linearize_vtree(Vtree* vtree, SddManager* manager) {
  Vtree** root_location = sdd_vtree_location(vtree,manager);
  while(INTERNAL(vtree)) {
    while(INTERNAL(vtree->right)) {
      sdd_vtree_rotate_left(vtree->right,manager,0);
      vtree = vtree->parent;
    }
    vtree = vtree->left;
  }
  return *root_location;
}

/****************************************************************************************
 * convert vtree into a right-linear one (obdd)
 * return new root
 *
 * proposed by Stout and Warren
 ****************************************************************************************/
 
Vtree* right_linearize_vtree(Vtree* vtree, SddManager* manager) {
  Vtree** root_location = sdd_vtree_location(vtree,manager);
  while(INTERNAL(vtree)) {
    while(INTERNAL(vtree->left)) {
      sdd_vtree_rotate_right(vtree,manager,0);
      vtree = vtree->parent;
    }
    vtree = vtree->right;
  }
  return *root_location;
}

/****************************************************************************************
 * convert vtree into a balanced one
 * return new root 
 *
 * this is known as the Stout and Warren algorithm, which improves Day's algorithm
 ****************************************************************************************/

Vtree* balance_vtree(Vtree* vtree, SddManager* manager) {

  Vtree** root_location = sdd_vtree_location(vtree,manager);
  vtree = right_linearize_vtree(vtree,manager);

  SddLiteral b = vtree->var_count -2;
  SddLiteral m = b/2;
  
  while(m > 0) {
    for(SddLiteral i=0; i<m; i++) { //perform m left rotations
      vtree = vtree->right;
      sdd_vtree_rotate_left(vtree,manager,0);
      vtree = vtree->right;
    }
    b = b - m - 1;
    m = b / 2;
    vtree = *root_location; //go to root
  }
  
  return *root_location;
}

 
/****************************************************************************************
 * END
 ****************************************************************************************/
