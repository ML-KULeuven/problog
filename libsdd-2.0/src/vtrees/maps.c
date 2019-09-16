/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//vtree nodes use zero-based indexing
//vtree variables use one-based indexing

/****************************************************************************************
 * computes a map from vnode positions in inorder to vnodes
 ****************************************************************************************/

//helper function for id2vtree
static
Vtree** fill_vtree_array(Vtree* vtree, Vtree** array) {
  if(LEAF(vtree)) {
    *array=vtree;
	return array;
  }
  else {
    Vtree** root_loc = 1+fill_vtree_array(vtree->left,array);
	*root_loc = vtree;
	return fill_vtree_array(vtree->right,1+root_loc);
  }
}

//returns an array that maps ids to their vnodes: array[id]=vnode 
Vtree** pos2vnode_map(Vtree* vtree) {
  SddSize count = 2*(vtree->var_count)-1; 
  Vtree** array;
  CALLOC(array,Vtree*,count,"pos2vnode_map");
  fill_vtree_array(vtree,array);
  return array;
}


/****************************************************************************************
 * end
 ****************************************************************************************/
