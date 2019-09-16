/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//declarations

//vtrees/vtree.c
Vtree* new_leaf_vtree(SddLiteral var);
Vtree* new_internal_vtree(Vtree* left_child, Vtree* right_child);
void set_vtree_properties(Vtree* vtree);

/****************************************************************************************
 * utilities
 ****************************************************************************************/
 
//returns 
//'R' if vtree is root of manager vtree
//'l' if vtree is a left child of its parent
//'r' if vtree is a right child of its parent
static inline
char vtree_loc(Vtree* vtree) {
  Vtree* parent = vtree->parent;
  if(parent==NULL) return 'R'; //vtree is root
  else if(parent->left==vtree) return 'l'; //vtree is left child
  else return 'r'; //vtree is right child
}

//location=='R' if old_vtree is root of manager vtree
//location=='l' if old_vtree is a left child of its parent
//location=='r' if old_vtree is a right child of its parent
//puts new_vtree in the location of old_vtree
static inline
void replace_vtree_with(Vtree* old_vtree, Vtree* new_vtree, char location, SddManager* manager) {
  if(location=='R') {
    new_vtree->parent = NULL;
    manager->vtree = new_vtree;
  }
  else {
    Vtree* parent = old_vtree->parent;
    assert(parent!=NULL);
    
    //replace
    if(location=='l') parent->left  = new_vtree;
    else              parent->right = new_vtree;
    old_vtree->parent = NULL;
    new_vtree->parent = parent;
  }
}

//if on_left='l', make l_sibling a left child of parent, and r_sibling a right child
//if on_left='r', make r_sibling a left child of parent, and l_sibling a right child
static inline 
void connect_children(Vtree* parent, Vtree* l_sibling, Vtree* r_sibling, char on_left) {
  
  //update parents
  l_sibling->parent = parent;
  r_sibling->parent = parent;
  
  //update children
  if(on_left=='l') {
    parent->left  = l_sibling;
    parent->right = r_sibling;
  }
  else { //on_left = 'r'
    parent->left  = r_sibling;
    parent->right = l_sibling;
  }
}

/****************************************************************************************
 * add variable to vtree of manager
 ****************************************************************************************/
 
//adds a new leaf vtree node corresponding to var
//if location = 'l', added leaf vtree node is to the left of sibling
//if location = 'r', added leaf vtree node is to the right of sibling
//returns added leaf node

//assumes manager has at least one variable
Vtree* add_var_to_vtree(SddLiteral var, char location, Vtree* sibling, SddManager* manager) {
  
  assert(manager->var_count>0);
  assert(location=='l' || location=='r');
  
  //save parent as it will be erased by new_internal_vtree
  Vtree* parent = sibling->parent; //could be NULL
    
  //construct new leaf node
  Vtree* leaf = new_leaf_vtree(var);
  
  //construct new internal node (parent of sibling and new leaf vtree node)
  Vtree* internal;
  if(location=='l') internal = new_internal_vtree(leaf,sibling);
  else              internal = new_internal_vtree(sibling,leaf); //location = 'r'
    
  //replace sibling with internal as a child of parent
  internal->parent = parent; //could be NULL
  if(parent!=NULL) {
    if(parent->left==sibling) parent->left = internal;
    else parent->right = internal;
  }
  else manager->vtree = internal; //vtree has only a single node before addition
  
  //update properties to reflect new inorder and var counts
  //CAN BE done more efficiently
  set_vtree_properties(manager->vtree);
  
  return leaf;
}

/****************************************************************************************
 * remove variable from vtree of manager
 ****************************************************************************************/

void remove_var_from_vtree(SddLiteral var, SddManager* manager) {

  assert(manager->var_count>1);
  
  Vtree* leaf    = sdd_manager_vtree_of_var(var,manager);
  Vtree* parent  = leaf->parent;
  assert(parent!=NULL);
  Vtree* sibling = parent->left==leaf? parent->right: parent->left;
  
  char location = vtree_loc(parent);
  //location=='R' if parent is root of manager vtree
  //location=='l' if it is a left child
  //location=='r' if it is a right child
  
  replace_vtree_with(parent,sibling,location,manager);
  //sibling has now assumed the location of parent in the vtree
  //parent (and leaf) are now outside the vtree

  free(leaf);
  free(parent);
  
  //update properties to reflect new inorder and var counts
  //CAN BE done more efficiently
  set_vtree_properties(manager->vtree);
}

/****************************************************************************************
 * move variable in vtree of manager
 ****************************************************************************************/
 
//move the leaf vtree corresponding to var so it becomes a sibling of new_sibling
//if var_location = 'l', move leaf to the left of new_sibling
//if var_location = 'r', move leaf to the right of new_sibling

void move_var_in_vtree(SddLiteral var, char var_location, Vtree* new_sibling, SddManager* manager) {
  
  CHECK_ERROR(manager->var_count<=1,ERR_MSG_TWO_VARS,"move_var_in_vtree");
  CHECK_ERROR(sdd_manager_is_var_used(var,manager),ERR_MSG_MOV_VAR,"move_var_in_vtree");
  
  assert(LEAF(new_sibling));
  
  Vtree* leaf = sdd_manager_vtree_of_var(var,manager);
  if(leaf==new_sibling) return; //already in requested location
    
  Vtree* parent = leaf->parent;
  assert(parent!=NULL);
  assert(parent->node_count==0);
  Vtree* old_sibling = parent->left==leaf? parent->right: parent->left;
  
  //remove parent (and leaf) from its current location and replace with old_sibling
  
  char old_location = vtree_loc(parent);
  //location=='R' if parent is root of manager vtree
  //location=='l' if it is a left child
  //location=='r' if it is a right child
  
  replace_vtree_with(parent,old_sibling,old_location,manager);
  //old_sibling has now assumed the location of parent in the vtree (no longer pointing to parent)
  //parent (and leaf) are now outside the vtree
  
  //remove new_sibling from its current location and replace with parent
  
  char new_location = vtree_loc(new_sibling);
  replace_vtree_with(new_sibling,parent,new_location,manager);
  //parent has now assumed the location of new_sibling in the vtree
  //new_sibling is now outside the vtree
  
  //connect parent to its children according to var_location
  //this must be done after replacement above so new_sibling has its parent during replacement
  connect_children(parent,leaf,new_sibling,var_location);
  //leaf and new_sibling are now back into the vtree

  //update properties to reflect new inorder and var counts
  //CAN BE done more efficiently
  set_vtree_properties(manager->vtree);
}


/****************************************************************************************
 * end
 ****************************************************************************************/
