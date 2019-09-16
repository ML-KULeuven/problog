/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//declarations

//basic/memory.c
void gc_sdd_node(SddNode* node, SddManager* manager);

//manager/manager.c
void setup_literal_sdds(Vtree* vtree, SddManager* manager);

//vtrees/vtree.c
Vtree* first_leaf_vtree(Vtree* vtree);
Vtree* last_leaf_vtree(Vtree* vtree);

//vtrees/compare.c
Vtree* sdd_manager_lca_of_literals(int count, SddLiteral* variables, SddManager* manager);

//vtrees/edit.c
Vtree* add_var_to_vtree(SddLiteral var, char location, Vtree* sibling, SddManager* manager);
void remove_var_from_vtree(SddLiteral var, SddManager* manager);
void move_var_in_vtree(SddLiteral var, char var_location, Vtree* new_sibling, SddManager* manager);

/****************************************************************************************
 * a variable is "used" iff 
 *  one of its literals appear as prime or sub of an sdd node (dead or alive)
 *
 * NOTE: this is a weak notion of usage since an unused variable may still have one of 
 * its literals assigned to some variable by the user
 *
 * MOVING unused variables is always safe
 *
 * REMOVING unused variables may not be safe: it is the user responsibility to ensure
 * that no direct references (i.e., variable assignmnets) exist to its literals
 ****************************************************************************************/

//a variable is "used" iff one of its literals appear as prime or sub in some sdd node (dead or alive)
//returns 1 if variable is used, 0 otherwise
int sdd_manager_is_var_used(SddLiteral var, SddManager* manager) {
  return (sdd_manager_literal(var,manager))->parent_count > 0 || (sdd_manager_literal(-var,manager))->parent_count > 0;
}

//returns an array map with the following properties:
//size    : 1+number of variables in manager
//map[var]: 1 if var is used, 0 otherwise
//map[0]  : not used 
int* var_usage_map(SddManager* manager) {
  int* map;
  CALLOC(map,int,1+manager->var_count,"var_usage_map");
  for(int var=1; var<=manager->var_count; var++) map[var] = sdd_manager_is_var_used(var,manager);
  return map;
}

/****************************************************************************************
 * add var to a manager
 *
 * index of added variable is 1+largest index of variables in manager
 *
 * a new leaf_node (for variable) will be added as a side effect to the manager's vtree
 * a new internal_node will be added as a side effect to the manager's vtree
 * this new internal node will inherit the parent of sibling (if any)
 *
 * if location='l', internal_node will have children (leaf_node,sibling)
 * if location='r', internal_node will have children (sibling,leaf_node)
 *
 ****************************************************************************************/
 
void add_var_to_manager(char location, Vtree* sibling, SddManager* manager) {

  SddLiteral last_var_count = manager->var_count;
  SddLiteral new_var_count = ++manager->var_count;
  assert(last_var_count >= 1);
  
  //add new leaf node to current vtree
  Vtree* new_leaf = add_var_to_vtree(new_var_count,location,sibling,manager); //added leaf node

  //expand literals array to hold entries for added variable
  SddLiteral last_array_size = 1+2*last_var_count;
  SddLiteral new_array_size  = 1+2*new_var_count;
  //literal indices are arranged as follows (pointers are positioned at 0):
  //initially: -last_var_count,...,-2,-1,0,+1,+2,...,+last_var_count
  //expanded: -new_var_count,-last_var_count,...,-2,-1,0,+1,+2,...,+last_var_count,+new_var_count
  
  //position pointer at beginning of array
  manager->literals -= last_var_count;
  //increase size of array (by 2 cells)
  REALLOC(manager->literals,SddNode*,new_array_size,"add_var_to_manager");
  //shift entries of array forward by one cell
  memmove(1+manager->literals,manager->literals,last_array_size*sizeof(SddNode*));
  //position pointer back at 0
  manager->literals += new_var_count;
  
  //expand the index of leaf vtrees to hold an entry for the new leaf vtree
  //NULL, 1, 2, ..., new_var
  REALLOC(manager->leaf_vtrees,Vtree*,1+new_var_count,"add_var_to_manager");

  //initialization for newly constructed nodes
  
  //construct literal sdds for new leaf node 
  //index new leaf vtree
  setup_literal_sdds(new_leaf,manager);   
}


//
//in all of the following functions:
//
//--index of added variable is 1+largest index of variables in manager
//--before: means left sibling of
//--after : means right sibling of

//before the first variable in the vtree inorder
void sdd_manager_add_var_before_first(SddManager* manager) {
  Vtree* sibling = first_leaf_vtree(manager->vtree);
  add_var_to_manager('l',sibling,manager);
}

//after the last variable in the vtree inorder
void sdd_manager_add_var_after_last(SddManager* manager) {
  Vtree* sibling = last_leaf_vtree(manager->vtree);
  add_var_to_manager('r',sibling,manager);
}

//before the root of the vtree
void add_var_before_top(SddManager* manager) {
  Vtree* sibling = manager->vtree;
  add_var_to_manager('l',sibling,manager);
}

//after the root of the vtree
void add_var_after_top(SddManager* manager) {
  Vtree* sibling = manager->vtree;
  add_var_to_manager('r',sibling,manager);
}

//before var
void sdd_manager_add_var_before(SddLiteral target_var, SddManager* manager) {
  Vtree* sibling = sdd_manager_vtree_of_var(target_var,manager);
  add_var_to_manager('l',sibling,manager);
}

//after var
void sdd_manager_add_var_after(SddLiteral target_var, SddManager* manager) {
  Vtree* sibling = sdd_manager_vtree_of_var(target_var,manager);
  add_var_to_manager('r',sibling,manager);
}
 
//variables is an array of size count containing the variables 
//before the least common ancestor (lca) of the given variables
void add_var_before_lca(int count, SddLiteral* variables, SddManager* manager) {
  Vtree* sibling = sdd_manager_lca_of_literals(count,variables,manager);
  add_var_to_manager('l',sibling,manager);
}

//variables is an array of size count containing the variables
//after the least common ancestor (lca) of the given variables
void add_var_after_lca(int count, SddLiteral* variables, SddManager* manager) {
  Vtree* sibling = sdd_manager_lca_of_literals(count,variables,manager);
  add_var_to_manager('r',sibling,manager);
}


/****************************************************************************************
 * moving var in a manager
 *
 * assumes:
 * --the manager vtree has at least two variables
 * --the variable is not used
 *
 ****************************************************************************************/

//
//in all of the following functions:
//
//--before: means left sibling of
//--after : means right sibling of

//before first variable in vtree inorder
void move_var_before_first(SddLiteral var, SddManager* manager) {
  Vtree* sibling = first_leaf_vtree(manager->vtree);
  move_var_in_vtree(var,'l',sibling,manager);
}

//after last variable in vtree inorder
void move_var_after_last(SddLiteral var, SddManager* manager) {
  Vtree* sibling = last_leaf_vtree(manager->vtree);
  move_var_in_vtree(var,'r',sibling,manager);
}

//before target_var
void move_var_before(SddLiteral var, SddLiteral target_var, SddManager* manager) {
  Vtree* sibling = sdd_manager_vtree_of_var(target_var,manager);
  move_var_in_vtree(var,'l',sibling,manager);
}

//after target_var
void move_var_after(SddLiteral var, SddLiteral target_var, SddManager* manager) {
  Vtree* sibling = sdd_manager_vtree_of_var(target_var,manager);
  move_var_in_vtree(var,'r',sibling,manager);
}
 
 
/****************************************************************************************
 * remove last var from manager
 *
 * index of removed variable is largest index of variables in manager
 *
 * assumes:
 * --variable is not used
 * --manager has at least two variables
 *
 ****************************************************************************************/
 
void remove_var_added_last(SddManager* manager) {

  CHECK_ERROR(manager->var_count<=1,ERR_MSG_TWO_VARS,"remove_last_var");
  CHECK_ERROR(sdd_manager_is_var_used(manager->var_count,manager),ERR_MSG_REM_VAR,"remove_last_var");

  SddLiteral last_var_count = manager->var_count;
  SddLiteral new_var_count  = --manager->var_count;
  assert(new_var_count!=0);
  
  //nodes to be removed from vtree
  Vtree* removed_leaf     = sdd_manager_vtree_of_var(last_var_count,manager);
  assert(removed_leaf->parent);
  assert(removed_leaf->parent->nodes==NULL);
    
  //gc literal sdds for removed leaf (var)
  //these are not "freed" as they may still exist in some computed cache and, hence, the
  //test INVALID_COMPUTED may end up accessing them
  gc_sdd_node(removed_leaf->nodes->vtree_next,manager); //negative literal (must be gc'd first)
  gc_sdd_node(removed_leaf->nodes,manager); //positive literal (must be gc'd second)
   
  //remove leaf node (of last variable) and corresponding internal node from vtree
  remove_var_from_vtree(last_var_count,manager);
  
  //shrink literals array to delete entries for removed variable

  SddLiteral new_array_size  = 1+2*new_var_count;
  //literal indices are arranged as follows (pointers are positioned at 0):
  //initially: -last_var_count,...,-2,-1,0,+1,+2,...,+last_var_count
  //shrunk   : -new_var_count ,...,-2,-1,0,+1,+2,...,+new_var_count
  
  //position pointer at beginning of array
  manager->literals -= last_var_count;
  //shift entries of array one cell backwards
  memmove(manager->literals,1+manager->literals,new_array_size*sizeof(SddNode*));
  //shrink size of array (by 2 cells)
  REALLOC(manager->literals,SddNode*,new_array_size,"remove_last_var");
  //reposition pointer at 0
  manager->literals += new_var_count;
  
  //shrink the index of leaf vtrees to delete the entry for the removed leaf vtree
  //NULL, 1, 2, ..., last_var-1
  REALLOC(manager->leaf_vtrees,Vtree*,1+new_var_count,"remove_last_var");
  
}

 
/****************************************************************************************
 * end
 ****************************************************************************************/
