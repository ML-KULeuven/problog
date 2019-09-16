/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/nodes.c
void insert_in_unique_table(SddNode* node, SddManager* manager);
void inc_element_parent_counts(SddSize size, SddElement* elements, SddManager* manager);
void dec_element_parent_counts(SddSize size, SddElement* elements, SddManager* manager);

//basic/replace.c
void reverse_node_replacement(SddNode* node, Vtree* vtree, SddManager* manager);
void confirm_node_replacement(SddNode* node, SddManager* manager);
    
/****************************************************************************************
 * moves nodes to a new vtree
 ****************************************************************************************/

//replace the vtree of nodes
//insert nodes in unique table
//nodes is a linked list
static inline
void move_to_vtree(SddNode* nodes, Vtree* vtree, SddManager* manager) {
  FOR_each_linked_node(n,nodes,{ //move to nodes of vtree
    n->vtree = vtree;
    insert_in_unique_table(n,manager);
  });
}

/****************************************************************************************
 * confirm nodes replacement as rollback is not longer needed
 ****************************************************************************************/

//free saved elements of replaced_nodes
//insert replaced_nodes into unique table (nodes already normalized for vtree)
//move moved_nodes to vtree
//replaced_nodes and moved_nodes are a linked list
void finalize_vtree_op(SddNode* replaced_nodes, SddNode* moved_nodes, Vtree* vtree, SddManager* manager) {  
  FOR_each_linked_node(n,replaced_nodes,{
    assert(n->replaced);
    confirm_node_replacement(n,manager); //frees saved elements
    insert_in_unique_table(n,manager);
  });
  move_to_vtree(moved_nodes,vtree,manager);
}

/****************************************************************************************
 * undo node replacement due a rollback
 ****************************************************************************************/
 
//reverse the replacement of replaced_nodes (if any)
//insert replaced_nodes back into unique table
//move moved_nodes to vtree
//replaced_nodes and moved_nodes are a linked list
void rollback_vtree_op(SddNode* replaced_nodes, SddNode* moved_nodes, Vtree* vtree, SddManager* manager) {
  FOR_each_linked_node(n,replaced_nodes,{
    if(n->replaced) reverse_node_replacement(n,vtree,manager); //not all nodes may have been replaced
    insert_in_unique_table(n,manager);
  });
  move_to_vtree(moved_nodes,vtree,manager);
}
 
/****************************************************************************************
 * END
 ****************************************************************************************/
