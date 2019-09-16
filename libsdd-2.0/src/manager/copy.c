/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

/****************************************************************************************
 * copying manager
 *
 * no external references to nodes in copied manager (all nodes are dead)
 * new manager has defaults (e.g., auto-mode, etc)
 ****************************************************************************************/

static
void initialize_decomposition_and_literal_maps(Vtree* from_vtree, SddManager* from_manager, SddManager* into_manager) {
  if(LEAF(from_vtree)) {
    SddLiteral var = from_vtree->var;
    sdd_manager_literal(+var,from_manager)->map = sdd_manager_literal(+var,into_manager);
    sdd_manager_literal(-var,from_manager)->map = sdd_manager_literal(-var,into_manager);
  }
  else {
    FOR_each_sdd_node_normalized_for(node,from_vtree,node->map = NULL);
    initialize_decomposition_and_literal_maps(from_vtree->left,from_manager,into_manager);
    initialize_decomposition_and_literal_maps(from_vtree->right,from_manager,into_manager);
  }
}

static 
void copy_decomposition_nodes(Vtree* from_vtree, Vtree* into_vtree, SddManager* into_manager) {
  if(LEAF(from_vtree)) {
    assert(LEAF(into_vtree));
  }
  else {
    //bottom up copying
    copy_decomposition_nodes(from_vtree->left,into_vtree->left,into_manager);
    copy_decomposition_nodes(from_vtree->right,into_vtree->right,into_manager);
    
    FOR_each_sdd_node_normalized_for(node,from_vtree,{
      GET_node_from_compressed_partition(node->map,into_vtree,into_manager,{
        FOR_each_prime_sub_of_node(prime,sub,node,{
          DECLARE_compressed_element(prime->map,sub->map,into_vtree,into_manager);
        });
      });
    });
  }
}

//return a copy of manager
//nodes are in manager and there are size of them
//replace these nodes by their copies in the new manager
SddManager* sdd_manager_copy(SddSize size, SddNode** nodes, SddManager* from_manager) {

  Vtree* from_vtree        = from_manager->vtree;
  SddManager* into_manager = sdd_manager_new(from_vtree);
  Vtree* into_vtree        = into_manager->vtree;
  //literal and constant sdds already copied
  
  //initialize node maps
  sdd_manager_true(from_manager)->map  = sdd_manager_true(into_manager);
  sdd_manager_false(from_manager)->map = sdd_manager_false(into_manager);
  initialize_decomposition_and_literal_maps(from_vtree,from_manager,into_manager);  
  
  //copy decomposition nodes of manager into new_manager
  copy_decomposition_nodes(from_vtree,into_vtree,into_manager);
  
  assert(from_manager->node_count==into_manager->node_count);  
  assert(from_manager->sdd_size==into_manager->sdd_size);
  assert(into_manager->node_count==into_manager->dead_node_count);
  assert(into_manager->sdd_size==into_manager->dead_sdd_size);
  
  //replace manager nodes by their copies in new manager
  for(SddSize i=0; i<size; i++) {
    assert(nodes[i] && nodes[i]->map);
    nodes[i] = nodes[i]->map;
  }
  
  return into_manager;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
