/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//local declarations
void sdd_rename_variables_aux(SddNode* node, SddLiteral* variable_map, SddManager* manager);
static void initialize_map(SddNode* node, SddLiteral* variable_map);

/****************************************************************************************
 * given an sdd, and a variable map,
 * construct a new sdd by mapping the variables of the original sdd into new ones
 *
 * will not do auto gc/minmize as its computations are done in no-auto mode
 ****************************************************************************************/

//returns an sdd which is obtained by renaming variables in the original sdd (node)
//variable_map is an array with the following properties:
//size             : 1+number of variables in manager
//variable_map[var]: is a variable into which var is mapped
//variable_map[0]  : not used

SddNode* sdd_rename_variables(SddNode* node, SddLiteral* variable_map, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_rename_variables");
  assert(!GC_NODE(node));
  
  if(node->type==FALSE || node->type==TRUE) return node;
  //node is not trivial

  WITH_no_auto_mode(manager,{
    initialize_map(node,variable_map);
    sdd_rename_variables_aux(node,variable_map,manager);
  });
  
  return node->map;
}

//compute node maps and store them in the map field
void sdd_rename_variables_aux(SddNode* node, SddLiteral* variable_map, SddManager* manager) {	

  if(node->map!=NULL) return;
  
  SddNode* node_map;
  
  if(node->type==FALSE || node->type==TRUE) node_map = node;
  else if(node->type==LITERAL) { //must be renamed
    SddLiteral old_literal = LITERAL_OF(node);
    SddLiteral old_var     = VAR_OF(node);
    SddLiteral new_var     = variable_map[old_var];
    assert(new_var!=old_var);
    assert(1<= old_var && old_var <= sdd_manager_var_count(manager));
    assert(1<= new_var && new_var <= sdd_manager_var_count(manager));
    SddLiteral new_literal = old_literal>0? new_var: -new_var;
    node_map = sdd_manager_literal(new_literal,manager);
  }
  else { //decomposition
    node_map = manager->false_sdd;
    FOR_each_prime_sub_of_node(prime,sub,node,{
	  sdd_rename_variables_aux(prime,variable_map,manager);
	  sdd_rename_variables_aux(sub,variable_map,manager);
	  SddNode* element_map = sdd_apply(prime->map,sub->map,CONJOIN,manager);
	  node_map = sdd_apply(node_map,element_map,DISJOIN,manager);
	});
  }
  
  node->map = node_map;
  
}

/****************************************************************************************
 * initialized the mapped field of nodes:
 *
 * node->map=node if node does not include a renamed var
 * node->map=NULL otherwise
 ****************************************************************************************/

//this function will leave all bits of nodes set to 1
static inline
void initialize_map_aux(SddNode* node, SddLiteral* variable_map) {
  if (node->bit) return; //node visited before
  //this is the first visit for this node
  node->bit=1;
  
  node->map = NULL; //default
  
  if(node->type==FALSE || node->type==TRUE) node->map = node;
  else if(node->type==LITERAL) {
    SddLiteral var = VAR_OF(node);
    if(variable_map[var]==var) node->map = node;
  }
  else { //decomposition
    int rename = 0;
    FOR_each_prime_sub_of_node(prime,sub,node,{
      initialize_map_aux(prime,variable_map);
	  initialize_map_aux(sub,variable_map);
	  rename = rename || prime->map==NULL || sub->map==NULL;
	});
    if(rename==0) node->map = node; 
  }
}

static
void initialize_map(SddNode* node, SddLiteral* variable_map) {
  initialize_map_aux(node,variable_map);
  sdd_clear_node_bits(node);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
