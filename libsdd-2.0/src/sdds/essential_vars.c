/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

/****************************************************************************************
 * identify variables that an sdd essentially depends on
 ****************************************************************************************/

static void sdd_variables_aux(SddNode* node, int* dependence_map);

//returns a is_sdd_var, which is an array with the following properties:
//size               : 1+number of variables in manager
//is_sdd_var[var]: 1 if the sdd depends on var, 0 otherwise
//is_sdd_var[0]  : not used

int* sdd_variables(SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_variables");
  assert(!GC_NODE(node));
  
  //allocate dependence map
  int* is_sdd_var;
  CALLOC(is_sdd_var,int,1+manager->var_count,"sdd_variables");
  //all cells initialized to 0
  
  sdd_variables_aux(node,is_sdd_var);
  //all nodes are maked 1 now
  
  sdd_clear_node_bits(node);
  //all nodes are marked 0 now
  
  return is_sdd_var;
}

//set dependence_map[var] to 1 if the node depends on var
void sdd_variables_aux(SddNode* node, int* is_sdd_var) {	

  if(node->bit==1) return; //node has been visited before
  else node->bit=1; //this is the first visit to this node

  if(IS_LITERAL(node)) {
    SddLiteral var = VAR_OF(node);
    is_sdd_var[var] = 1; //node essentially depends on this variable
  }
  else if(IS_DECOMPOSITION(node)) {
    FOR_each_prime_sub_of_node(prime,sub,node,{
      //recursive calls on descendants
      sdd_variables_aux(prime,is_sdd_var);
	  sdd_variables_aux(sub,is_sdd_var);
	});
  }
  
}


/****************************************************************************************
 * v->all_vars_in_sdd: all vars of vtree v appear in sdd
 * v->no_var_in_sdd : no var of vtree v appears in sdd
 *
 * sets the above flags for each vtree node v in node->vtree 
 ****************************************************************************************/

static void set_sdd_variables_aux(SddNode* node);
static void initialize_sdd_variables(Vtree* vtree);
static void propagate_sdd_variables(Vtree* vtree);

void set_sdd_variables(SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"set_no_var_in_sdd");
  assert(!GC_NODE(node));
  
  initialize_sdd_variables(manager->vtree);
  
  //these nodes have no vtree
  if(IS_FALSE(node) || IS_TRUE(node)) return;
  
  set_sdd_variables_aux(node);
  //all nodes are maked 1 now
    
  sdd_clear_node_bits(node);
  //all nodes are marked 0 now
  
  propagate_sdd_variables(node->vtree);
}

static
void set_sdd_variables_aux(SddNode* node) {	

  if(node->bit==1) return; //node has been visited before
  else node->bit=1; //this is the first visit to this node

  if(IS_LITERAL(node)) {
    node->vtree->all_vars_in_sdd = 1;
    node->vtree->no_var_in_sdd   = 0;
  }
  else if(IS_DECOMPOSITION(node)) {
    FOR_each_prime_sub_of_node(prime,sub,node,{
      //recursive calls on descendants
      set_sdd_variables_aux(prime);
	  set_sdd_variables_aux(sub);
	});
  }
  
}

static
void initialize_sdd_variables(Vtree* vtree) {
  FOR_each_vtree_node(v,vtree,{
    v->all_vars_in_sdd = 0;
    v->no_var_in_sdd   = 1;
  });
}

static
void propagate_sdd_variables(Vtree* vtree) {
  if(INTERNAL(vtree)) {
    propagate_sdd_variables(vtree->left);
    propagate_sdd_variables(vtree->right);
    vtree->all_vars_in_sdd = vtree->left->all_vars_in_sdd && vtree->right->all_vars_in_sdd;
    vtree->no_var_in_sdd   = vtree->left->no_var_in_sdd && vtree->right->no_var_in_sdd;
  }
}

/****************************************************************************************
 * end
 ****************************************************************************************/
