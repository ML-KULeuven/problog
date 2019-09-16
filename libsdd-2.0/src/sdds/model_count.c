/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//local declarations
static void sdd_model_count_aux(SddNode* node, SddModelCount* start, SddModelCount** model_counts);
static SddLiteral var_count(Vtree* vtree);	
static SddLiteral gap_var_count(Vtree* vtree, Vtree* sub_vtree);

/****************************************************************************************
 * number of models for an sdd
 ****************************************************************************************/
 
//NOTE: when node is trivial, we don't know the vtree for which the node is normalized
//we assume that it is normalized for the vtree root (manager->vtree)

//count the models of an sdd
SddModelCount sdd_model_count(SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_model_count");
  assert(!GC_NODE(node));
  
  if(node->type==FALSE) return 0;
  if(node->type==TRUE) return 1; 
  
  //mark vtree nodes depending on whether their variables appear in the sdd
  set_sdd_variables(node,manager);
  
  //find how many nodes
  SddSize size = sdd_all_node_count_leave_bits_1(node);
  //all nodes are marked 1 now

  //create array to hold model counts of sdd nodes
  SddModelCount* model_counts;
  CALLOC(model_counts,SddModelCount,size,"sdd_model_count");

  sdd_model_count_aux(node,model_counts,&model_counts);
  //all nodes are maked 0 now
  
  model_counts -= size;
  //model_counts is again pointing to first cell in allocated array
  
  SddModelCount model_count = model_counts[node->index];
  free(model_counts);
	
  return model_count;
}


//computes model counts and stores them in the associated array
static
void sdd_model_count_aux(SddNode* node, SddModelCount* start, SddModelCount** model_counts_loc) {	

  if(node->bit==0) return; //node has been visited before
  else node->bit=0; //this is the first visit to this node

  SddModelCount mc = 0;
  
  if(node->type==FALSE) mc = 0; //node must be a sub
  else if(node->type==TRUE) ; //node must be a sub, case not used
  else if(node->type==LITERAL) mc = 1;
  else { //decomposition
	FOR_each_prime_sub_of_node(prime,sub,node,{
	  //recursive calls
	  sdd_model_count_aux(prime,start,model_counts_loc);
	  sdd_model_count_aux(sub,start,model_counts_loc);
	  
	  //prime is neither true nor false, sub could be true or false
	  if(!IS_FALSE(sub)) {
	    Vtree* left  = node->vtree->left;
	    Vtree* right = node->vtree->right;
	  
	    SddModelCount prime_mc = 
	      start[prime->index]*pow(2,gap_var_count(left,prime->vtree));
	    SddModelCount sub_mc = (IS_TRUE(sub)? pow(2,var_count(right)): 
	      start[sub->index]*pow(2,gap_var_count(right,sub->vtree)));
	    
	    mc += prime_mc*sub_mc;
	  }
	});
  }

  //saving model count
  **model_counts_loc = mc; 
  
  //location of saved model count
  node->index = *model_counts_loc-start;
   
  //advance to next cell in array
  (*model_counts_loc)++; 
  
}

// performs model count relative to all variables, in contrast to used
// variables as in sdd_model_count
SddModelCount sdd_global_model_count(SddNode* node, SddManager* manager) {
  SddModelCount mc = sdd_model_count(node,manager);

  // count unused variables
  int* vars = sdd_variables(node,manager);
  SddLiteral var_count = sdd_manager_var_count(manager);
  SddLiteral unused = 0;
  for (SddLiteral var = 1; var <= var_count; var++)
    if ( vars[var] == 0 ) unused += 1;
  free(vars);

  return mc << unused; // multiply by 2^unused
}

/****************************************************************************************
 * counting vtree variables that appear in the sdd
 *
 * these are needed to handle 
 * --true subs: variables of true sub depend on where the sub appears (in which decomposition node)
 * --primes not normalized for vtree->left: variables in vtree->right but not in sub->vtree
 * --subs not normalized for vtree->right: variables in vtree->right but not in sub->vtree
 *
 ****************************************************************************************/
 
//returns count of sdd variables in vtree
static
SddLiteral var_count(Vtree* vtree) {
  if(vtree->all_vars_in_sdd) return vtree->var_count;
  else if(vtree->no_var_in_sdd) return 0;
  else return var_count(vtree->left) + var_count(vtree->right);
}

//returns count of sdd variables that appear in vtree but not in sub_vtree
static
SddLiteral gap_var_count(Vtree* vtree, Vtree* sub_vtree) {
  if(vtree==sub_vtree) return 0;
  if(sdd_vtree_is_sub(sub_vtree,vtree->left)) {
    return gap_var_count(vtree->left,sub_vtree) + var_count(vtree->right);
  }
  else {
    return var_count(vtree->left) + gap_var_count(vtree->right,sub_vtree);
  }
}


/****************************************************************************************
 * end
 ****************************************************************************************/
