/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//local declarations
static void sdd_minimum_cardinality_aux(SddNode* node, SddLiteral* min_cards, SddLiteral** min_cards_loc);
static void sdd_minimize_cardinality_aux(SddNode* node, SddLiteral* min_cards, int* minimize_bits, SddNode** minimized_nodes, SddNode*** minimized_nodes_loc, SddManager* manager);
static SddNode* negative_term(Vtree* vtree, SddManager* manager);
static SddNode* add_negative_term(SddNode* node, Vtree* vtree, SddManager* manager);

/****************************************************************************************
 * minimum cardinality 
 *
 * cardinality of a model is the number of variables set to true in the model
 * minimum cardinality of an sdd: is the minimum cardinality attained by any of its models
 *
 * Note: model is defined over the variables appearing in the given sdd 
 ****************************************************************************************/

//returns the minimum cardinality of an sdd
//returns -1 for false sdd
SddLiteral sdd_minimum_cardinality(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_minimum_cardinality");
  assert(!GC_NODE(node));
 
  //find how many nodes
  SddSize size = sdd_all_node_count_leave_bits_1(node);
  //all nodes are marked 1 now

  //create array to hold minimum cardinality of sdd nodes
  SddLiteral* min_cards;
  CALLOC(min_cards,SddLiteral,size,"sdd_minimum_cardinality");
  
  //compute minimum cardinalities
  sdd_minimum_cardinality_aux(node,min_cards,&min_cards);
  //all nodes are marked 0 now
  min_cards -= size;
  
  SddLiteral minimum_cardinality = min_cards[node->index];
	
  free(min_cards);
	
  return minimum_cardinality;
}


//computes minimum cardinalities and stores results in min_cards array
void sdd_minimum_cardinality_aux(SddNode* node, SddLiteral* min_cards, SddLiteral** min_cards_loc) {	

  if(node->bit==0) return;  //node has been visited before
  else node->bit=0; //this is the first visit to this node
  
  SddLiteral min_card;

  if(node->type==FALSE) min_card = -1; //infinity
  else if(node->type==TRUE) min_card = 0; 
  else if(node->type==LITERAL && LITERAL_OF(node) > 0) min_card = 1;
  else if(node->type==LITERAL && LITERAL_OF(node) < 0) min_card = 0;
  else { //decomposition
    min_card = -1; //infinity
	FOR_each_prime_sub_of_node(prime,sub,node,{  
	  //recursive calls
	  sdd_minimum_cardinality_aux(prime,min_cards,min_cards_loc);
	  sdd_minimum_cardinality_aux(sub,min_cards,min_cards_loc);
	  //update minimum cardinality
	  if(min_cards[sub->index]!=-1) { //false sub
        SddLiteral element_min_card = min_cards[prime->index]+min_cards[sub->index];
	    if(min_card==-1 || element_min_card < min_card) min_card = element_min_card;
	  }
	});
  }

  //cache computed value
  **min_cards_loc = min_card;
  
  //save location of cached value
  node->index = *min_cards_loc-min_cards;

  //move to next cell in array
  ++(*min_cards_loc);
 
}

//mark each node that is part of an element having minimum cardinality in some decomposition
void mark_nodes_needing_minimization(SddNode* node, SddLiteral* min_cards, int* minimize_bits, int** minimize_bits_loc) {	

  if(node->bit==1) return; //node has been visited before
  else node->bit=1; //this is the first visit to this node

  if(node->type==DECOMPOSITION) { 
    SddLiteral min_card = min_cards[node->index];
	FOR_each_prime_sub_of_node(prime,sub,node,{  
	  //recursive calls
	  mark_nodes_needing_minimization(prime,min_cards,minimize_bits,minimize_bits_loc);
	  mark_nodes_needing_minimization(sub,min_cards,minimize_bits,minimize_bits_loc);
	  //check
	  if(min_cards[sub->index]!=-1) { //false sub
        SddLiteral element_min_card = min_cards[prime->index]+min_cards[sub->index];
	    if(element_min_card==min_card) {
	      //this element will need to be minimized
	      minimize_bits[prime->index] = 1;
	      minimize_bits[sub->index]   = 1;
	    }
	  }
	});
  }
  
  //move to next cell in array
  ++(*minimize_bits_loc);
 
}

/****************************************************************************************
 * minimize cardinality
 *
 * generate an sdd which has only minimum cardinality models
 *
 * Note: minimization is done with respect to the variables that appear in the node
 *
 ****************************************************************************************/

SddNode* sdd_minimize_cardinality(SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_minimize_cardinality");
  assert(!GC_NODE(node));
  
  if(IS_FALSE(node) || IS_TRUE(node)) return node;
   
  //mark vtree nodes depending on whether their variables appear in the sdd
  set_sdd_variables(node,manager);
  
  //find how many nodes
  SddSize size = sdd_all_node_count_leave_bits_1(node);
  //all nodes are marked 1 now

  //create array to hold minimum cardinality of sdd nodes
  SddLiteral* min_cards;
  CALLOC(min_cards,SddLiteral,size,"sdd_minimize_cardinality");
  
  //create array to hold bits of sdd nodes needing minimization
  int* minimize_bits;
  CALLOC(minimize_bits,int,size,"sdd_minimize_cardinality");
  
  //create array to hold minimizations of sdd nodes
  SddNode** minimized_nodes;
  CALLOC(minimized_nodes,SddNode*,size,"sdd_minimize_cardinality");
  
  //compute minimum cardinalities
  sdd_minimum_cardinality_aux(node,min_cards,&min_cards);
  //all nodes are marked 0 now
  min_cards -= size;
  
  //mark nodes needing minimization
  mark_nodes_needing_minimization(node,min_cards,minimize_bits,&minimize_bits);
  //all nodes marked 1 now
  minimize_bits -= size;
  minimize_bits[node->index] = 1; //root needs minimization
  
  //minimize
  SddNode* minimized_node;
  WITH_no_auto_mode(manager,{
    sdd_minimize_cardinality_aux(node,min_cards,minimize_bits,minimized_nodes,&minimized_nodes,manager);
    //all nodes are maked 0 now
    minimized_nodes -= size;
    minimized_node = minimized_nodes[node->index];
  });
	
  free(min_cards);
  free(minimized_nodes);
  free(minimize_bits);
	
  return minimized_node;
}


//compute minimized sdds and their carries and store in corresponding arrays
void sdd_minimize_cardinality_aux(SddNode* node, SddLiteral* min_cards, int* minimize_bits, SddNode** minimized_nodes, SddNode*** minimized_nodes_loc, SddManager* manager) {	

  if(node->bit==0) return; //node has been visited before
  else node->bit=0; //this is the first visit to this node

  SddNode* minimized_node = NULL;
  
  if(node->type==FALSE) minimized_node = sdd_manager_false(manager);
  else if(node->type==TRUE) assert(1); //nothing to do here
  else if(node->type==LITERAL) minimized_node = node;
  else { //decomposition    
  
    //recursive calls
	FOR_each_prime_sub_of_node(prime,sub,node,{  
	  sdd_minimize_cardinality_aux(prime,min_cards,minimize_bits,minimized_nodes,minimized_nodes_loc,manager);
	  sdd_minimize_cardinality_aux(sub,min_cards,minimize_bits,minimized_nodes,minimized_nodes_loc,manager);
	});
	
	if(minimize_bits[node->index]) {
	  //node is part of an element that has minimum cardinality in some decomposition node
	  
	  SddLiteral min_card = min_cards[node->index]; //already computed
	  Vtree* vtree = node->vtree;
  
	  //compute minimization
	  GET_node_from_partition(minimized_node,vtree,manager,{
	    FOR_each_prime_sub_of_node(prime,sub,node,{
	      SddSize pi = prime->index;
	      SddSize si = sub->index;
	    
	      if(min_cards[si]==-1 || min_cards[pi]+min_cards[si] > min_card) {
	        DECLARE_element(prime,sdd_manager_false(manager),vtree,manager);
	      }
	      else {
	        SddNode* minimized_prime = add_negative_term(minimized_nodes[pi],vtree->left,manager);
	        SddNode* minimized_sub   = (IS_TRUE(sub)? negative_term(vtree->right,manager):
	                                   add_negative_term(minimized_nodes[si],vtree->right,manager));
	     
	        DECLARE_element(minimized_prime,minimized_sub,vtree,manager);
	      
	        SddNode* neg_minimized_prime = sdd_negate(minimized_prime,manager);
	        SddNode* prime_carry = sdd_apply(prime,neg_minimized_prime,CONJOIN,manager);
	        
	        if(!IS_FALSE(prime_carry)) DECLARE_element(prime_carry,sdd_manager_false(manager),vtree,manager);
	      }
	    });
	  });
	}
	
  }

  //cache computed values
  **minimized_nodes_loc = minimized_node; //may be NULL
  
  //move to next cell in array
  ++(*minimized_nodes_loc);

}

// minimize cardinality relative to all variables, in contrast to used
// variables as in sdd_minimize_cardinality
SddNode* sdd_global_minimize_cardinality(SddNode* node, SddManager* manager) {
  if(node->type==FALSE) return sdd_manager_false(manager);
  SddNode* minimized_node = sdd_minimize_cardinality(node,manager);

  int* vars = sdd_variables(node,manager);
  SddLiteral var_count = sdd_manager_var_count(manager);
  SddNode* term = sdd_manager_true(manager);
  WITH_no_auto_mode(manager,{
    for (SddLiteral var = 1; var <= var_count; var++)
      if ( vars[var] == 0 ) { // unused var, include negative literal
        term = sdd_apply(term,sdd_manager_literal(-var,manager),CONJOIN,manager);
      }
  });
  free(vars);
  minimized_node = sdd_apply(minimized_node,term,CONJOIN,manager);

  return minimized_node;
}


/****************************************************************************************
 * computing term sdds over vtree variables that appear in the sdd being minimized
 *
 * these are needed to handle 
 * --true subs: minimization depends on where they appear (in which decomposition node)
 * --primes not normalized for vtree->left: gap between vtree->right and sub->vtree
 * --subs not normalized for vtree->right: gap between vtree->right and sub->vtree
 *
 ****************************************************************************************/
 
//returns a term (sdd) corresponding to the negative literals of sdd variables in vtree
SddNode* negative_term(Vtree* vtree, SddManager* manager) {
  if(vtree->no_var_in_sdd) return sdd_manager_true(manager); 
  else if(LEAF(vtree)) return sdd_manager_literal(-(vtree->var),manager);
  else {
    SddNode* left  = negative_term(vtree->left,manager);
    SddNode* right = negative_term(vtree->right,manager);
    return sdd_apply(left,right,CONJOIN,manager);
  }
}

//returns a term (sdd) corresponding to the ngative literals of sdd variables that appear in vtree but not in sub_vtree
SddNode* gap_negative_term(Vtree* vtree, Vtree* sub_vtree, SddManager* manager) {
  if(vtree==sub_vtree) return sdd_manager_true(manager);
  
  SddNode* left;
  SddNode* right;
  if(sdd_vtree_is_sub(sub_vtree,vtree->left)) {
    left  = gap_negative_term(vtree->left,sub_vtree,manager);
    right = negative_term(vtree->right,manager);
  }
  else {
    left  = negative_term(vtree->left,manager);
    right = gap_negative_term(vtree->right,sub_vtree,manager);
  }
  return sdd_apply(left,right,CONJOIN,manager);
  
}

//conjoins node with a term (sdd) corresponding to the negative literals of sdd variables that 
//appear in vtree but not in node->vtree
//assumes node->vtree is in vtree
SddNode* add_negative_term(SddNode* node, Vtree* vtree, SddManager* manager) {
  if(IS_FALSE(node) || vtree==node->vtree) return node;
  else {
	SddNode* gap = gap_negative_term(vtree,node->vtree,manager);
	return sdd_apply(gap,node,CONJOIN,manager);
  } 
}

/****************************************************************************************
 * end
 ****************************************************************************************/
