/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//local declarations
static void initialize_sdd(SddNode* node, int* exists_map, SddManager* manager);
static void quantify_sdd(SddNode* node, SddManager* manager);

/****************************************************************************************
 * existentially quantify a number of variables from an sdd
 *
 * this version will NOT invoke auto gc and minimize
 ****************************************************************************************/

//exists_map is an array with the following properties:
//size             : 1+number of variables in manager
//exists_map[var]  : is 1 if var is to be existentially quantified, 0 otherwise
//exists_map[0]    : not used

SddNode* sdd_exists_multiple_static(int* exists_map, SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_exists_multiple_static");
  assert(!GC_NODE(node));
  
  if(node->type==FALSE || node->type==TRUE) return node;

  initialize_sdd(node,exists_map,manager); 
  
  WITH_no_auto_mode(manager,{ //no gc or minimize
    quantify_sdd(node,manager);
  });

  return node->map;
}

/****************************************************************************************
 * set node->map set to NULL if node contains quantified variables, set to node otherwise
 ****************************************************************************************/

//this function will leave all bits set to 1 --- bits must be cleared (see bits.c)
static
void initialize(SddNode* node, int* exists_map, SddManager* manager) {
  if(node->bit) return; //already visited
  else node->bit = 1; //first visit
  
  node->map = NULL; //default
  
  if(node->type==TRUE || node->type==FALSE) node->map = node;
  else if(node->type==LITERAL) {
    SddLiteral var = VAR_OF(node);
    if(exists_map[var]==0) node->map = node;
  }
  else {
    int q_vars = 0;
    FOR_each_prime_sub_of_node(prime,sub,node,{
      initialize(prime,exists_map,manager);
      initialize(sub,exists_map,manager);
      q_vars = q_vars || prime->map==NULL || sub->map==NULL;
    });
    if(q_vars==0) node->map = node;
  }
  
}

static void initialize_sdd(SddNode* node, int* exists_map, SddManager* manager) {
  initialize(node,exists_map,manager); //leave bits 1
  sdd_clear_node_bits(node);
}
  
/****************************************************************************************
 * quantifying nodes
 ****************************************************************************************/

//elements form an xy-partition (may not be compressed)
static
SddNode* quantify_from_partition(SddNodeSize size, SddElement* elements, Vtree* vtree, SddManager* manager) {
  SddNode* q_node;
  
  GET_node_from_partition(q_node,vtree,manager,{
    for(SddElement* e=elements; e<elements+size; e++) {
      DECLARE_element(e->prime,e->sub,vtree,manager);
    }
  });
  
  return q_node;
}

//elements form an xy-decomposition
static
SddNode* quantify_from_decomposition(SddNodeSize size, SddElement* elements, SddManager* manager) {

  SddNode* q_node = manager->false_sdd;
  
  for(SddElement* e=elements; e<elements+size; e++) {
    SddNode* x_node = e->prime;
    SddNode* y_node = e->sub;
    SddNode* element = sdd_apply(x_node,y_node,CONJOIN,manager);
	q_node = sdd_apply(element,q_node,DISJOIN,manager); 
  }
  
  return q_node; 
}  

//returned elements are not a compressed parition, but just x-nodes and y-nodes for some x and y
//reasons: quantification weakens nodes, potentially destroying disjointness (primes) or distinction (subs)
static
SddElement* get_quantified_elements(SddNode* node) {
  assert(node->type==DECOMPOSITION);

  SddNodeSize size     = node->size;
  SddElement* elements = ELEMENTS_OF(node);
  
  SddElement* q_elements;
  CALLOC(q_elements,SddElement,size,"get_quantified_elements");
  
  for(SddNodeSize i=0; i<size; i++) {
    //quantified primes may not form a partition
    q_elements[i].prime = elements[i].prime->map;
    //quantified subs may not be distinct
    q_elements[i].sub   = elements[i].sub->map;
  }
  
  return q_elements;
}

static
void quantify_sdd(SddNode* node, SddManager* manager) {
  if(node->map) return; //node already quantified
  assert(!TRIVIAL(node));
  
  if(node->type==LITERAL) node->map = manager->true_sdd;
  else { 
    
    int is_true      = 0;
    int is_partition = 1;
    FOR_each_prime_sub_of_node(prime,sub,node,{
      quantify_sdd(prime,manager); //prime->map is set now
      quantify_sdd(sub,manager); //sub->map is set now
      is_true      = is_true || (IS_TRUE(prime->map) && IS_TRUE(sub->map));
      is_partition = is_partition && prime==prime->map;
    });

    if(is_true) { //optimization
      node->map = manager->true_sdd;
      return;
    }
    
    Vtree* vtree           = node->vtree;    
    SddSize size           = node->size;
    SddElement* q_elements = get_quantified_elements(node);
    
    if(is_partition) node->map = quantify_from_partition(size,q_elements,vtree,manager);
    else node->map = quantify_from_decomposition(size,q_elements,manager);
    
    free(q_elements);
  }
  
}

/****************************************************************************************
 * end
 ****************************************************************************************/
