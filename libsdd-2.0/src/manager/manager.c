/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//declarations

//basic/hash.c
SddHash* new_unique_node_hash(SddManager* manager);
void free_hash(SddHash* hash);

//basic/nodes.c
void free_sdd_node(SddNode* node, SddManager* manager);
SddNode* construct_literal_sdd_node(SddLiteral literal, Vtree* vtree, SddManager* manager);
SddNode* construct_true_sdd_node(SddManager* manager);
SddNode* construct_false_sdd_node(SddManager* manager);

//vtrees/vtree.c
void set_vtree_properties(Vtree* vtree);

//local declarations
static void setup_terminal_sdds(SddManager* manager);

/****************************************************************************************
 * constructing manager
 ****************************************************************************************/

//used by interrupt signal
SddManager* last_constructed_manager = NULL;

//constructs a new manager and associated sdd nodes
SddManager* sdd_manager_new(Vtree* input_vtree) {

  CHECK_ERROR(input_vtree==NULL,ERR_MSG_INPUT_VTREE,"new_sdd_manager");
  
  //work with a copy of the passed vtree
  //input vtree will remain unchanged
  Vtree* vtree = copy_vtree(input_vtree);
  
  //positions, var counts, and linked list
  set_vtree_properties(vtree); 
  
  //allocate manager
  SddManager* manager;
  MALLOC(manager,SddManager,"new_sdd_manager");
  
  //initializations
  SddLiteral var_count = vtree->var_count;
  manager->vtree       = vtree;
  manager->id_counter  = 0;
  manager->var_count   = var_count;

  //basic counts
  manager->node_count      = 0;
  manager->dead_node_count = 0;
  manager->computed_count  = 0;

  //sizes
  manager->sdd_size      = 0;
  manager->dead_sdd_size = 0;

  //gc lists
  manager->gc_node_count     = 0;
  manager->gc_element_count  = 0;
  
  CALLOC(manager->gc_node_lists,SddNode*,GC_BUCKETS_COUNT,"new_sdd_manager");
  
  //unique nodes
  manager->unique_nodes = new_unique_node_hash(manager);
  
  //computation caches
  manager->computed_cache_lookup_count = 0;
  manager->computed_cache_hit_count    = 0;
  CALLOC(manager->conjoin_cache,SddComputed,COMPUTED_CACHE_SIZE,"new_sdd_manager");
  CALLOC(manager->disjoin_cache,SddComputed,COMPUTED_CACHE_SIZE,"new_sdd_manager");
  
  //apply
  manager->apply_depth = 0;
  manager->limited_apply_depth = 0;
  
  //literals and vtree leaves

  //indexing literal sdds: -var_count, ..., -2, -1, NULL, +1, +2, ..., +var_count
  CALLOC(manager->literals,SddNode*,1+2*var_count,"new_sdd_manager");
  manager->literals += var_count; //positioned at NULL
  
  //indexing leaf vtrees: NULL, 1, 2, ..., var_count
  CALLOC(manager->leaf_vtrees,Vtree *,1+var_count,"new_sdd_manager");
  
  //stacks
  CALLOC(manager->start_compression_stack,SddElement,INITIAL_SIZE_ELEMENT_STACK,"new_sdd_manager");
  manager->top_compression_stack = manager->start_compression_stack;
  manager->capacity_compression_stack = INITIAL_SIZE_ELEMENT_STACK;
  
  CALLOC(manager->start_cp_stack1,SddElement,INITIAL_SIZE_ELEMENT_STACK,"new_sdd_manager");
  manager->top_cp_stack1 = manager->start_cp_stack1;
  manager->capacity_cp_stack1 = INITIAL_SIZE_ELEMENT_STACK;
  
  CALLOC(manager->start_cp_stack2,SddElement,INITIAL_SIZE_ELEMENT_STACK,"new_sdd_manager");
  manager->top_cp_stack2 = manager->start_cp_stack2;
  manager->capacity_cp_stack2 = INITIAL_SIZE_ELEMENT_STACK;
  
  CALLOC(manager->start_cp_stack3,SddElement,INITIAL_SIZE_ELEMENT_STACK,"new_sdd_manager");
  manager->top_cp_stack3 = manager->start_cp_stack3;
  manager->capacity_cp_stack3 = INITIAL_SIZE_ELEMENT_STACK;
  
  CALLOC(manager->start_meta_compression_stack,SddSize,INITIAL_SIZE_COMPRESSION_STACK,"new_sdd_manager");
  manager->top_meta_compression_stack      = manager->start_meta_compression_stack;
  manager->capacity_meta_compression_stack = INITIAL_SIZE_COMPRESSION_STACK;
  
  CALLOC(manager->start_element_stack,SddElement,INITIAL_SIZE_ELEMENT_STACK,"new_sdd_manager");
  manager->top_element_stack      = manager->start_element_stack;
  manager->capacity_element_stack = INITIAL_SIZE_ELEMENT_STACK;
  
  CALLOC(manager->node_buffer,SddNode*,INITIAL_SIZE_NODE_BUFFER,"new_sdd_manager");
  manager->node_buffer_size = INITIAL_SIZE_NODE_BUFFER;  
  
  //manager options
  manager->options = NULL;
  
  //manager stats
  SddManagerStats stats = {0,0,0,0,0,0,0,0};
  manager->stats = stats;
  
  //vtree search  
  SddManagerVtreeOps vtree_ops = {VTREE_SEARCH_TIME_LIMIT,
                                  VTREE_FRAGMENT_TIME_LIMIT,
                                  VTREE_OP_TIME_LIMIT,
                                  VTREE_APPLY_TIME_LIMIT,
                                  0,0,0,0,0,0,0,0,0,0,
                                  VTREE_OP_SIZE_LIMIT,
                                  0,VTREE_OP_MEMORY_LIMIT,
                                  0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                  ' ',
                                  INITIAL_CONVERGENCE_THRESHOLD,
                                  CARTESIAN_PRODUCT_LIMIT};
  manager->vtree_ops = vtree_ops;
  
  //automatic garbage collection and search
  manager->auto_local_gc_and_search_on            = 0;
  manager->auto_gc_and_search_on                  = 0;
  manager->auto_vtree_search_on                   = 0;
  manager->auto_apply_vtree                       = NULL;
  manager->auto_apply_outside_live_size           = 0;
  manager->auto_apply_outside_live_count          = 0;
  manager->auto_apply_outside_dead_count          = 0;
  manager->auto_search_invocation_count           = 0;
  manager->auto_search_invocation_count_global    = 0;
  manager->auto_search_invocation_count_local     = 0;
  manager->auto_search_invocation_count_recursive = 0;
  manager->auto_search_invocation_count_aborted_apply     = 0;
  manager->auto_search_invocation_count_aborted_operation = 0;
  manager->auto_search_invocation_count_aborted_fragment  = 0;
  manager->auto_search_invocation_count_aborted_search    = 0;
  manager->auto_search_iteration_count            = 0;
  manager->auto_gc_invocation_count               = 0;
  manager->auto_search_reduction_sum              = 0;
  manager->vtree_search_function                  = NULL;
  
  //vtree fragments
  manager->max_fragment_shadow_count           = 0;
  manager->max_fragment_shadow_byte_count      = 0;
  manager->fragment_count                      = 0;
  manager->completed_fragment_count            = 0;
  manager->backward_completed_fragment_count   = 0;
  manager->successful_fragment_count           = 0;
  manager->backward_successful_fragment_count  = 0;
  manager->successful_completed_fragment_count = 0;
  
  //terminal sdds
  setup_terminal_sdds(manager); //must be done after setting properties
  
  last_constructed_manager = manager;
  declare_interrupt_signal(); //catch Ctrl+\ interrupts

  return manager;
}

//constructs a new manager using only a variable count
//uses a balanced vtree by default
//turns on auto mode
SddManager* sdd_manager_create(SddLiteral var_count, int auto_gc_and_minimize) {
  CHECK_ERROR(var_count<1,ERR_MSG_ONE_VAR,"sdd_manager_create");
  Vtree* vtree        = sdd_vtree_new(var_count,"balanced");
  SddManager* manager = sdd_manager_new(vtree);
  if(auto_gc_and_minimize) sdd_manager_auto_gc_and_minimize_on(manager); //auto mode
  sdd_vtree_free(vtree); //no longer needed (manager makes a copy)
  return manager;
}

/****************************************************************************************
 * initializing sdd manager
 ****************************************************************************************/

//construct and index true and false for manager
void setup_true_false_sdds(SddManager* manager) {
  //construct
  //CRITICAL to construct false first and true immediately next so that 
  //(id of false) = (id of true)-1
  //partitions.c depends crtitically on this
  SddNode* false_sdd = construct_false_sdd_node(manager);
  SddNode* true_sdd  = construct_true_sdd_node(manager);
  assert(true_sdd->id==1+false_sdd->id);
  //index
  manager->true_sdd  = true_sdd;
  manager->false_sdd = false_sdd;
  //cache negations
  true_sdd->negation = false_sdd;
  false_sdd->negation = true_sdd;
}

//construct and index literal sdds
void setup_literal_sdds(Vtree* vtree, SddManager* manager) {
  FOR_each_leaf_vtree_node(v,vtree,{
    //construct
    SddLiteral var = v->var;
    SddNode* plit  = construct_literal_sdd_node(var,v,manager);
    SddNode* nlit  = construct_literal_sdd_node(-var,v,manager);
    //declare as sdd nodes of vtree
    v->nodes         = plit; //first
    plit->vtree_next = nlit; //second
    nlit->vtree_next = NULL;
    v->node_count    = 2;
    //index
    manager->literals[var]  = plit;
    manager->literals[-var] = nlit;
    //cache negations
    plit->negation = nlit;
    nlit->negation = plit;
    //index leaf vtree
    manager->leaf_vtrees[var] = v;
  });
}

void setup_terminal_sdds(SddManager* manager) {
  //constant sdds
  setup_true_false_sdds(manager);
  //literal sdds
  setup_literal_sdds(manager->vtree,manager);
}

/****************************************************************************************
 * freeing manager and associated structures
 ****************************************************************************************/

//free manager and associated structures
void sdd_manager_free(SddManager* manager) {
  assert(manager->stats.element_count==manager->gc_element_count+manager->sdd_size);
  assert(manager->start_compression_stack==manager->top_compression_stack);
  assert(manager->start_element_stack==manager->top_element_stack);
  assert(manager->apply_depth==0);
  assert(manager->limited_apply_depth==0);
  
  //true and false sdds
  free_sdd_node(manager->true_sdd,manager);
  free_sdd_node(manager->false_sdd,manager);
  
  //literal sdds
  FOR_each_leaf_vtree_node(v,manager->vtree,{
    free_sdd_node(v->nodes->vtree_next,manager); //negative literal (must be freed first)
    free_sdd_node(v->nodes,manager); //positive literal (must be freed second)
  })

  //unique nodes
  FOR_each_unique_node(n,manager,free_sdd_node(n,manager)); //free unique nodes
  free_hash(manager->unique_nodes); //free hash tables

  //node structures in gc lists
  for(int i=0; i<GC_BUCKETS_COUNT; i++) {
    SddNode* list = manager->gc_node_lists[i];
    FOR_each_linked_node(node,list,free_sdd_node(node,manager));
  }
  free(manager->gc_node_lists);
  
  //computation caches
  free(manager->conjoin_cache);
  free(manager->disjoin_cache);

  //vtree and its associated structures
  sdd_vtree_free(manager->vtree);

  //manager indices
  free(manager->literals - manager->var_count);
  free(manager->leaf_vtrees);

  //elements stacks
  free(manager->start_compression_stack);
  free(manager->start_meta_compression_stack);
  free(manager->start_cp_stack1);
  free(manager->start_cp_stack2);
  free(manager->start_cp_stack3);
  free(manager->start_element_stack);

  //node buffer
  free(manager->node_buffer);
 
  assert(manager->stats.element_count==0);

  //manager
  free(manager);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
