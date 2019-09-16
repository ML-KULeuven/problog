/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"
#include <signal.h>

//vtrees/vtree.c
void set_vtree_properties(Vtree* vtree);

/****************************************************************************************
 * (Ctrl+\) can be used to print the last constructed sdd manager
 *
 * this is used to see what's going on in case the manager is stuck while computing
 ****************************************************************************************/

//a signal handling function for user interuption (Ctrl+\)
void SIGQUIT_handler(int signum) {
  extern SddManager* last_constructed_manager;
  if(last_constructed_manager!=NULL) {
    printf("\n---");
    sdd_manager_print(last_constructed_manager);
    char c = last_constructed_manager->vtree_ops.current_op;
    if(c!=' ') {
      printf("\ncurrent vtree operation: %c\n",c);
    }
    printf("---\n");
  }
}

void declare_interrupt_signal() {
  signal(SIGQUIT,SIGQUIT_handler);
}


/****************************************************************************************
 * dynamic garbage collection and vtree search 
 ****************************************************************************************/

void sdd_manager_auto_gc_and_minimize_on(SddManager* manager) {
  manager->auto_gc_and_search_on       = 1;
}

void sdd_manager_auto_gc_and_minimize_off(SddManager* manager) {
  manager->auto_gc_and_search_on = 0;
}

int sdd_manager_is_auto_gc_and_minimize_on(SddManager* manager) {
  return manager->auto_gc_and_search_on;
}

void sdd_manager_set_minimize_function(SddVtreeSearchFunc f, SddManager* manager) {
  manager->vtree_search_function = f;
}

void sdd_manager_unset_minimize_function(SddManager* manager) {
  manager->vtree_search_function = NULL;
}

/****************************************************************************************
 * basic manager lookups 
 ****************************************************************************************/

//returns number of variables in a manager
SddLiteral sdd_manager_var_count(SddManager* manager) {
  return manager->var_count;
}

/****************************************************************************************
 * SDD navigation 
 ****************************************************************************************/

int sdd_node_is_true(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_is_true");
  return node->type==TRUE;
}

int sdd_node_is_false(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_is_false");
  return node->type==FALSE;
}

int sdd_node_is_literal(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_is_literal");
  return node->type==LITERAL;
}

int sdd_node_is_decision(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_is_decision");
  return node->type==DECOMPOSITION;
}
 
SddNodeSize sdd_node_size(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_node_size");
  return node->size;
}

SddLiteral sdd_node_literal(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_literal_of");
  CHECK_ERROR(node->type!=LITERAL,ERR_MSG_NODE_ITR,"sdd_literal_of");
  return LITERAL_OF(node);
}

SddNode** sdd_node_elements(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_node_elements");
  CHECK_ERROR(node->type!=DECOMPOSITION,ERR_MSG_NODE_ITR,"sdd_node_elements");
  return (SddNode**) ELEMENTS_OF(node);
}

void sdd_node_set_bit(int bit, SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_node_set_bit");
  node->user_bit = bit;
}

int sdd_node_bit(SddNode* node) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_node_bit");
  return node->user_bit;
}


SddSize test_n(SddNode* node) {
  if(sdd_node_bit(node)) return 0;
  else {
    sdd_node_set_bit(1,node);
    SddSize size = 0;
    if(sdd_node_is_decision(node)) {
      SddNodeSize s = sdd_node_size(node);
      SddNode** e   = sdd_node_elements(node);
      size += s;
      while(s--) {
        size += test_n(*e++);
        size += test_n(*e++);
      }
    }
    return size;
  }
}

/****************************************************************************************
 * size and count
 ****************************************************************************************/
 
//total size of all sdd nodes
SddSize sdd_manager_size(const SddManager* manager) {
  return manager->sdd_size;
}

//total size of live sdd nodes
SddSize sdd_manager_live_size(const SddManager* manager) {
  return manager->sdd_size-manager->dead_sdd_size;
}

//total size of dead sdd nodes
SddSize sdd_manager_dead_size(const SddManager* manager) {
  return manager->dead_sdd_size;
}

//total count of all sdd nodes
SddSize sdd_manager_count(const SddManager* manager) {
  return manager->node_count;
}

//total count of live sdd nodes
SddSize sdd_manager_live_count(const SddManager* manager) {
  return manager->node_count-manager->dead_node_count;
}

//total count of dead sdd nodes
SddSize sdd_manager_dead_count(const SddManager* manager) {
  return manager->dead_node_count;
}

/****************************************************************************************
 * terminal SDD nodes
 ****************************************************************************************/
 
//returns a true sdd that is normalized for the root of the manager's vtree
SddNode* sdd_manager_true(const SddManager* manager) {
  return manager->true_sdd;
}

//returns a false sdd that is normalized for the root of the manager's vtree
SddNode* sdd_manager_false(const SddManager* manager) {
  return manager->false_sdd;
}

//returns the sdd corresponding to a literal, normalized for corresponding leaf vtree
//literal is an integer <> 0
SddNode* sdd_manager_literal(const SddLiteral literal, const SddManager* manager) {
  return manager->literals[literal];
}

/****************************************************************************************
 * sdd functions
 ****************************************************************************************/

SddSize sdd_id(SddNode* node) {
  return node->id; 
}

//returns 1 if the node having id has been garbage collected
//the test GC_NODE(node) may not succeed even though the node may have been gc'd
//this happens when the node structure has been reused by another newly created node
//in this case, the test node->id!=id will succeed
int sdd_garbage_collected(SddNode* node, SddSize id) {
  return GC_NODE(node) || node->id!=id; 
}

Vtree* sdd_vtree_of(SddNode* node) {
  return node->vtree;
}

SddNode* sdd_conjoin(SddNode* node1, SddNode* node2, SddManager* manager) {
  return sdd_apply(node1,node2,CONJOIN,manager);
}

SddNode* sdd_disjoin(SddNode* node1, SddNode* node2, SddManager* manager) {
  return sdd_apply(node1,node2,DISJOIN,manager);
}

/****************************************************************************************
 * garbage collection
 ****************************************************************************************/

//runs a global garbage collection on sdd nodes
void sdd_manager_garbage_collect(SddManager* manager) {
  sdd_vtree_garbage_collect(manager->vtree,manager);
}

//runs a global garbage collection when the number of dead nodes exceeds a certain threshold
//this is more efficient than calling sdd_manager_garbage_collect_if on vtree of manager
int sdd_manager_garbage_collect_if(float dead_node_threshold, SddManager* manager) {
  SddSize dead_node_count  = sdd_manager_dead_count(manager); //more efficient
  SddSize total_node_count = sdd_manager_count(manager); //more efficient
  if(dead_node_count > total_node_count*dead_node_threshold) {
    sdd_manager_garbage_collect(manager);
    return 1;
  }
  else return 0;
}

//runs a local garbage collection when the number of dead nodes exceeds a certain threshold
int sdd_vtree_garbage_collect_if(float dead_node_threshold, Vtree* vtree, SddManager* manager) {
  SddSize dead_node_count  = sdd_vtree_dead_count(vtree);
  SddSize total_node_count = sdd_vtree_count(vtree);
  if(dead_node_count > total_node_count*dead_node_threshold) {
    sdd_vtree_garbage_collect(vtree,manager);
    return 1;
  }
  else return 0;
}

/****************************************************************************************
 * vtree search (sdd minimization)
 ****************************************************************************************/

void sdd_manager_minimize(SddManager* manager) {
  sdd_vtree_minimize(sdd_manager_vtree(manager),manager);
}

void sdd_manager_minimize_limited(SddManager* manager) {
  sdd_vtree_minimize_limited(sdd_manager_vtree(manager),manager);
}


// these are checked by exceeded_limits(), invoked by l_apply
void sdd_manager_set_vtree_search_time_limit(float time_limit, SddManager* manager) {
  manager->vtree_ops.search_time_limit = time_limit*CLOCKS_PER_SEC;
}
void sdd_manager_set_vtree_fragment_time_limit(float time_limit, SddManager* manager) {
  manager->vtree_ops.fragment_time_limit = time_limit*CLOCKS_PER_SEC;
}
void sdd_manager_set_vtree_operation_time_limit(float time_limit, SddManager* manager) {
  manager->vtree_ops.op_time_limit = time_limit*CLOCKS_PER_SEC;
}
void sdd_manager_set_vtree_apply_time_limit(float time_limit, SddManager* manager) {
  manager->vtree_ops.apply_time_limit = time_limit*CLOCKS_PER_SEC;
}
void sdd_manager_set_vtree_operation_memory_limit(float memory_limit, SddManager* manager) {
  manager->vtree_ops.op_memory_limit = memory_limit;
}

// this is checked by exceeded_size_limit(), invoked by vtree operations
void sdd_manager_set_vtree_operation_size_limit(float size_limit, SddManager* manager) {
  manager->vtree_ops.op_size_limit = size_limit;
}

void sdd_manager_set_vtree_search_convergence_threshold(float threshold, SddManager* manager) {
  manager->vtree_ops.convergence_threshold = threshold;
}

void sdd_manager_set_vtree_cartesian_product_limit(SddSize size_limit, SddManager* manager) {
  manager->vtree_ops.cartesian_product_limit = size_limit;
}

/****************************************************************************************
 * manager options
 ****************************************************************************************/

void* sdd_manager_options(SddManager* manager) {
  return manager->options;
}

void sdd_manager_set_options(void* options, SddManager* manager) {
  manager->options = options;
}

/****************************************************************************************
 * manager vtree and order
 ****************************************************************************************/
 
//returns a copy of the vtree for manager
Vtree* sdd_manager_vtree_copy(const SddManager* manager) {
  Vtree* vtree = copy_vtree(manager->vtree);
  set_vtree_properties(vtree);
  return vtree;
}

//returns vtree of manager
Vtree* sdd_manager_vtree(const SddManager* manager) {
  return manager->vtree;
}


//fills the given array with manager variables according to their vtree inorder
//the array length must be equal to the number of manager variables
void sdd_manager_var_order(SddLiteral* var_order, SddManager *manager) {
  void var_order_aux(SddLiteral** var_order_loc, Vtree* vtree);
  var_order_aux(&var_order,manager->vtree);
}

void var_order_aux(SddLiteral** var_order_loc, Vtree* vtree) {
  if(LEAF(vtree)) {
    **var_order_loc = vtree->var;
    (*var_order_loc)++;
  }
  else {
    var_order_aux(var_order_loc,vtree->left);
    var_order_aux(var_order_loc,vtree->right);
  }
}

/****************************************************************************************
 * vtree traversal
 ****************************************************************************************/

Vtree* sdd_vtree_left(const Vtree* vtree) {
  return vtree->left;
}

Vtree* sdd_vtree_right(const Vtree* vtree) {
  return vtree->right;
}

Vtree* sdd_vtree_parent(const Vtree* vtree) {
  return vtree->parent;
}

/****************************************************************************************
 * vtree properties
 ****************************************************************************************/

SddLiteral sdd_vtree_var_count(const Vtree* vtree) {
  return vtree->var_count;
}

SddLiteral sdd_vtree_var(const Vtree* vtree) {
  if(LEAF(vtree)) return vtree->var;
  else return 0;
}

SddLiteral sdd_vtree_position(const Vtree* vtree) {
  return vtree->position;
}

/****************************************************************************************
 * vtree state
 ****************************************************************************************/
 
void sdd_vtree_set_bit(int bit, Vtree* vtree) {
  vtree->user_bit = bit;
}

int sdd_vtree_bit(const Vtree* vtree) {
  return vtree->user_bit;
}

void sdd_vtree_set_data(void* data, Vtree* vtree) {
  vtree->user_data = data;
}

void* sdd_vtree_data(const Vtree* vtree) {
  return vtree->user_data;
}

void sdd_vtree_set_search_state(void* search_state, Vtree* vtree) {
  vtree->user_search_state = search_state;
}

void* sdd_vtree_search_state(const Vtree* vtree) {
  return vtree->user_search_state;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
