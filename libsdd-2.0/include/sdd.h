/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <execinfo.h>
#include "parameters.h"
#include "macros.h"
#include "iterators.h"
#include "stacks.h"

#ifndef SDD_H_
#define SDD_H_

/****************************************************************************************
 * typedefs and printf controls
 ****************************************************************************************/

//sdd types
typedef size_t SddSize; //number of nodes, sizes of hash tables, etc
typedef unsigned int SddNodeSize; //size of decomposition for sdd nodes
typedef unsigned int SddRefCount; //refcount
typedef unsigned long long SddModelCount; //model count
typedef double SddWmc; // weighted model count
typedef long SddLiteral; //literals of clauses

//control strings
#define PRIsS "zu"
#define PRInsS "u"
#define PRIrcS "u"
#define PRImcS "llu"
#define PRIwmcS "f"
#define PRIlitS "ld"
 
/****************************************************************************************
 * Enumerated Types
 ****************************************************************************************/

typedef char SddNodeType; //holds one of two values defined next
#define FALSE 0
#define TRUE 1
#define LITERAL 2
#define DECOMPOSITION 3

typedef unsigned short BoolOp; //holds one of two values defined next
#define CONJOIN 0
#define DISJOIN 1


/****************************************************************************************
 * vtree
 ****************************************************************************************/

// local search state, stored at each vtree node
typedef struct {
  struct vtree_t* previous_left;
  struct vtree_t* previous_right;
  SddSize previous_size;  
  SddSize previous_count;
  unsigned fold:1;
  unsigned virtually_empty:1;
} VtreeSearchState;

//vtree is a complete binary tree
typedef struct vtree_t { 
  struct vtree_t* parent; //parent
  struct vtree_t* left; //left child
  struct vtree_t* right; //right child

  //vtree nodes are maintained as a linked list
  struct vtree_t* next; //next node in in-order
  struct vtree_t* prev; //previous node in in-order
  struct vtree_t* first; //first node in in-order (which is part of this vtree)
  struct vtree_t* last; //last node in in-order (which is part of this vtree)
  
  //position of vtree node in the vtree inorder
  //position may CHANGE, e.g., due to swapping or adding/removing/moving variables, 
  //but is invariant to rotations
  SddLiteral position; //start from 0
  
  SddLiteral var_count; //number of variables in vtree
  SddSize sdd_size; //sum of sizes for all sdd nodes normalized for this vnode
  SddSize dead_sdd_size; //sum of sizes for all dead sdd nodes normalized for this vnode
  SddSize node_count; //number of sdd nodes normalized for vtree
  SddSize dead_node_count; //number of sdd nodes normalized for vtree with ref_count==0
  
  SddLiteral var; //variable associated with vtree (for leaf vtrees only)
  
  struct sdd_node_t* nodes; //linked list of nodes normalized for vtree (linked using ->vtree_next)
  //only two sdd nodes for leaf vtrees: first is positive literal, second is negative literal
    
  //used to associate secondary data with vtree structures by user
  void* user_data;
  void* user_search_state;
  
  //vtree search
  SddSize auto_last_search_live_size;
  VtreeSearchState* search_state; //for library version of vtree search
  
  unsigned some_X_constrained_vars:1;
  unsigned all_vars_in_sdd:1;
  unsigned no_var_in_sdd:1;
  unsigned bit:1;
  unsigned user_bit:1; //for user convenience
} Vtree;

/****************************************************************************************
 * FNF: Flat Normal Form (CNF, DNF)
****************************************************************************************/

typedef struct {
  SddSize id;
  SddLiteral literal_count;
  SddLiteral* literals; 
  BoolOp op; //DISJOIN (clause) or CONJOIN (term)
  Vtree* vtree;
  unsigned bit:1;
} LitSet;

typedef struct {
  SddLiteral var_count; // number of variables
  SddSize litset_count; // number of literal sets
  LitSet* litsets;  // array of literal sets
  BoolOp op; //CONJOIN (CNF) or DISJOIN (DNF)
} Fnf;

typedef Fnf Cnf;
typedef Fnf Dnf;

/****************************************************************************************
 * SDD nodes
 ****************************************************************************************/

//size should not be used to iterate over elements: use macros for iteration 
//order of fields below is important for the sizeof(SddNode)
typedef struct sdd_node_t {
  //put small types next to each other to reduce space used by structure
  //these are also the more used fields (more efficient if put first?)
  SddNodeType type;
  char shadow_type;
  SddNodeSize size; //number of elements for decomposition nodes, 0 for terminal nodes
  SddNodeSize saved_size; //used for reversing node replacement
  SddRefCount ref_count; //number of parents elements that have non-zero ref_count
  SddRefCount parent_count; //number of parents for node in the SDD DAG
  
  union {
	struct sdd_element_t* elements; // for decompositions
	SddLiteral literal;             // for literal terminals
  } alpha;
  struct sdd_element_t* saved_elements; //used for reversing node replacement

  struct sdd_node_t* next;  //linking into collision list of hash table  
  struct sdd_node_t** prev; //linking into collision list of hash table
  struct sdd_node_t* vtree_next; //linking into list of nodes normalized for same vtree
  struct sdd_node_t** vtree_prev; //linking into list of nodes normalized for same vtree
  struct sdd_node_t* negation; //caches negation of node when it exists
  Vtree* vtree; //vtree for which node is normalize for
  
  SddSize id; //unique id for each node
  SddSize index; //used mainly by graph traversal algorithms to cache results 
  
  struct sdd_node_t* multiply_sub; //used by multiply-decompositions
  struct sdd_node_t* map; //used for caching node transformations (exists, condition, rename vars,...)
                          //used also as a next field for distributing sdd nodes over vtree nodes
  struct sdd_node_shadow_t* shadow; //used by fragments
      
  unsigned bit:1; //used for navigating sdd graphs (should be kept 0 by default)
  unsigned cit:1; //used for navigating sdd graphs (should be kept 0 by default)
  unsigned dit:1; //used for multiplying decompositions
  unsigned git:1; //used for garbage collection
  unsigned in_unique_table:1; //used for maintaining counts and sizes
  unsigned replaced:1; //used for marking rotated or swapped nodes
  unsigned user_bit:1; //for user convenience
} SddNode;

/****************************************************************************************
 * SDD elements
 ****************************************************************************************/
 
typedef struct sdd_element_t {
  SddNode* prime;
  SddNode* sub;
} SddElement;

/****************************************************************************************
 * SDD Computed
 ****************************************************************************************/

typedef struct sdd_computed_t {
  struct sdd_node_t* result;
  SddSize id; //for result
  SddSize id1; //for argument1
  SddSize id2; //for argument2
} SddComputed;

/****************************************************************************************
 * hash tables for nodes
 ****************************************************************************************/

typedef struct sdd_hash_t {
  int qsize; //qualitative size of hash table
  SddSize size; //number of collision lists  
  SddSize count; //number of entries in hash table
  SddSize lookup_count; //number of lookup requests
  SddSize hit_count; //number of hists (lookups that succeeded)
  SddSize increase_size_count; //number of times size of hash table increased
  SddSize decrease_size_count; //number of times size of hash table decreased
  SddSize resize_age; //age since last resize
  SddSize lookup_cost; //total cost of lookups
  SddNode** clists; //array of collision lists
  //members of a collision list are linked through ->next and ->prev
} SddHash;


/****************************************************************************************
 * shadows
 ****************************************************************************************/
 
typedef struct sdd_element_shadow_t {
  struct sdd_node_shadow_t* prime;
  struct sdd_node_shadow_t* sub;
} ElmShadow;

typedef struct sdd_node_shadow_t {  
  union {
	ElmShadow* elements;
	SddNode* node;
  } alpha;
  SddNode* cache;
  SddSize cache_id;
  Vtree* vtree;
  SddNodeSize size;
  SddRefCount ref_count;
  unsigned bit:1; //used for navigating shadows
  unsigned reuse:1; //used for shadows that require reusing sdd node
} NodeShadow;

typedef struct sdd_shadows_t {
  struct sdd_manager_t* manager;
  SddSize root_count;
  NodeShadow** root_shadows; //roots of the shadow DAG
  SddSize shadow_count;
  SddSize shadow_byte_count;
  unsigned bit:1; //current value of shadow bit
} SddShadows;

/****************************************************************************************
 * vtree fragment
 ****************************************************************************************/
 
typedef struct vtree_fragment_t {

  //state
  int state; //0...11
  char mode; //'i', 'n', or 'g'
  Vtree* cur_root;
  Vtree* cur_child;
  
  //fragment identity
  struct sdd_manager_t* manager;
  char type; //'l' or 'r'
  Vtree* root;
  Vtree* child;
  char* moves;

  //see definitions of IR, IC and Ic nodes in vtree_fragments/construction.c
  SddSize IR_IC_count; 
  SddSize Ic_count;
  SddNode** IR_IC_nodes;
  SddNode** Ic_nodes;
  
  SddShadows* shadows;
} VtreeFragment;


/****************************************************************************************
 * manager
 ****************************************************************************************/

typedef struct sdd_manager_stats_t {
  clock_t auto_search_time;
  clock_t auto_max_search_time;
  SddSize apply_count;
  SddSize apply_count_top;
  //largest decomposition
  SddNodeSize max_decomposition_size;
  SddNodeSize max_uncompressed_decomposition_size;
  //elements
  SddSize element_count; //number of elements in memory
  SddSize max_element_count; //maximum number of elements that every existed in memory
} SddManagerStats;

//options and stats for vtree operations (rotate and swap)
typedef struct sdd_manager_vtree_ops_t {
  //time limits
  clock_t search_time_limit;
  clock_t fragment_time_limit;
  clock_t op_time_limit;
  clock_t apply_time_limit;
  //time stamps
  clock_t search_time_stamp;
  clock_t fragment_time_stamp;
  clock_t op_time_stamp;
  clock_t apply_time_stamp;
  //aborted
  int search_aborted;
  int fragment_aborted;
  int op_aborted;
  int apply_aborted;
  //size limits
  SddSize op_size_stamp;
  SddSize outside_size;
  float op_size_limit;
  //memory limits
  float op_memory_stamp;
  float op_memory_limit;
  //counts for moves
  SddSize lr_count;
  SddSize rr_count;
  SddSize sw_count;
  SddSize failed_lr_count_time; 
  SddSize failed_rr_count_time;
  SddSize failed_sw_count_time;  
  SddSize failed_lr_count_size; 
  SddSize failed_rr_count_size;
  SddSize failed_sw_count_size;
  SddSize failed_count_cp;
  SddSize failed_lr_count_memory;
  SddSize failed_rr_count_memory;
  SddSize failed_sw_count_memory;
  //for printing status info (^\)
  SddLiteral current_vtree;
  char current_op;
  float convergence_threshold;
  SddSize cartesian_product_limit;
} SddManagerVtreeOps;

typedef struct sdd_manager_t {

  SddSize id_counter; //used to generate new ids for nodes and elements
  SddLiteral var_count; //number of variables in manager (vtree leaves)

  //counts
  SddSize node_count; //dead + live
  SddSize dead_node_count; 
  SddSize computed_count; 
  
  //size of sdds
  SddSize sdd_size; //dead + live
  SddSize dead_sdd_size; //dead only
	
  //counts of free structures (available to be reclaimed)
  SddSize gc_node_count; //number of free nodes
  SddSize gc_element_count; //number of free elements

  //linked lists of free structures (linked through next)
  SddNode** gc_node_lists; //buckets of linked lists
 
  struct vtree_t* vtree;
  SddNode* true_sdd;
  SddNode* false_sdd;

  //indexing literal sdds and leaf vtrees
  struct sdd_node_t** literals; //array of literals
  struct vtree_t** leaf_vtrees; //array of leaf vtrees

  //unique_nodes
  SddHash* unique_nodes;
  
  //computation caches
  SddSize computed_cache_lookup_count;
  SddSize computed_cache_hit_count;
  SddComputed* conjoin_cache;
  SddComputed* disjoin_cache;
  
  //apply
  SddLiteral apply_depth; //depth of apply call (1 means top-level apply)
  SddLiteral limited_apply_depth; //depth of apply call with first limited ancestor having depth 1
  
  //stacks
  
  //prime_sub stack for unique_node
  SddElement* top_compression_stack; 
  SddElement* start_compression_stack; 
  SddSize capacity_compression_stack;
  
  //element stacks for cartesian products
  SddElement* top_cp_stack1; 
  SddElement* start_cp_stack1; 
  SddSize capacity_cp_stack1;
  
  SddElement* top_cp_stack2; 
  SddElement* start_cp_stack2; 
  SddSize capacity_cp_stack2;
  
  SddElement* top_cp_stack3; 
  SddElement* start_cp_stack3; 
  SddSize capacity_cp_stack3;
  
  //stack for nesting compression stacks
  SddSize* top_meta_compression_stack; //stack for storing pointers to prime_sub stack
  SddSize* start_meta_compression_stack;
  SddSize capacity_meta_compression_stack;
  
  //stack for compressed elements
  SddElement* top_element_stack;
  SddElement* start_element_stack;
  SddNodeSize capacity_element_stack;
  
  //buffer for sorting nodes
  SddNode** node_buffer;
  SddSize node_buffer_size;

  //general options for manager 
  void* options;
    
  //manager stats
  SddManagerStats stats; 
    
  //options and stats for vtree operations
  SddManagerVtreeOps vtree_ops; 
  
  //automatic garbage collection and search
  int auto_local_gc_and_search_on; //gc and search will be conducted locally only
  int auto_gc_and_search_on; //automatic gc and search are possible
  int auto_vtree_search_on; //whether we are currently searching for a vtree
  Vtree* auto_apply_vtree; //vtree of top-level apply call
  SddSize auto_apply_outside_live_size;  //live size outside vtree of top-level apply call
  SddSize auto_apply_outside_live_count; //live count outside vtree of top-level apply call
  SddSize auto_apply_outside_dead_count; //dead count outside vtree of top-level apply call
  int auto_gc_invocation_count; //number of auto gc's performed
  int auto_search_invocation_count; //number of auto searches performed
  int auto_search_invocation_count_global; //number of global auto searches performed
  int auto_search_invocation_count_local; //number of local auto searches performed
  int auto_search_invocation_count_recursive; //number of recursive auto searches performed
  int auto_search_invocation_count_aborted_apply; //number of aborted auto searches (apply limits)
  int auto_search_invocation_count_aborted_operation; //number of aborted auto searches (operation limits)
  int auto_search_invocation_count_aborted_fragment; //number of aborted auto searches (fragment limits)
  int auto_search_invocation_count_aborted_search; //number of aborted auto searches (search limits)
  int auto_search_iteration_count; //number of iterations (passes) per search
  float auto_search_reduction_sum; //sum of percentage size reduction over all searches
  
  //fragments
  SddSize max_fragment_shadow_count; //maximum number of shadows constructed by any fragment
  SddSize max_fragment_shadow_byte_count; //maximum number of shadows bytes constructed by any fragment
  SddSize fragment_count; //number of fragments searched
  SddSize completed_fragment_count; //number of fragments searched completely (all 12 states)
  SddSize backward_completed_fragment_count; //number of fragments searched completely using backward search (all 12 states)
  SddSize successful_fragment_count; //number of searched fragments leading to a better state
  SddSize backward_successful_fragment_count; //number of backward searched fragments leading to a better state
  SddSize successful_completed_fragment_count; //number of fragments searched completely and leading to a better state
  
  //the vtree search function to be used in auto vtree search
  void* vtree_search_function;
  
} SddManager;


/****************************************************************************************
 * function types
 ****************************************************************************************/

//vtree search function
typedef struct vtree_t* SddVtreeSearchFunc(struct vtree_t*, struct sdd_manager_t*);

/****************************************************************************************
 * WmcManager
 *
 * An environment for computing weighted model counts
 *
 ****************************************************************************************/
 
typedef struct wmc_manager_t {
  int log_mode;
  SddNode* node; //root of sdd
  SddSize node_count; //number of nodes in sdd
  SddNode** nodes; //sorted so children before parents
  SddSize* node_indices; //indices of nodes in topologically sorted array
  SddWmc* node_wmcs;
  SddWmc* node_derivatives;
  SddWmc* literal_weights;
  SddWmc* literal_derivatives;
  SddWmc* used_true_wmcs;
  SddWmc* unused_true_wmcs;
  SddWmc wmc;
  SddManager* sdd_manager;
} WmcManager;

/****************************************************************************************
 * SatManager
 *
 * An environment for deciding satisfiability
 *
 ****************************************************************************************/
 
typedef struct sat_manager_t {
  SddLiteral var_count; //number of variables
  SddSize node_count; //number of nodes in sdd
  SddNode* node; //root of sdd
  SddNode** nodes; //sorted so children before parents
  SddSize* node_indices; //indices of nodes in topologically sorted array
  int* node_sats; //holds 0 or 1
  char* var_values; //holds 'T', 'F', '?'
  int sat; //0 or 1
  int needs_update; //0 or 1
} SatManager;

/****************************************************************************************
 * function prototypes
 ****************************************************************************************/

// util.c
void press(char* str);
void header_strtok(char* buffer, const char* expected_token);
void test_and_exit(int test, const char* message);
void unexpected_node_type_error(char node_type);
int int_strtok();
char char_strtok();
char* read_file(const char* filename);
char* filter_comments(const char* buffer);
char* literal_to_label(SddLiteral lit);

//verify.c
int verify_vtree_properties(const Vtree* vtree);
int verify_counts_and_sizes(const SddManager* manager);
int verify_normalization(const SddManager* manager);
int verify_negations(const SddManager* manager);
int verify_gc(const Vtree* vtree, SddManager* manager);
int verify_X_constrained(const Vtree* vtree);

//
//basic
//

//computed.c
SddNode* lookup_computation(SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager);
void cache_computation(SddNode* node1, SddNode* node2, SddNode* node, BoolOp op, SddManager* manager);

//gc.c
void sdd_vtree_garbage_collect(Vtree* vtree, SddManager* manager);

//hash.c
float hit_rate(SddHash* hash);
float ave_lookup_cost(SddHash* hash);
float saturation(SddHash* hash);

//reference.c
SddRefCount sdd_ref_count(SddNode* node);
SddNode* sdd_ref(SddNode* node, SddManager* manager);
SddNode* sdd_deref(SddNode* node, SddManager* manager);

//sort.c
void sort_linked_nodes(SddSize count, SddNode** list, SddManager* manager);
void sort_uncompressed_elements(SddSize size, SddElement* elements);
void sort_compressed_elements(SddNodeSize size, SddElement* elements);
int elements_sorted_and_compressed(SddNodeSize size, SddElement* elements);

//
//manager
//

//manager.c
SddManager* sdd_manager_new(Vtree* vtree);
void sdd_manager_free(SddManager* manager);
void sdd_manager_print(SddManager* manager);

//interface.c
void declare_interrupt_signal();
void sdd_manager_auto_gc_and_minimize_on(SddManager* manager);
void sdd_manager_auto_gc_and_minimize_off(SddManager* manager);
int sdd_manager_is_auto_gc_and_minimize_on(SddManager* manager);
SddLiteral sdd_manager_var_count(SddManager* manager);
SddSize sdd_manager_size(const SddManager* manager);
SddSize sdd_manager_live_size(const SddManager* manager);
SddSize sdd_manager_dead_size(const SddManager* manager);
SddSize sdd_manager_count(const SddManager* manager);
SddSize sdd_manager_live_count(const SddManager* manager);
SddSize sdd_manager_dead_count(const SddManager* manager);
SddNode* sdd_manager_true(const SddManager* manager);
SddNode* sdd_manager_false(const SddManager* manager);
SddNode* sdd_manager_literal(const SddLiteral literal, const SddManager* manager);
Vtree* sdd_manager_vtree_copy(const SddManager* manager);
void sdd_manager_var_order(SddLiteral* var_order, SddManager* manager);
void sdd_manager_minimize(SddManager* manager);
void sdd_manager_minimize_limited(SddManager* manager);

//variables.c
int sdd_manager_is_var_used(SddLiteral var, SddManager* manager);
int* var_usage_map(SddManager* manager);
void sdd_manager_add_var_before_first(SddManager* manager);
void sdd_manager_add_var_after_last(SddManager* manager);
void add_var_before_top(SddManager* manager);
void add_var_after_top(SddManager* manager);
void sdd_manager_add_var_before(SddLiteral target_var, SddManager* manager);
void sdd_manager_add_var_after(SddLiteral target_var, SddManager* manager);
void add_var_before_lca(int count, SddLiteral* variables, SddManager* manager);
void add_var_after_lca(int count, SddLiteral* variables, SddManager* manager);
void move_var_before_first(SddLiteral var, SddManager* manager);
void move_var_after_last(SddLiteral var, SddManager* manager);
void move_var_before(SddLiteral var, SddLiteral target_var, SddManager* manager);
void move_var_after(SddLiteral var, SddLiteral target_var, SddManager* manager);
void remove_var_added_last(SddManager* manager);

//
//vtree_ops
//

//dissect.c
Vtree* left_linearize_vtree(Vtree* vtree, SddManager* manager);
Vtree* right_linearize_vtree(Vtree* vtree, SddManager* manager);
Vtree* balance_vtree(Vtree* vtree, SddManager* manager);

//move.c
void move_vtree_up_to(Vtree* vtree, Vtree** root_location, SddManager* manager);
void move_vtree_down(Vtree* vtree, SddManager* manager);

//rotate_left.c
int sdd_vtree_rotate_left(Vtree* vtree, SddManager* manager, int limited);
//rotate_right.c
int sdd_vtree_rotate_right(Vtree* x, SddManager* manager, int limited);
//swap.c
int sdd_vtree_swap(Vtree* v, SddManager* manager, int limited);

//fragment.c
VtreeFragment* vtree_fragment_new(Vtree* root, Vtree* child, SddManager* manager);
void vtree_fragment_free(VtreeFragment* fragment);
int vtree_fragment_is_initial(VtreeFragment* fragment);
Vtree* vtree_fragment_root(VtreeFragment* fragment);
int vtree_fragment_state(VtreeFragment* fragment);
int vtree_fragment_next(char direction, VtreeFragment* fragment, int limited);
Vtree* vtree_fragment_goto(int state, char direction, VtreeFragment* fragment);
Vtree* vtree_fragment_rewind(VtreeFragment* fragment);

//
//vtree search
//

//search.c
Vtree* sdd_vtree_minimize(Vtree* vtree, SddManager* manager);
Vtree* sdd_vtree_minimize_limited(Vtree* vtree, SddManager* manager);

//
//fnf
//

//compiler.c
SddNode* fnf_to_sdd(Fnf* fnf, SddManager* manager);

//fnf.c
int is_cnf(Fnf* fnf);
int is_dnf(Fnf* fnf);
void free_fnf(Fnf* fnf);
int sdd_implies_cnf(SddNode* node, Cnf* cnf, SddManager* manager);
int dnf_implies_sdd(SddNode* node, Dnf* dnf, SddManager* manager);

//io.c
Cnf* sdd_cnf_read(const char* filename);
Dnf* sdd_dnf_read(const char* filename);
void print_cnf(FILE* file, const Cnf* cnf);
void print_dnf(FILE* file, const Dnf* dnf);

//vtree.c
void minimize_vtree_width(Fnf* fnf, Vtree** vtree_loc);


//
//sdd
//

//apply.c
SddNode* sdd_apply(SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager);
SddNode* sdd_negate(SddNode* node, SddManager* manager);

//bits.c
void sdd_clear_node_bits(SddNode* node);
SddNode** sdd_topological_sort(SddNode* node, SddSize* size);
SddSize sdd_count_multiple_parent_nodes(SddNode* node);
SddSize sdd_count_multiple_parent_nodes_to_leaf(SddNode* node, Vtree* leaf);

//cardinality.c
SddLiteral sdd_minimum_cardinality(SddNode* node);
SddNode* sdd_minimize_cardinality(SddNode* node, SddManager* manager);

//condition.c
SddNode* sdd_condition(SddLiteral lit, SddNode* node, SddManager* manager);

//copy.c
SddNode* sdd_copy(SddNode* node, SddManager* dest_manager);

//exists.c
SddNode* sdd_exists(SddLiteral var, SddNode* node, SddManager* manager);

//exists.c
SddNode* sdd_forall(SddLiteral var, SddNode* node, SddManager* manager);

//io.c
void sdd_save_as_dot(const char* fname, SddNode *node);
void save_shared_sdd_as_dot_vt(const char* fname, Vtree* vtree);
void sdd_shared_save_as_dot(const char* fname, SddManager* manager);
void save_sdd_vt(const char* fname, SddNode *node, Vtree* vtree);
void sdd_save(const char* fname, SddNode *node);
SddNode* sdd_read(const char* filename, SddManager* manager);

//model_count.c
SddModelCount sdd_model_count(SddNode* node, SddManager* manager);

//node_count.c
SddSize sdd_count(SddNode* node);
SddSize sdd_all_node_count(SddNode* node);
SddSize sdd_all_node_count_leave_bits_1(SddNode* node);

//rename_vars.c
SddNode* sdd_rename_variables(SddNode* node, SddLiteral* variable_map, SddManager* manager);

//size.c
SddSize sdd_size(SddNode* node);
SddSize sdd_shared_size(SddNode** nodes, SddSize count);

//essential_vars.c
int* sdd_variables(SddNode* node, SddManager* manager);
void set_sdd_variables(SddNode* node, SddManager* manager);

//weighted_model_count.c
WmcManager* wmc_manager_new(SddNode* node, int log_mode, SddManager* manager);
void wmc_manager_free(WmcManager* wmc_manager);
SddWmc wmc_propagate(WmcManager* wmc_manager);
SddWmc wmc_zero_weight(WmcManager* wmc_manager);
SddWmc wmc_one_weight(WmcManager* wmc_manager);
void wmc_set_literal_weight(const SddLiteral literal, const SddWmc weight, WmcManager* wmc_manager);
SddWmc literal_weight(const SddLiteral literal, const WmcManager* wmc_manager);
SddWmc wmc_literal_derivative(const SddLiteral literal, const WmcManager* wmc_manager);
SddWmc wmc_literal_pr(const SddLiteral literal, const WmcManager* wmc_manager);

//
//vtree
//

//io.c
Vtree* sdd_vtree_read(const char* filename);
void sdd_vtree_save(const char* fname, Vtree* vtree);
void sdd_vtree_save_as_dot(const char* fname, Vtree* vtree);

//compare.c
int sdd_vtree_is_sub(const Vtree* vtree1, const Vtree* vtree2);
Vtree* sdd_vtree_lca(Vtree* vtree1, Vtree* vtree2, Vtree* root);
Vtree* lca_of_compressed_elements(SddNodeSize size, SddElement* elements, SddManager* manager);

//moves.c
int is_left_rotatable(Vtree* x);
int is_right_rotatable(Vtree* x);
void rotate_vtree_left(Vtree* x, SddManager* manager);
void rotate_vtree_right(Vtree* x, SddManager* manager);
void swap_vtree_children(Vtree* vtree, SddManager* manager);

//static.c
Vtree* copy_vtree(Vtree* vtree);
Vtree* sdd_vtree_new(SddLiteral var_count, const char* type);
Vtree* sdd_vtree_new_with_var_order(SddLiteral var_count, SddLiteral* var_order, const char* type);
  
//vtree.c
void sdd_vtree_free(Vtree* vtree);
Vtree* sdd_manager_vtree_of_var(const SddLiteral var, const SddManager* manager);
Vtree** sdd_vtree_location(Vtree* vtree, SddManager* manager);
Vtree* sibling(Vtree* vtree);

//size.c
SddSize sdd_vtree_size(const Vtree* vtree);
SddSize sdd_vtree_live_size(const Vtree* vtree);
SddSize sdd_vtree_dead_size(const Vtree* vtree);
SddSize sdd_vtree_count(const Vtree* vtree);
SddSize sdd_vtree_live_count(const Vtree* vtree);
SddSize sdd_vtree_dead_count(const Vtree* vtree);
SddSize sdd_vtree_size_at(const Vtree* vtree);
SddSize sdd_vtree_live_size_at(const Vtree* vtree);
SddSize sdd_vtree_dead_size_at(const Vtree* vtree);
SddSize sdd_vtree_count_at(const Vtree* vtree);
SddSize sdd_vtree_live_count_at(const Vtree* vtree);
SddSize sdd_vtree_dead_count_at(const Vtree* vtree);
SddSize sdd_vtree_live_size_above(const Vtree* vtree);
SddSize sdd_vtree_dead_count_above(const Vtree* vtree);

//
//search
//
Vtree* sdd_vtree_left(const Vtree* vtree);
Vtree* sdd_vtree_right(const Vtree* vtree);
Vtree* sdd_vtree_parent(const Vtree* vtree);
void sdd_vtree_set_search_state(void* search_state, Vtree* vtree);
void* sdd_vtree_search_state(const Vtree* vtree);
Vtree* sdd_manager_vtree(const SddManager* manager);
SddLiteral sdd_vtree_var_count(const Vtree* vtree);

#endif // SDD_H_

/****************************************************************************************
 * end
 ****************************************************************************************/
