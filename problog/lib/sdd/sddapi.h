/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 1.1.1, January 31, 2014
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include <stdio.h>
#include <stdlib.h>

/****************************************************************************************
 * this file contains the api for the sdd library: 
 * --function prototypes 
 * --associated declarations and definitions
 *
 * it also has flags for turning on assertions for choosing between 64 vs 32 versions
 ****************************************************************************************/

//comment the next line to activate assertions
#define NDEBUG
#include <assert.h>

#ifndef SDDAPI_H_
#define SDDAPI_H_

//comment this line out for 32bit compilation
#define _64BIT

/****************************************************************************************
 * typedefs and printf controls based on 32/64 bit
 ****************************************************************************************/

#ifdef _64BIT 
//sdd types for 64 bits
typedef size_t SddSize; //number of nodes, sizes of hash tables, etc
typedef unsigned int SddNodeSize; //size of decomposition for sdd nodes
typedef unsigned int SddRefCount; //refcount
typedef unsigned long long SddModelCount; //model counts
typedef double SddWmc; // weighted model count
typedef long SddLiteral; //literals of clauses
//control strings for 64 bits
#define PRIsS "zu"
#define PRInsS "u"
#define PRIrcS "u"
#define PRImcS "llu"
#define PRIlitS "ld"
#else
//sdd types for 32 bits
typedef unsigned int SddSize; //number of nodes, sizes of hash tables, etc
typedef unsigned int SddNodeSize; //size of decomposition for sdd nodes
typedef unsigned int SddRefCount; //refcount
typedef unsigned long long SddModelCount; //model counts
typedef double SddWmc; // weighted model count
typedef int SddLiteral; //literals of clauses
//control strings for 32 bits
#define PRIsS "u"
#define PRInsS "u"
#define PRIrcS "u"
#define PRImcS "llu"
#define PRIlitS "d"
#endif

typedef SddSize SddID;

typedef unsigned short BoolOp; //holds one of two values defined next
#define CONJOIN 0
#define DISJOIN 1

/****************************************************************************************
 * struct and function definitions
 ****************************************************************************************/

typedef struct vtree_t Vtree;
typedef struct sdd_node_t SddNode;
typedef struct sdd_manager_t SddManager;
typedef struct wmc_manager_t WmcManager;

typedef struct vtree_t* SddVtreeSearchFunc(struct vtree_t*, struct sdd_manager_t*);

/****************************************************************************************
 * function prototypes
 ****************************************************************************************/

// SDD MANAGER FUNCTIONS
SddManager* sdd_manager_new(Vtree* vtree);
SddManager* sdd_manager_create(SddLiteral var_count, int auto_gc_and_minimize);
void sdd_manager_free(SddManager* manager);
void sdd_manager_print(SddManager* manager);
void sdd_manager_auto_gc_and_minimize_on(SddManager* manager);
void sdd_manager_auto_gc_and_minimize_off(SddManager* manager);
void sdd_manager_set_minimize_function(SddVtreeSearchFunc func, SddManager* manager);
void sdd_manager_unset_minimize_function(SddManager* manager);
void* sdd_manager_options(SddManager* manager);
void sdd_manager_set_options(void* options, SddManager* manager);
int sdd_manager_is_var_used(SddLiteral var, SddManager* manager);
Vtree* sdd_manager_vtree_of_var(const SddLiteral var, const SddManager* manager);
Vtree* sdd_manager_lca_of_literals(int count, SddLiteral* literals, SddManager* manager);
SddLiteral sdd_manager_var_count(SddManager* manager);
void sdd_manager_var_order(SddLiteral* var_order, SddManager *manager);
void sdd_manager_add_var_before_first(SddManager* manager);
void sdd_manager_add_var_after_last(SddManager* manager);
void sdd_manager_add_var_before(SddLiteral target_var, SddManager* manager);
void sdd_manager_add_var_after(SddLiteral target_var, SddManager* manager);

// TERMINAL SDDS
SddNode* sdd_manager_true(const SddManager* manager);
SddNode* sdd_manager_false(const SddManager* manager);
SddNode* sdd_manager_literal(const SddLiteral literal, SddManager* manager);

// SDD QUERIES AND TRANSFORMATIONS
SddNode* sdd_apply(SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager);
SddNode* sdd_apply_in_vtree(SddNode* node1, SddNode* node2, BoolOp op, Vtree* vtree, SddManager* manager);
SddNode* sdd_conjoin(SddNode* node1, SddNode* node2, SddManager* manager);
SddNode* sdd_disjoin(SddNode* node1, SddNode* node2, SddManager* manager);
SddNode* sdd_negate(SddNode* node, SddManager* manager);
SddNode* sdd_condition(SddLiteral lit, SddNode* node, SddManager* manager);
SddNode* sdd_exists(SddLiteral var, SddNode* node, SddManager* manager);
SddNode* sdd_forall(SddLiteral var, SddNode* node, SddManager* manager);
SddNode* sdd_minimize_cardinality(SddNode* node, SddManager* manager);
SddLiteral sdd_minimum_cardinality(SddNode* node);
SddModelCount sdd_model_count(SddNode* node, SddManager* manager);

// SDD NAVIGATION
int sdd_node_is_true(SddNode* node);
int sdd_node_is_false(SddNode* node);
int sdd_node_is_literal(SddNode* node);
int sdd_node_is_decision(SddNode* node);
SddNodeSize sdd_node_size(SddNode* node);
SddLiteral sdd_node_literal(SddNode* node);
SddNode** sdd_node_elements(SddNode* node);
void sdd_node_set_bit(int bit, SddNode* node);
int sdd_node_bit(SddNode* node);

// SDD FUNCTIONS
SddSize sdd_id(SddNode* node);
int sdd_garbage_collected(SddNode* node, SddSize id);
Vtree* sdd_vtree_of(SddNode* node);
SddNode* sdd_copy(SddNode* node, SddManager* dest_manager);
SddNode* sdd_rename_variables(SddNode* node, SddLiteral* variable_map, SddManager* manager);
int* sdd_variables(SddNode* node, SddManager* manager);

// SDD FILE I/O
SddNode* sdd_read(const char* filename, SddManager* manager);
void sdd_save(const char* fname, SddNode *node);
void sdd_save_as_dot(const char* fname, SddNode *node);
void sdd_shared_save_as_dot(const char* fname, SddManager* manager);

// SDD SIZE AND NODE COUNT
//SDD
SddSize sdd_count(SddNode* node);
SddSize sdd_size(SddNode* node);
SddSize sdd_shared_size(SddNode** nodes, SddSize count);
//SDD OF MANAGER
SddSize sdd_manager_size(const SddManager* manager);
SddSize sdd_manager_live_size(const SddManager* manager);
SddSize sdd_manager_dead_size(const SddManager* manager);
SddSize sdd_manager_count(const SddManager* manager);
SddSize sdd_manager_live_count(const SddManager* manager);
SddSize sdd_manager_dead_count(const SddManager* manager);
//SDD OF VTREE
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

// CREATING VTREES
Vtree* sdd_vtree_new(SddLiteral var_count, const char* type);
Vtree* sdd_vtree_new_with_var_order(SddLiteral var_count, SddLiteral* var_order, const char* type);
void sdd_vtree_free(Vtree* vtree);

// VTREE FILE I/O
void sdd_vtree_save(const char* fname, Vtree* vtree);
Vtree* sdd_vtree_read(const char* filename);
void sdd_vtree_save_as_dot(const char* fname, Vtree* vtree);

// SDD MANAGER VTREE
Vtree* sdd_manager_vtree(const SddManager* manager);
Vtree* sdd_manager_vtree_copy(const SddManager* manager);

// VTREE NAVIGATION
Vtree* sdd_vtree_left(const Vtree* vtree);
Vtree* sdd_vtree_right(const Vtree* vtree);
Vtree* sdd_vtree_parent(const Vtree* vtree);

//VTREE FUNCTIONS
int sdd_vtree_is_leaf(const Vtree* vtree);
int sdd_vtree_is_sub(const Vtree* vtree1, const Vtree* vtree2);
Vtree* sdd_vtree_lca(Vtree* vtree1, Vtree* vtree2, Vtree* root);
SddLiteral sdd_vtree_var_count(const Vtree* vtree);
SddLiteral sdd_vtree_var(const Vtree* vtree);
SddLiteral sdd_vtree_position(const Vtree* vtree);
Vtree** sdd_vtree_location(Vtree* vtree, SddManager* manager);

// VTREE/SDD EDIT OPERATIONS
int sdd_vtree_rotate_left(Vtree* vtree, SddManager* manager, SddSize time_limit, float size_limit);
int sdd_vtree_rotate_right(Vtree* vtree, SddManager* manager, SddSize time_limit, float size_limit, int cartesian_product_limit);
int sdd_vtree_swap(Vtree* vtree, SddManager* manager, SddSize time_limit, float size_limit, int cartesian_product_limit);

// LIMITS FOR VTREE/SDD EDIT OPERATIONS
void sdd_manager_set_size_limit_context(Vtree* vtree, SddManager* manager);
void sdd_manager_update_size_limit_context(SddManager* manager);

// VTREE STATE
int sdd_vtree_bit(const Vtree* vtree);
void sdd_vtree_set_bit(int bit, Vtree* vtree);
void* sdd_vtree_data(Vtree* vtree);
void sdd_vtree_set_data(void* data, Vtree* vtree);
void* sdd_vtree_search_state(const Vtree* vtree);
void* sdd_vtree_set_search_state(void* search_state, const Vtree* vtree);

// GARBAGE COLLECTION
SddRefCount sdd_ref_count(SddNode* node);
SddNode* sdd_ref(SddNode* node, SddManager* manager);
SddNode* sdd_deref(SddNode* node, SddManager* manager);
void sdd_manager_garbage_collect(SddManager* manager);
void sdd_vtree_garbage_collect(Vtree* vtree, SddManager* manager);
int sdd_manager_garbage_collect_if(float dead_node_threshold, SddManager* manager);
int sdd_vtree_garbage_collect_if(float dead_node_threshold, Vtree* vtree, SddManager* manager);

// MINIMIZATION
void sdd_manager_minimize(SddManager* manager);
Vtree* sdd_vtree_minimize(Vtree* vtree, SddManager* manager);
void sdd_manager_set_lr_time_limit(SddSize time_limit, SddManager* manager);
void sdd_manager_set_rr_time_limit(SddSize time_limit, SddManager* manager);
void sdd_manager_set_sw_time_limit(SddSize time_limit, SddManager* manager);
void sdd_manager_set_lr_size_limit(float size_limit, SddManager* manager);
void sdd_manager_set_rr_size_limit(float size_limit, SddManager* manager);
void sdd_manager_set_sw_size_limit(float size_limit, SddManager* manager);
void sdd_manager_set_rr_cartesian_product_limit(int cartesian_product_limit, SddManager* manager);
void sdd_manager_set_sw_cartesian_product_limit(int cartesian_product_limit, SddManager* manager);
void sdd_manager_set_convergence_threshold(float threshold, SddManager* manager);

// WMC
WmcManager* wmc_manager_new(SddNode* node, int log_mode, SddManager* manager);
void wmc_manager_free(WmcManager* wmc_manager);
void wmc_set_literal_weight(const SddLiteral literal, const SddWmc weight, WmcManager* wmc_manager);
SddWmc wmc_propagate(WmcManager* wmc_manager);
SddWmc wmc_zero_weight(WmcManager* wmc_manager);
SddWmc wmc_one_weight(WmcManager* wmc_manager);
SddWmc wmc_literal_weight(const SddLiteral literal, const WmcManager* wmc_manager);
SddWmc wmc_literal_derivative(const SddLiteral literal, const WmcManager* wmc_manager);
SddWmc wmc_literal_pr(const SddLiteral literal, const WmcManager* wmc_manager);

#endif // SDDAPI_H_

/****************************************************************************************
 * end
 ****************************************************************************************/
