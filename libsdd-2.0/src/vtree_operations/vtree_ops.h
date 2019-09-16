/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

/****************************************************************************************
 * used by rotate and swap
 ****************************************************************************************/

//basic/gc.c
void garbage_collect_in(Vtree* vtree, SddManager* manager);

//basic/partitions.c
void START_partition(SddManager* manager);
void DECLARE_element(SddNode* prime, SddNode* sub, Vtree* vtree, SddManager *manager);
int GET_elements_of_partition(SddNodeSize* size, SddElement** elements, Vtree* vtree, SddManager* manager, int limited);

//basic/replace.c
void replace_node(int reversible, SddNode* node, SddNodeSize new_size, SddElement* new_elements, Vtree* vtree, SddManager* manager);

//sdds/apply.c
SddNode* sdd_conjoin_lr(SddNode* node1, SddNode* node2, Vtree* lca, SddManager* manager);

//vtree_operations/cartesian_product.c
void open_cartesian_product(SddManager* manager);
int close_cartesian_product(int compress, SddNodeSize* size, SddElement** elements, Vtree* vtree, SddManager* manager, int limited);
void open_partition(SddManager* manager);
void declare_element_of_partition(SddNode* prime, SddNode* sub, Vtree* vtree, SddManager* manager);
int close_partition(int compress, Vtree* vtree, SddManager* manager, int limited);

//vtree_operations/limits.c
void start_op_limits(SddManager* manager);
void end_op_limits(SddManager* manager);
int exceeded_size_limit(SddSize offset_size, SddManager* manager);

//vtree_operations/rollback.c
void finalize_vtree_op(SddNode* replaced_nodes, SddNode* moved_nodes, Vtree* vtree, SddManager* manager);
void rollback_vtree_op(SddNode* replaced_nodes, SddNode* moved_nodes, Vtree* vtree, SddManager* manager);

//vtree_operations/split.c
void split_nodes_for_left_rotate(SddSize* bc_count, SddNode** bc_list, SddNode** c_list, Vtree* w, Vtree* x, SddManager* manager);
void split_nodes_for_right_rotate(SddSize *ab_count, SddNode** ab_list, SddNode** a_list, Vtree* x, Vtree* w, SddManager* manager);
SddNode* split_nodes_for_swap(Vtree* vtree, SddManager* manager);

/****************************************************************************************
 * END
 ****************************************************************************************/
