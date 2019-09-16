/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/memory.c
void gc_sdd_node(SddNode* node, SddManager* manager);

//basic/nodes.c
void remove_from_unique_table(SddNode* node, SddManager* manager);

/****************************************************************************************
 * mark nodes that need to be gc'd
 * --every dead node in vtree gets gc
 * --dead nodes above vtree are gc'd iff they reference a node inside vtree (whether
 *   the reference is direct or indirect)
 ****************************************************************************************/

static int is_gc(SddNode* node)   { return node->git; }
static int is_dead(SddNode* node) { return node->ref_count==0; }

//static SddSize all;
//static SddSize marked;

//nodes above vtree must be visited bottom up
static
void mark_gc_nodes(Vtree* vtree) {
  //mark nodes inside vtree for gc
  FOR_each_internal_vtree_node(v,vtree,FOR_each_sdd_node_normalized_for(n,v,n->git=1));
  //mark selected nodes above vtree for gc
  Vtree* ancestor = vtree->parent;
  while(ancestor) {
    int lchild = sdd_vtree_is_sub(vtree,ancestor->left);
//    all += ancestor->dead_node_count;
    if(ancestor->dead_node_count) {
      FOR_each_sdd_node_normalized_for(n,ancestor,{
        if(n->ref_count==0) {
          FOR_each_prime_sub_of_node(p,s,n,{
            n->git = lchild? p->git: s->git;
//            if(n->git) ++marked;
            if(n->git) break; //done with node n: it will be gc'd
          });
        }
      });
    }
    ancestor = ancestor->parent;
  }
  FOR_each_internal_vtree_node(v,vtree,FOR_each_sdd_node_normalized_for(n,v,n->git=0));
}

/****************************************************************************************
 * garbage collection of decomposition sdd nodes: moving them to the gc list
 ****************************************************************************************/
 
static inline
void garbage_collect_at(int test(SddNode*), Vtree* vtree, SddManager* manager) {
  if(vtree->dead_node_count) {
    FOR_each_sdd_node_normalized_for(n,vtree,{
      if(test(n)) { //gc node n
        n->git = 0;
        assert(n->parent_count==0); //node cannot have parents
        remove_from_unique_table(n,manager); //first
	    gc_sdd_node(n,manager); //second
	  }
	});
  }
}

//visits ancestors vtree top-down: parents before children
static
void garbage_collect_above(Vtree* vtree, SddManager* manager) {
  Vtree* root = manager->vtree;
  while(root!=vtree && manager->dead_node_count) {
    garbage_collect_at(is_gc,root,manager);
//    garbage_collect_at(is_dead,root,manager);
    if(sdd_vtree_is_sub(vtree,root->left)) root = root->left;
    else root = root->right;
  }
}

//visits nodes inside vtree top-down: parents before children
void garbage_collect_in(Vtree* vtree, SddManager* manager) {
  if(INTERNAL(vtree) && manager->dead_node_count) {
    garbage_collect_at(is_dead,vtree,manager);
    garbage_collect_in(vtree->left,manager);
    garbage_collect_in(vtree->right,manager);
  }
}

//visit nodes top-down: when a node is gc'd, it must have no parents
void sdd_vtree_garbage_collect(Vtree* vtree, SddManager* manager) {
  mark_gc_nodes(vtree);
  garbage_collect_above(vtree,manager);
  garbage_collect_in(vtree,manager);
  assert(!FULL_DEBUG || verify_counts_and_sizes(manager));
  assert(!FULL_DEBUG || verify_gc(vtree,manager));
}

/****************************************************************************************
 * end
 ****************************************************************************************/
