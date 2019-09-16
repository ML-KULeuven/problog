/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

/****************************************************************************************
 * several functions which can be used to verify the coherence of a manager:
 *
 * --vtree in-order (positions) and variable counts
 * --manager/vtree/unique-table counts and sizes
 * --sdd node normalization
 * --sdd node negations
 * --garbage collection
 ****************************************************************************************/

//if S does not hold, print S and return 0
#define VERIFY(S) if(!(S)) { printf("\nFailed: "#S"\n"); return 0; }

/****************************************************************************************
 * vtree in-order (positions) and variable counts
 ****************************************************************************************/

int verify_vtree_properties(const Vtree* vtree) {
 
  //verify leaves
  FOR_each_leaf_vtree_node(v,vtree,{
    VERIFY(v->var_count==1);
    VERIFY(v==v->first);
    VERIFY(v==v->last);
  });
 
  //verify internal nodes
  FOR_each_internal_vtree_node(v,vtree,{
    VERIFY(LEAF(v->first));
    VERIFY(LEAF(v->last));
    VERIFY(v->first==v->left->first);
    VERIFY(v->last==v->right->last);
    VERIFY(v->left->last->next==v);
    VERIFY(v->right->first->prev==v);
    VERIFY(v->prev==v->left->last);
    VERIFY(v->next==v->right->first);
    VERIFY(v->first->prev==NULL || v->first->prev->next==v->first);
    VERIFY(v->last->next==NULL  || v->last->next->prev==v->last);
    VERIFY(v->var_count==v->left->var_count+v->right->var_count);
    VERIFY(v->first->position < v->last->position);
    VERIFY(v->position>v->first->position);
    VERIFY(v->position<v->last->position);
    VERIFY((v->last->position-v->first->position+1)==(2*v->var_count -1));
  });
  
  return 1;
}

static
int verify_X_constrained_aux(const Vtree* vtree) {
  if(LEAF(vtree)) return vtree->some_X_constrained_vars;
  else {
    int l = verify_X_constrained_aux(vtree->left);
    int r = verify_X_constrained_aux(vtree->right);
    VERIFY(l || r || vtree->some_X_constrained_vars==0);
    return vtree->some_X_constrained_vars;
  }
}

int verify_X_constrained(const Vtree* vtree) {
  verify_X_constrained_aux(vtree);
  
  const Vtree* r = vtree;
  while(INTERNAL(r) && r->some_X_constrained_vars) r = r->right;
  //if r is root, then there are no x constrained variables
  
  VERIFY(r->some_X_constrained_vars==0);
  
  FOR_each_vtree_node(v,vtree,{
    VERIFY(v->some_X_constrained_vars || sdd_vtree_is_sub(v,r));
  });
  
  return 1;
}

/****************************************************************************************
 * counts ans sizes
 *
 * dead/live counts and sizes match between vtrees, manager, and normalized node lists
 ****************************************************************************************/
 
int verify_counts_and_sizes(const SddManager* manager) {

  SddSize vtree_count       = 0;
  SddSize vtree_dead_count  = 0;
  SddSize vtree_size        = 0;
  SddSize vtree_dead_size   = 0;

  FOR_each_internal_vtree_node(v,manager->vtree,{
    VERIFY(v->node_count >= v->dead_node_count);
    
    SddSize live_count = 0;
    SddSize dead_count = 0;
    SddSize live_size  = 0;
    SddSize dead_size  = 0;
    
    FOR_each_sdd_node_normalized_for(n,v,{ //iterating over nodes n normalized for vtree node v
      if(n->ref_count) { //live node
        ++live_count;
        live_size += n->size;
      } 
      else { //dead node
        ++dead_count;
        dead_size += n->size;
      }
    });
    
    VERIFY(v->node_count==live_count+dead_count);
    VERIFY(v->dead_node_count==dead_count);
    VERIFY(v->sdd_size==live_size+dead_size);
    VERIFY(v->dead_sdd_size==dead_size);
    
    vtree_count      += live_count+dead_count;
    vtree_dead_count += dead_count;
    vtree_size       += live_size+dead_size;
    vtree_dead_size  += dead_size;
  });
 
  VERIFY(manager->node_count==vtree_count);
  VERIFY(manager->dead_node_count==vtree_dead_count);
  VERIFY(manager->sdd_size==vtree_size);
  VERIFY(manager->dead_sdd_size==vtree_dead_size);
  
  VERIFY(manager->node_count==sdd_vtree_count(manager->vtree));
  VERIFY(manager->dead_node_count==sdd_vtree_dead_count(manager->vtree));
  VERIFY(manager->sdd_size==sdd_vtree_size(manager->vtree));
  VERIFY(manager->dead_sdd_size==sdd_vtree_dead_size(manager->vtree));

  return 1;
}
 
/****************************************************************************************
 * normalization
 *
 * nodes normalized for correct vtree
 ****************************************************************************************/
 
int verify_normalization(const SddManager* manager) {
  Vtree* root = manager->vtree;
  FOR_each_internal_vtree_node(v,root,
    FOR_each_sdd_node_normalized_for(node,v,VERIFY(node->vtree==v)));
  FOR_each_decomposition_in(n,root,{
    Vtree* l = NULL;
    Vtree* r = NULL;
    FOR_each_prime_sub_of_node(p,s,n,{
      if(p->vtree) {
         if(l) l=sdd_vtree_lca(p->vtree,l,root); else l=p->vtree;
      }
      if(s->vtree) {
        if(r) r=sdd_vtree_lca(s->vtree,r,root); else r=s->vtree;
      }
    });
    VERIFY(sdd_vtree_is_sub(l,n->vtree->left));
    VERIFY(sdd_vtree_is_sub(r,n->vtree->right));
    VERIFY(n->vtree==sdd_vtree_lca(l,r,root));
  });
  return 1;
}

/****************************************************************************************
 * negations
 *
 * nodes and their negations point to each other
 * nodes and their negations normalized for same vtree
 ****************************************************************************************/

int verify_negations(const SddManager* manager) {
  FOR_each_decomposition_in(n,manager->vtree,{
    if(n->negation) {
      VERIFY(!GC_NODE(n->negation));
      VERIFY(n==n->negation->negation);
      VERIFY(n->vtree==n->negation->vtree);
    }
  });
  return 1;
}

/****************************************************************************************
 * garbage collection
 ****************************************************************************************/

static 
int check_gc_at(const Vtree *vtree) {
  VERIFY(vtree->dead_node_count==0);
  FOR_each_sdd_node_normalized_for(n,vtree,{
    VERIFY(n->ref_count);
    VERIFY(!GC_NODE(n));
    FOR_each_prime_sub_of_node(p,s,n,{
      VERIFY(p->parent_count);
      VERIFY(s->parent_count);
    });
  });
  return 1;
}

static
int check_gc_above(const Vtree* vtree) {
  FOR_each_ancestral_vtree_node(v,vtree,if(!check_gc_at(v)) return 0);
  return 1;
}
 
static
int check_gc_in(const Vtree* vtree) {
  FOR_each_internal_vtree_node(v,vtree,if(!check_gc_at(v)) return 0);
  return 1;
}

//checks coherence of garbage collector
int verify_gc(const Vtree* vtree, SddManager* manager) {
  VERIFY(check_gc_in(vtree));
  VERIFY(check_gc_above(vtree));
  
  //verifying parent_count
  FOR_each_unique_node(n,manager,n->index=0);
  FOR_each_unique_node(n,manager,{
    FOR_each_prime_sub_of_node(p,s,n,{
      ++p->index;
      ++s->index;
    });
  });
  FOR_each_unique_node(n,manager,VERIFY(n->index==n->parent_count));
  
  return 1;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
