/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

/****************************************************************************************
 * computation cache
 *
 * there are two hash tables for caching computations, one for conjoin and one for disjoin
 *
 * there are no collision lists in these hash tables: just one entry (similar to CUDD)
 *
 * a cached computation may become invalid when one of its nodes is gc'd; such invalid
 * entries are identified in a lazy manner.
 *
 * a node whose id is 0 must have been gc'd and is sitting on the gc list
 * however, a node whose id is not 0 may have also been gc'd and its structure re-allocated
 * 
 * ids of nodes are monotonically increasing: if a node is gc'd and then re-allocated, 
 * its new id will be larger than the old one. moreover, id=0 is reserved for nodes
 * on the gc list. hence, if the id of a node is different than the one saved, then
 * the node must have been gc'd 
 ****************************************************************************************/

static inline
void delete_computed(SddComputed* computed, SddManager* manager) {
  --manager->computed_count;
  computed->result = NULL; //computed now deleted
}

static inline
SddSize hash_key(SddNode* node1, SddNode* node2) {
  return ((16777619*node1->id)^(node2->id)) % COMPUTED_CACHE_SIZE; 
}

/****************************************************************************************
 * save
 ****************************************************************************************/
 
void cache_computation(SddNode* node1, SddNode* node2, SddNode* node, BoolOp op, SddManager* manager) {
  assert(!GC_NODE(node) && !GC_NODE(node1) && !GC_NODE(node2));
  assert(NON_TRIVIAL(node1) && NON_TRIVIAL(node2));
  
  if(node1->id > node2->id) SWAP(SddNode*,node1,node2); //for hash key
  
  SddSize key           = hash_key(node1,node2);
  SddComputed* table    = op==CONJOIN? manager->conjoin_cache: manager->disjoin_cache;
  SddComputed* computed = table+key;
  
  if(computed->result!=NULL) { //override current computed
    delete_computed(computed,manager);
  }
  
  ++manager->computed_count;
  
  computed->result  = node;
  computed->id    = node->id; //needed to check whether result has been gc'd
  computed->id1   = node1->id; //only id is needed for lookup
  computed->id2   = node2->id; //only id is needed for lookup
}
 
/****************************************************************************************
 * lookup
 ****************************************************************************************/
 
SddNode* lookup_computation(SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager) {
  assert(!GC_NODE(node1) && !GC_NODE(node2));
  assert(NON_TRIVIAL(node1) && NON_TRIVIAL(node2));
  
  if(node1->id > node2->id) SWAP(SddNode*,node1,node2); //for hash key and ordered comparison
  
  SddSize key           = hash_key(node1,node2);
  SddComputed* table    = op==CONJOIN? manager->conjoin_cache: manager->disjoin_cache;
  SddComputed* computed = table+key;
    
  ++manager->computed_cache_lookup_count;
  
  if(computed->result==NULL || computed->id!=computed->result->id) return NULL; //missing or invalid computed
  else if(computed->id1==node1->id && computed->id2==node2->id) {
    ++manager->computed_cache_hit_count; //found it
    return computed->result; //hit
  }
  else return NULL; //miss: non-matching computed for this key
}
 
/****************************************************************************************
 * end
 ****************************************************************************************/
