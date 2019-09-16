/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

/********************************************************************************************
 * allowed sized of hash tables
 *******************************************************************************************/

//the maximum qsize when resizing a hash table
#define MAX_QSIZE_SDD_NODE_HASH 21

//hash tables have sizes which are prime
//16 different sizes are predefined
//when a hash table is resized, the next size is used (roughly doubles the previous size) 
//the largest hash table has about 10 million collision lists
static SddSize hash_qsizes[] = {
  317,
  631, 
  1259, 
  2503, 
  5003, 
  10007, 
  20001, 
  40009, 
  80021, 
  160001, 
  320009, 
  640007, 
  1280023, 
  2560021, 
  5000011, 
  10000019,
  20000003,
  40000003,
  80000023,
  160000003,
  320000077,
  640000019
};

/********************************************************************************************
 * creating hash tables
 *******************************************************************************************/

SddHash* new_unique_node_hash(SddManager* manager) {
  //convert qualitative size to numeric size
  SddSize size = hash_qsizes[INITIAL_SIZE_UNIQUE_NODE_TABLE];
  
  SddHash* hash;
  MALLOC(hash,SddHash,"NEW_HASH");
  //allocate array of collision lists
  //array must be initialized to null pointers: empty collision lists
  CALLOC(hash->clists,SddNode*,size,"NEW_HASH");
  
  hash->size  = size;
  hash->qsize = INITIAL_SIZE_UNIQUE_NODE_TABLE;
  hash->count = 0;
  hash->lookup_count = 0;
  hash->hit_count    = 0;
  hash->resize_age   = 0;
  hash->lookup_cost  = 0;
  hash->increase_size_count = 0;
  hash->decrease_size_count = 0;
  
  return hash;
}

/********************************************************************************************
 * freeing hash tables
 *******************************************************************************************/

void free_hash(SddHash* hash) {
  free(hash->clists);
  free(hash);
}

/********************************************************************************************
 * hash keys
 *******************************************************************************************/

//hash key of an sdd node that has not been constructed yet
static inline
SddSize key_sdd_elements(SddSize size, SddElement* elements, SddHash* hash) {
  SddSize key = 0;
  FOR_each_prime_sub_of_elements(p,s,size,elements,{ 
    key += (16777619*key)^(p->id);
    key += (16777619*key)^(s->id);
  }); 
  return key % hash->size;
}

//hash key for existing node
static inline
SddSize key_sdd_node(SddNode* node, SddHash* hash) {
  return key_sdd_elements(node->size,ELEMENTS_OF(node),hash);
}

/********************************************************************************************
 * hash table properties
 *******************************************************************************************/

inline
float hit_rate(SddHash* hash) {
  return 100.0*hash->hit_count/hash->lookup_count;
}

inline
float ave_lookup_cost(SddHash* hash) {
  return ((float)hash->lookup_cost)/hash->lookup_count;
}

//percentage of entries in hash table that have non-empty collision lists 
float saturation(SddHash* hash) {
  SddSize count = 0;
  for(SddNode** clist=hash->clists; clist < hash->clists+hash->size; ++clist) {
    if(*clist) ++count;
  }
  return ((float)count)/hash->size;
}

/********************************************************************************************
 * resizing hash table
 *******************************************************************************************/

//resizing up or down depends on hash table load
//keeps contents as is
void try_resizing_hash(SddHash* hash, SddManager* manager) {
  
  if(hash->qsize != MAX_QSIZE_SDD_NODE_HASH && hash->count > LOAD_TO_INCREASE_HASH_SIZE*hash->size) {
    ++hash->qsize; //increase size
    ++hash->increase_size_count; 
  }
  else if(hash->qsize != 0 && hash->count < LOAD_TO_DECREASE_HASH_SIZE*hash->size) {
    --hash->qsize; //decrease size
    ++hash->decrease_size_count;
  }
  else return; //no resizing

  //save old table
  SddSize old_size     = hash->size;
  SddNode** old_clists = hash->clists;
  
  //new array of collision lists
  hash->size = hash_qsizes[hash->qsize];
  CALLOC(hash->clists,SddNode*,hash->size,"resize_sdd_node_hash");
  
  //insert nodes into new array of collision lists (using new hash keys)
  for(SddNode** old_clist=old_clists; old_clist<old_clists+old_size; ++old_clist) {
    FOR_each_linked_node(node,*old_clist,{
      SddNode** new_clist = hash->clists+key_sdd_node(node,hash); //location of new collision list
      if(*new_clist) (*new_clist)->prev = &(node->next);
	  node->next = *new_clist;
	  node->prev = new_clist;
      *new_clist = node;
    });
  }
  
  free(old_clists); //free old table
  hash->resize_age = 0; //reset resize age
}

/********************************************************************************************
 * looking up entries in hash tables
 *******************************************************************************************/

//assumes elements are sorted
SddNode* lookup_sdd_node(SddElement* elements, SddNodeSize size, SddHash* hash, SddManager* manager) {	
  ++hash->lookup_count;
  ++hash->resize_age;
  //search
  SddNode* node = *(hash->clists+key_sdd_elements(size,elements,hash)); //collision list
  SddSize s = size*sizeof(SddElement);
  while(node) {
    ++hash->lookup_cost;
	if(node->size==size && memcmp(ELEMENTS_OF(node),elements,s)==0) { //found it
	  ++hash->hit_count;
	  return node;
	}
	else node=node->next;
  }
  return NULL; //did not find it
}

/********************************************************************************************
 * inserting entries in hash table
 *******************************************************************************************/

//insert node into hash table
void insert_sdd_node(SddNode* node, SddHash* hash, SddManager* manager) {
  ++hash->count;
  SddNode** clist = hash->clists+key_sdd_node(node,hash); //collision list location
  if(*clist) (*clist)->prev = &(node->next);
  node->next = *clist;
  node->prev = clist;
  *clist = node;
  //resizing is tried only after adding node to hash table (and not after removing nodes)
  if(hash->resize_age > AGE_BEFORE_HASH_RESIZE_CHECK) try_resizing_hash(hash,manager);
}

/********************************************************************************************
 * removing entries in hash table
 *******************************************************************************************/
 
void remove_sdd_node(SddNode* node, SddHash* hash, SddManager* manager) {
  --hash->count;
  if(node->next) node->next->prev = node->prev;
  *(node->prev) = node->next;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
