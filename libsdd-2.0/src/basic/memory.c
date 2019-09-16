/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/nodes.c
void declare_lost_parent(SddNode* node, SddManager* manager);

/*****************************************************************************************
 * the following are critical assumptions about ids of nodes:
 *
 * --id 0 is reserved for nodes on the gc list
 * --id 1 is reserved for false
 * --id 2 is reserved for true
 * --id of true  = (id of false) + 1 (see basic/compress.c)
 * --ids are NEVER re-used: every newly created sdd node (by malloc or recovery from the
 *   list) gets a new id; hence, the ids of nodes returned by new_sdd_node() are 
 *   monotonically increasing (see the definition of macro INVALID_COMPUTED)
 ****************************************************************************************/
 
/*****************************************************************************************
 * nodes are either:
 * --ALLOCATED: terminal sdd nodes, or nodes indexed by a unique table
 * --GC'D: part of the gc list
 *
 * allocated decomposition nodes are either:
 * --DEAD: have reference count == 0 (ready to be garbage collected)
 * --LIVE: have reference count > 0
 *
 ****************************************************************************************/

/****************************************************************************************
 * allocating nodes
 * first, allocation is attempted from gc lists
 * otherwise, malloc is called
 *
 * elements are different though: rotation and swap replace the elements of a node,
 * so they create and free elements which are not accounted for by manager->sdd_size
 * and manager->gc_element_count). hence, one must explicitly keep track of the
 * maximum number of elements existing in memory.
 ****************************************************************************************/

/****************************************************************************************
 * allocating and freeing sdd elements
 ****************************************************************************************/
 
//allocate an array to hold elements
SddElement* new_elements(SddNodeSize size, SddManager* manager) {
  assert(size>0);
  SddElement* elements;
  CALLOC(elements,SddElement,size,"new_element_array");
  manager->stats.element_count += size; //number of elements currently in memory
  manager->stats.max_element_count = MAX(manager->stats.max_element_count,manager->stats.element_count);
  return elements;
}

void free_elements(SddNodeSize size, SddElement* elements, SddManager* manager) {
  assert(size>0 || elements==NULL);
  assert(manager->stats.element_count >= size);
  free(elements);
  manager->stats.element_count -= size; //number of elements currently in memory
  assert(manager->stats.max_element_count >= manager->stats.element_count);
}

/****************************************************************************************
 * allocating sdd nodes 
 ****************************************************************************************/
 
//gc'd nodes are kept in linked lists, depending on their size
//if size is >0 and <GC_BUCKETS_COUNT, then the node is stored in gc_node_lists[size]
//if size >= GC_BUCKETS_COUNT, then the node is stored in gc_node_lists[0]
//allocate from gc list if non empty, otherwise malloc
//new node is normalized for vtree
SddNode* new_sdd_node(SddNodeType type, SddNodeSize size, Vtree* vtree, SddManager* manager) {

  SddNode* node;
  
  if(type!=DECOMPOSITION) { //allocate terminal node
    MALLOC(node,SddNode,"new_sdd_node");
  }
  else { //allocate decomposition 
    assert(size > 0);
    //allocate decomposition node
    int bucket = (size < GC_BUCKETS_COUNT? size: 0);
    node       = manager->gc_node_lists[bucket];	
    if(node != NULL) { //allocating from gc list
      --manager->gc_node_count;
      manager->gc_element_count      -= node->size; //0 for bucket 0
      manager->gc_node_lists[bucket]  = node->next; //remove node from gc list
      if(bucket==0) ELEMENTS_OF(node) = new_elements(size,manager);
      //nodes in bucket zero have no elements allocated (since these nodes
      //were of varying size when they were put there, their elements were freed)
    }
    else { //allocating new memory
      MALLOC(node,SddNode,"new_sdd_node");
      ELEMENTS_OF(node) = new_elements(size,manager);
    }
  }

  //initialize
  node->id             = ++manager->id_counter;
  node->vtree          = vtree; //node is normalized for vtree
  node->type           = type;
  node->size           = size;
  node->negation       = NULL;
  node->multiply_sub   = NULL;
  node->map            = NULL;
  node->shadow         = NULL;
  node->saved_elements = NULL;
  node->saved_size      = 0;
  node->ref_count       = 0; //nodes are born dead
  node->parent_count    = 0;
  node->index           = 0;
  node->bit             = 0;
  node->cit             = 0;
  node->dit             = 0;
  node->git             = 0;
  node->in_unique_table = 0;
  node->replaced        = 0;
  node->user_bit        = 0;
  
  return node;
}


/****************************************************************************************
 * garbage collecting (freeing) sdd nodes 
 ****************************************************************************************/

//gc'd nodes are kept in linked lists, depending on their size
//if size is >=0 and <GC_BUCKETS_COUNT, then the node is stored in gc_node_lists[size]
//if size >= GC_BUCKETS_COUNT, then the node is stored in gc_node_lists[0]
//
//note: literal sdds are not garbage collected except when their corresponding variable
//is removed from the manager by remove_var().
//
//literal sdds go into gc_node_lists[0]
//
void gc_sdd_node(SddNode* node, SddManager* manager) {
  assert(node->parent_count==0); //nothing should be pointing to node
  assert(NON_TRIVIAL(node)); //only decomposition and literal nodes
    
  //remove reference by negation reference (if any)
  if(node->negation!=NULL) node->negation->negation=NULL;
  
  if(IS_DECOMPOSITION(node)) {
    assert(node->ref_count==0); //must be dead
    assert(node->in_unique_table==0); //cannot be in unique table
    declare_lost_parent(node,manager); //before gc which may free elements
  }
  
  //add to the corresponding free list
  ++manager->gc_node_count;
  manager->gc_element_count += node->size; //no change for literal sdds
  
  int bucket = (node->size < GC_BUCKETS_COUNT? node->size: 0); //literal sdds go into bucket 0
  node->next = manager->gc_node_lists[bucket]; 
  manager->gc_node_lists[bucket] = node; //add node to corresponding bucket
  
  if(bucket==0 && IS_DECOMPOSITION(node)) {
    //nodes put in this bucket have varying size so their elements cannot be easily reused
    //we will therefore free these elements
    manager->gc_element_count -= node->size; //correction: these elements are not on gc-list
    free_elements(node->size,ELEMENTS_OF(node),manager);
    node->size = 0;
    ELEMENTS_OF(node) = NULL; //so that free(ELEMENTS_OF(node)) is still ok
  }

  node->id = 0; //node is now garbage
  
  //note: size and element array are kept intact except for bucket 0
}

/****************************************************************************************
 * end
 ****************************************************************************************/
