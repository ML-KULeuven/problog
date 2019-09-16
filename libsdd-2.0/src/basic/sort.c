/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

/****************************************************************************************
 * sorting nodes and elements
 ****************************************************************************************/

/****************************************************************************************
 * sorting a linked list of nodes (linked using the ->next field)
 *
 * this is use to process nodes according to their size (e.g., in rotate and swap)
 ****************************************************************************************/

//compare nodes based on size then id (no two nodes are equal according to this function,
//which leaves no ambiguity when sorting)
static inline
int size_cmp(const void* n1, const void* n2) {
  const SddNodeSize s1 = (*((const SddNode**)n1))->size;
  const SddNodeSize s2 = (*((const SddNode**)n2))->size;
  if(s1 > s2) return 1; //(s1 > s2): smaller to larger
  else if(s1 < s2) return -1; //(s1 < s2): smaller to larger
  else {
    const SddSize id1 = (*((const SddNode**)n1))->id;
    const SddSize id2 = (*((const SddNode**)n2))->id;
    if(id1 > id2) return 1;
    else if(id1 < id2) return -1;
    else return 0;
  }
}

//sort a linked list of sdd nodes
void sort_linked_nodes(SddSize count, SddNode** list, SddManager* manager) {
  if(count<2) return;
  
  //make sure node buffer is big enough
  if(count > manager->node_buffer_size) {
    manager->node_buffer_size = 2*count;
    REALLOC(manager->node_buffer,SddNode*,manager->node_buffer_size,"sort_linked_nodes");
  }
  
  SddNode** buffer = manager->node_buffer;
  
  //place nodes in buffer, then sort
  FOR_each_linked_node(n,*list,*buffer++=n); buffer -= count;
  qsort((SddNode**)buffer,count,sizeof(SddNode*),size_cmp);
  //put sorted nodes back into a linked list
  while(--count) { (*buffer)->next = *(buffer+1); ++buffer; }
  (*buffer)->next = NULL; //last node

  *list = manager->node_buffer[0]; //first node
  //check smaller to larger order
  assert((*list)->size <= (*list)->next->size); //smaller to larger size
}

/****************************************************************************************
 * sorting an array of elements
 *
 * elements of a node are sorted for 
 * --efficient lookup from unique node tables
 * --use during element compression (expects the elements to be sorted)
 *
 * elements are sorted by the id of subs (this leaves no ambiguity when sorting
 * compressed elements)
 ****************************************************************************************/
 
//sort elements from larger to smaller sub->id
// then from smaller to larger prime sizes
// then from smaller to larger prime ids
//
//Note: want no ambiguity in the final order of elements, otherwise the sort is unstable,
//producing different results on different machines
//
static inline
int cmp_by_sub_id_L(const void* e1, const void* e2) {
  //sort by id of sub: smaller to larger id
  const SddSize sid1 = ((const SddElement*)e1)->sub->id;
  const SddSize sid2 = ((const SddElement*)e2)->sub->id;
  if(sid1 > sid2) return 1;
  else if(sid1 < sid2) return -1;
  else {
    //sort by prime size: smaller to larger
    //this affects order of disjoining primes during compression
    //this appears more efficient than sorting larger to smaller
    const SddNodeSize ps1 = ((const SddElement*)e1)->prime->size;
    const SddNodeSize ps2 = ((const SddElement*)e2)->prime->size;
    if(ps1 > ps2) return 1;
    else if(ps1 < ps2) return -1;
    else {
      //sort by prime id: smaller to larger
      //so the element order is unique
  	  //without this, final element order may depend on system
      const SddSize pid1 = ((const SddElement*)e1)->prime->id;
      const SddSize pid2 = ((const SddElement*)e2)->prime->id;
      if(pid1 > pid2) return 1;
      else if(pid1 < pid2) return -1;
      else return 0;
    }
  }
}

//sort by id of sub: larger to smaller sub->id
//used to sort elements that are compressed (i.e., with distinct)
static inline
int cmp_by_sub_id_G(const void* e1, const void* e2) {
  const SddSize sid1 = ((const SddElement*)e1)->sub->id;
  const SddSize sid2 = ((const SddElement*)e2)->sub->id;
  if(sid1 < sid2) return 1;
  else if(sid1 > sid2) return -1;
  else {
    assert(0); //should not reach this since subs are distinct
    return 0;
  }
}

//sort elements from larger to smaller sub->id (then other tie breakers -- see above)
//equal subs will be adjacent in the sorted list
//
//NOTE: this is using cmp_by_sub_id_L (above) instead of cmp_by_sub_id_G (below) since
//the order of these elements will be reversed later when pushed on the elements stack
//
void sort_uncompressed_elements(SddSize size, SddElement* elements) {
  qsort((SddElement*)elements,size,sizeof(SddElement),cmp_by_sub_id_L);
}

//check that elements are sorted (from larger to smaller sub->id)
//check that all subs are distinct (i.e., compressed complements)
int elements_sorted_and_compressed(SddNodeSize size, SddElement* elements) {
  assert(size > 1);
  for(SddNodeSize i=1; i<size; i++) {
    if(elements[i-1].sub->id <= elements[i].sub->id) return 0;
  }
  return 1;
}

//sort elements from larger to smaller sub->id
//assumes that subs are distinct (i.e., their ids are distinct)
void sort_compressed_elements(SddNodeSize size, SddElement* elements) {
  qsort((SddElement*)elements,size,sizeof(SddElement),cmp_by_sub_id_G);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
