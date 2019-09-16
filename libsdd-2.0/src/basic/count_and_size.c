/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

/*****************************************************************************************
 * node counts and sdd sizes apply only to sdd nodes that are in the unique table
 *
 * the following counts and sizes are maintained:
 *
 * manager:
 * --node_count: total NUMBER of DEAD/ALIVE nodes in the manager
 * --dead_node_count: total NUMBER of dead nodes in the manager
 * --sdd_size: total SIZE of DEAD/ALIVE nodes (i.e., sum of their decomposition sizes) in the manager
 * --dead_sdd_size: total SIZE of DEAD nodes in the manager
 *
 * vtree:
 * --node_count: total NUMBER of DEAD/ALIVE nodes normalized for vtree
 * --dead_node_count: total NUMBER of DEAD nodes that are normalized for vtree
 * --sdd_size: total SIZE of DEAD/ALIVE nodes (i.e., sum of their decomposition sizes) normalized for vtree
 * --dead_sdd_size: total SIZE of DEAD nodes that are normalized for vtree
 *
 * 
 * counts and sizes change (and require update) when a node:
 * --is added to or removed from the unique table
 * --changes its status between live and dead
 *
 * NOTE: during vtree operations, nodes may be removed temporarily from the unique
 * table to be modified (rotated or swapped). until the vtree operation is concluded,
 * and the nodes are put pack in the unique table, their counts and sizes will not 
 * be reflected by the counts and sizes stored at manager and vtree nodes
 ****************************************************************************************/

/****************************************************************************************
 * sdd node sizes: at vtree nodes and for whole vtrees
 ****************************************************************************************/

//at

SddSize sdd_vtree_size_at(const Vtree* vtree) {
  if(LEAF(vtree)) return 0;
  else return vtree->sdd_size;
}

SddSize sdd_vtree_live_size_at(const Vtree* vtree) {
  if(LEAF(vtree)) return 0;
  else return vtree->sdd_size-vtree->dead_sdd_size;
}

SddSize sdd_vtree_dead_size_at(const Vtree* vtree) {
  if(LEAF(vtree)) return 0;
  else return vtree->dead_sdd_size;
}

//above

//the size of all sdd nodes that are normalized for vnodes above this vtree
SddSize sdd_vtree_size_above(const Vtree* vtree) {
  SddSize size = 0;
  FOR_each_ancestral_vtree_node(v,vtree,size += v->sdd_size);
  return size;
}

//the size of all live sdd nodes that are normalized for vnodes above this vtree
SddSize sdd_vtree_live_size_above(const Vtree* vtree) {
  SddSize size = 0;
  FOR_each_ancestral_vtree_node(v,vtree,size += (v->sdd_size-v->dead_sdd_size));
  return size;
}

//the size of all dead sdd nodes that are normalized for vnodes above this vtree
SddSize sdd_vtree_dead_size_above(const Vtree* vtree) {
  SddSize size = 0;
  FOR_each_ancestral_vtree_node(v,vtree,size += v->dead_sdd_size);
  return size;
}

//inside

//the size of all sdd nodes that are normalized for vnodes in this vtree
SddSize sdd_vtree_size(const Vtree* vtree) {
  SddSize size = 0;
  FOR_each_internal_vtree_node(v,vtree,size += v->sdd_size);
  return size;
}

//the size of all live sdd nodes that are normalized for vnodes in this vtree
SddSize sdd_vtree_live_size(const Vtree* vtree) {
  SddSize size = 0;
  FOR_each_internal_vtree_node(v,vtree,size += (v->sdd_size-v->dead_sdd_size));
  return size;
}

//the size of all dead sdd nodes that are normalized for vnodes in this vtree
SddSize sdd_vtree_dead_size(const Vtree* vtree) {
  SddSize size = 0;
  FOR_each_internal_vtree_node(v,vtree,size += v->dead_sdd_size);
  return size;
}

/****************************************************************************************
 * sdd node counts: at vtree nodes and for whole vtrees
 ****************************************************************************************/

//at

SddSize sdd_vtree_count_at(const Vtree* vtree) {
  if(LEAF(vtree)) return 0;
  else return vtree->node_count;
}

SddSize sdd_vtree_live_count_at(const Vtree* vtree) {
  if(LEAF(vtree)) return 0;
  else return vtree->node_count-vtree->dead_node_count;
}

SddSize sdd_vtree_dead_count_at(const Vtree* vtree) {
  if(LEAF(vtree)) return 0;
  else return vtree->dead_node_count;
}

//above

//number of decomposition nodes that are normalized for vnodes above this vtree
SddSize sdd_vtree_count_above(const Vtree* vtree) {
  SddSize count = 0;
  FOR_each_ancestral_vtree_node(v,vtree,count += v->node_count);
  return count;
}

//number of live decomposition nodes that are normalized for vnodes above this vtree
SddSize sdd_vtree_live_count_above(const Vtree* vtree) {
  SddSize count = 0;
  FOR_each_ancestral_vtree_node(v,vtree,count += (v->node_count-v->dead_node_count));
  return count;
}

//number of dead decomposition nodes that are normalized for vnodes above this vtree
SddSize sdd_vtree_dead_count_above(const Vtree* vtree) {
  SddSize count = 0;
  FOR_each_ancestral_vtree_node(v,vtree,count += v->dead_node_count);
  return count;
}

//inside

//number of decomposition nodes (live and dead) that are normalized for vnodes in this vtree
SddSize sdd_vtree_count(const Vtree* vtree) {
  SddSize count = 0;
  FOR_each_internal_vtree_node(v,vtree,count += v->node_count);
  return count;
}

//number of live decomposition nodes that are normalized for vnodes in this vtree
SddSize sdd_vtree_live_count(const Vtree* vtree) {
  SddSize count = 0;
  FOR_each_internal_vtree_node(v,vtree,count += (v->node_count-v->dead_node_count));
  return count;
}

//number of dead decomposition nodes that are normalized for vnodes in this vtree
SddSize sdd_vtree_dead_count(const Vtree* vtree) {
  SddSize count = 0;
  FOR_each_internal_vtree_node(v,vtree,count += v->dead_node_count);
  return count;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
