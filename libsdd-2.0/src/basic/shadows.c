/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/memory.c
SddElement* new_elements(SddNodeSize size, SddManager* manager);

//basic/nodes.c
void remove_from_unique_table(SddNode* node, SddManager* manager);
void insert_in_unique_table(SddNode* node, SddManager* manager);

//basic/replace.c
void replace_node(int reversible, SddNode* node, SddNodeSize new_size, SddElement* new_elements, Vtree* vtree, SddManager* manager);

/****************************************************************************************
 * A shadow-DAG is constructed for a multi-rooted sdd-DAG. 
 *
 * A shadow-DAG is used to create a light replica of an sdd-DAG, which can be used 
 * for various purposes, including:
 *
 * --recovering the sdd-DAG to its original state after it has been modified 
 *   (for example, during a rollback of vtree operations).
 * --navigate an sdd-DAG when navigation operations may change the DAG structure
 *   (for example, by changing the underlying vtree).
 *
 * To create a shadow-DAG, one needs: 
 * --a sdd-DAG which is defined by its root nodes (root_nodes)
 * --a set of non-root nodes in the sdd DAG (changeable_nodes)
 *
 * The semantics and assumptions underlying shadow-DAGs are as follows:
 * --all nodes in the sdd-DAG must be live when a shadow-DAG is constructed
 * --after constructing a shadow-DAG, and before using it to recover the sdd-DAG:
 *   --sdd nodes outside the sdd-DAG remain intact
 *   --root_nodes and changeable_nodes are not gc'd
 *   --root_nodes and changeable_nodes are the only nodes whose elements may be changed
 * --root_nodes and changeable_nodes are decomposition nodes
 *
 * When recovering the sdd-DAG from its shadow-DAG, the structures of root_nodes and 
 * changeable_nodes are reused.
 *
 * Only a subset of the sdd-DAG needs to be replicated, which is defined as follows. 
 * Let n be a node in the sdd-DAG which has external references from outside the 
 * sdd-DAG, and which does not belong to root_nodes or changeable_nodes. This will be 
 * called a VIRTUAL leaf of the sdd-DAG. The subset of the sdd-DAG which will be 
 * replaced has root_nodes as its roots, and virtual leaves as its leaves.
 *
 * An implication of the assumptions underlying shadow-DAGs is that virtual leaves 
 * will neither be gc'd nor their elements changed before an sdd-DAG is recovered
 * from its shadow-DAG. 
 ****************************************************************************************/

/****************************************************************************************
 * A shadow-DAG is defined recursively as follow.
 *
 * A SPEC for decomposition node n consists of:
 * --the size of node n.
 * --shadows for its primes and subs.
 *
 * A SHADOW for sdd node n is one of three types:
 *
 * --Type 't': A pointer to node n.
 *   To recover node n from this shadow, we simply return the node pointer.
 *
 *   These shadows are used only for virtual leaf nodes of the sdd-DAG.
 *
 * --Type 'c': A spec for decomposition node n + a pointer to node n.
 *   To recover node n from this shadow, we first recover its primes and subs from the
 *   shadow spec, and then use them to replace the current elements of node n.
 *
 *   These shadows are used only for root_nodes and changeable_nodes.
 *
 * --Type 'g': A spec for decomposition node n.
 *   To recover node n from this shadow, we first recover its primes and subs from the
 *   shadow spec, and then use them to lookup the node from the unique node tables or 
 *   recreate it.
 *
 *   These shadows are used for all other replicated nodes of the sdd-DAG. 
 *
 *
 * Hence, leaves of the shadow-DAG are type 't' shadows, roots are type 'c' shadows, 
 * and internal nodes are type 'g' or type 'c' shadows.
 *
 * Shadow-DAGs are supported by three functions:
 *
 * --shadows_new()
 * --shadows_recover()
 * --shadows_free()
 ****************************************************************************************/

#define NODE_OF(S) (S->alpha.node)
#define ELMS_OF(S) (S->alpha.elements)

#define INTERNAL_g(S) (S->size>0 && S->reuse==0)
#define INTERNAL_c(S) (S->size>0 && S->reuse==1)

SddNode* shadow_node(NodeShadow* shadow) {
  assert(shadow->size==0);
  return NODE_OF(shadow);
}

ElmShadow* shadow_elements(NodeShadow* shadow) {
  assert(shadow->size>0);
  return ELMS_OF(shadow);
}

int shadow_is_terminal(NodeShadow* shadow) {
  return shadow->size==0;
}
int shadow_is_internal(NodeShadow* shadow) {
  return shadow->size>0;
}

/****************************************************************************************
 * allocating and freeing shadows and specs
 ****************************************************************************************/

static 
NodeShadow* leaf_shadow_new(SddNode* node, SddShadows* shadows) {
  assert(node);
  assert(node->shadow_type=='t');
  ++shadows->shadow_count;
  shadows->shadow_byte_count += sizeof(NodeShadow);
  
  NodeShadow* shadow;
  MALLOC(shadow,NodeShadow,"leaf_shadow_new");
  
  sdd_ref(node,shadows->manager);  //protect
  
  NODE_OF(shadow)      = node;
  shadow->vtree        = node->vtree;
  shadow->size         = 0;
  shadow->ref_count    = 1;
  shadow->cache        = NULL;
  shadow->bit          = 0;
  shadow->reuse        = 0;
  return shadow;
}

static 
NodeShadow* internal_shadow_new(SddNode* node, SddShadows* shadows) {
  assert(node->shadow_type=='g' || node->shadow_type=='c');
  ++shadows->shadow_count;
  shadows->shadow_byte_count += sizeof(NodeShadow) + node->size*sizeof(ElmShadow);
  
  NodeShadow* shadow;
  MALLOC(shadow,NodeShadow,"internal_shadow_new");
  
  CALLOC(ELMS_OF(shadow),ElmShadow,node->size,"internal_shadow_new");
  shadow->vtree        = node->vtree;
  shadow->size         = node->size;
  shadow->ref_count    = 1;
  shadow->cache        = node->shadow_type=='c'? node: NULL; //reuse node structure
  shadow->bit          = 0;
  shadow->reuse        = node->shadow_type=='c';
  return shadow;
}

static 
void leaf_shadow_free(NodeShadow* shadow, SddShadows* shadows) {
  assert(shadow_is_terminal(shadow));
  assert(shadows->shadow_count);
  --shadows->shadow_count;
  shadows->shadow_byte_count -= sizeof(NodeShadow);
  
  SddNode* node = NODE_OF(shadow);
  if(node) sdd_deref(node,shadows->manager); //release
  
  free(shadow);
}

/****************************************************************************************
 * converting shadow types
 ****************************************************************************************/

static
void convert_internal_shadow_to_leaf(SddNode* node, NodeShadow* shadow, SddShadows* shadows) {
  assert(shadow_is_internal(shadow));
  shadows->shadow_byte_count -= shadow->size*sizeof(ElmShadow);
  
  if(node) sdd_ref(node,shadows->manager); //protect
  
  free(ELMS_OF(shadow)); //before setting node
  NODE_OF(shadow)  = node; //may be NULL (when freeing shadows)
  shadow->vtree    = node? node->vtree: NULL;
  shadow->size     = 0;
  //ref_count, cache, bit, reuse unchanged
}

/****************************************************************************************
 * constructing shadows for sdd nodes
 *
 * assumes node->shadow has been initialized to NULL, and node->shadow_type has been set
 ****************************************************************************************/

static
NodeShadow* shadow_from_node(SddNode* node, SddShadows* shadows) {
  assert(node->shadow_type=='t' || node->shadow_type=='g' || node->shadow_type=='c');
  
  if(node->shadow) { //already constructed
    ++node->shadow->ref_count; //additional reference
    return node->shadow; 
  }

  //construct shadow
  NodeShadow* shadow;
      
  if(node->shadow_type=='t') shadow = leaf_shadow_new(node,shadows);
  else {
    assert(node->type==DECOMPOSITION);
    shadow                = internal_shadow_new(node,shadows);
    SddElement* elements  = ELEMENTS_OF(node); //shadow these elements
    ElmShadow* s_elements = ELMS_OF(shadow);
    for(SddNodeSize i=0; i<node->size; i++) {
      s_elements[i].prime = shadow_from_node(elements[i].prime,shadows);
      s_elements[i].sub   = shadow_from_node(elements[i].sub,shadows);
    }
  }
  
  return node->shadow = shadow;
}

/****************************************************************************************
 * recovering a node from its shadow (frees shadows along the way)
 ****************************************************************************************/

static
SddNode* node_from_shadow(NodeShadow* shadow, SddShadows* shadows) {
  assert(shadow->ref_count);
  
  SddNode* node;
   
  if(shadow_is_terminal(shadow)) node = NODE_OF(shadow); //terminal shadow: lookup node
  else {
    SddManager* manager   = shadows->manager;
    SddNodeSize size      = shadow->size;
    Vtree* vtree          = shadow->vtree;
    ElmShadow* s_elements = ELMS_OF(shadow);
    
    if(INTERNAL_g(shadow)) { //internal shadow with no reuse: lookup or recreate node
      GET_node_from_compressed_partition(node,vtree,manager,{
        for(SddNodeSize i=0; i<size; i++) {
          SddNode* prime = node_from_shadow(s_elements[i].prime,shadows);
          SddNode* sub   = node_from_shadow(s_elements[i].sub,shadows);
          DECLARE_compressed_element(prime,sub,vtree,manager);
        }
      });
      assert(node->vtree==vtree);
    }
    else { //internal shadow with reuse: replace elements/vtree of saved node structure
      node = shadow->cache;   
      assert(node->in_unique_table);
      SddElement* elements  = new_elements(size,manager);
      for(SddNodeSize i=0; i<size; i++) {
        elements[i].prime = node_from_shadow(s_elements[i].prime,shadows);
        elements[i].sub   = node_from_shadow(s_elements[i].sub,shadows);
      } 
      remove_from_unique_table(node,manager); //hash key will change
      int reversible = 0; //replacement irreversible (current elements will be freed)
      replace_node(reversible,node,size,elements,vtree,manager);
      insert_in_unique_table(node,manager);
    }
    
    //elements no longer needed: convert to leaf shadow
    convert_internal_shadow_to_leaf(node,shadow,shadows);
  }
  
  if(--shadow->ref_count==0) leaf_shadow_free(shadow,shadows); //shadow no longer needed
  
  return node;
 }
 
/****************************************************************************************
 * traversing a shadow and its descendants, applying function to each visited shadow
 ****************************************************************************************/
  
void shadow_traverse(int bit, NodeShadow* shadow, void (*fn)(NodeShadow*,SddShadows*), SddShadows* shadows) {
  if(shadow->bit==bit) return;
  shadow->bit = bit;
  (*fn)(shadow,shadows); //apply function to shadow
  if(shadow_is_internal(shadow)) {
    ElmShadow* elements = ELMS_OF(shadow);
    for(ElmShadow* e=elements; e<elements+shadow->size; e++) {
      shadow_traverse(bit,e->prime,fn,shadows);
      shadow_traverse(bit,e->sub,fn,shadows);
    }
  }
}

/****************************************************************************************
 * freeing a node shadow
 ****************************************************************************************/

static
void shadow_free(NodeShadow* shadow, SddShadows* shadows) {
  assert(shadow);
  assert(shadow->ref_count);
  
  if(shadow_is_internal(shadow)) {
    ElmShadow* elements = ELMS_OF(shadow);
    for(SddNodeSize i=0; i<shadow->size; i++) {
      shadow_free(elements[i].prime,shadows);
      shadow_free(elements[i].sub,shadows);
    } 
    //elements no longer needed: convert to leaf shadow
    convert_internal_shadow_to_leaf(NULL,shadow,shadows); //will free elements
  }
  
  if(--shadow->ref_count==0) leaf_shadow_free(shadow,shadows); //shadow no longer needed
}

/****************************************************************************************
 * Shadows interface: constructing, recovering and freeing shadows
 ****************************************************************************************/

//constructing
//assumes node->shadow and node->shadow_type has been initialized for all relevant nodes
SddShadows* shadows_new(SddSize root_count, SddNode** root_nodes, SddManager* manager) {
  
  SddShadows* shadows;
  MALLOC(shadows,SddShadows,"shadows_new");
  
  shadows->manager           = manager;
  shadows->root_count        = root_count;
  shadows->root_shadows      = NULL;
  shadows->shadow_count      = 0;
  shadows->shadow_byte_count = 0;
  shadows->bit               = 0;
  
  if(root_count==0) return shadows;
  
  //construct and save root shadows
  CALLOC(shadows->root_shadows,NodeShadow*,root_count,"shadows_new");
  for(SddSize i=0; i<root_count; i++) {
    SddNode* node = root_nodes[i];
    shadows->root_shadows[i] = shadow_from_node(node,shadows);
  }
  
  assert(shadows->shadow_count);
  return shadows;
}

//recovering
//will free shadows as a side effect
void shadows_recover(SddShadows* shadows) {
  assert(shadows->shadow_count); //shadows have been saved
  
  for(SddSize i=0; i<shadows->root_count; i++) node_from_shadow(shadows->root_shadows[i],shadows);
  
  //all shadows now freed
  assert(shadows->shadow_count==0); 
  assert(shadows->shadow_byte_count==0); 
  
  free(shadows->root_shadows);  
  free(shadows);
}

//traversing
void shadows_traverse(void (*fn)(NodeShadow*,SddShadows*), SddShadows* shadows) {
  int bit = shadows->bit = !shadows->bit;
  for(SddSize i=0; i<shadows->root_count; i++) {
    shadow_traverse(bit,shadows->root_shadows[i],fn,shadows);
  }
}

//freeing
void shadows_free(SddShadows* shadows) {
  assert(shadows->shadow_count!=0 || shadows->root_count==0);
  
  for(SddSize i=0; i<shadows->root_count; i++) shadow_free(shadows->root_shadows[i],shadows);
  
  assert(shadows->shadow_count==0); 
  assert(shadows->shadow_byte_count==0);
  
  free(shadows->root_shadows);
  free(shadows);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
