/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/shadows.c
SddNode* shadow_node(NodeShadow* shadow);
ElmShadow* shadow_elements(NodeShadow* shadow);
int shadow_is_terminal(NodeShadow* shadow);
SddShadows* shadows_new(SddSize root_count, SddNode** root_nodes, SddManager* manager);
void shadows_traverse(void (*fn)(NodeShadow*,SddShadows*), SddShadows* shadows);
void shadows_free(SddShadows* shadows);

//sdds/apply.c
SddNode* apply(SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager, int limited);

//local declarations
static void initialize_for_shadows(SddNode* node, int* exists_map, SddManager* manager);
static SddNode* quantify_shadow(NodeShadow* shadow, int* exists_map, SddManager* manager);
static void ref_nodes_of_terminal_shadows(SddShadows* shadows);
static void deref_nodes_of_terminal_shadows(SddShadows* shadows);

/****************************************************************************************
 * existentially quantify a number of variables from an sdd
 ****************************************************************************************/

//exists_map is an array with the following properties:
//size             : 1+number of variables in manager
//exists_map[var]  : is 1 if var is to be existentially quantified, 0 otherwise
//exists_map[0]    : not used

static long ref_count; //sanity check

SddNode* sdd_exists_multiple(int* exists_map, SddNode* node, SddManager* manager) {
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_exists_multiple");
  assert(!GC_NODE(node));
  
  if(node->type==FALSE || node->type==TRUE) return node;

  ref_count = 0;

  SddNode** roots = (SddNode**) malloc(sizeof(SddNode*));
  roots[0]        = node; 
  initialize_for_shadows(node,exists_map,manager); //before constructing shadows
  
  SddShadows* shadows   = shadows_new(1,roots,manager);
  NodeShadow* shadow    = shadows->root_shadows[0];
  assert(shadow);
  assert(shadow->ref_count==1);

  ref_nodes_of_terminal_shadows(shadows);
  SddNode* q_node;

  q_node = quantify_shadow(shadow,exists_map,manager);

  deref_nodes_of_terminal_shadows(shadows);
    
  shadows_free(shadows);
  free(roots);
  
  assert(ref_count==0);

  return q_node;
}

/****************************************************************************************
 * set node->shadow_type to
 * --'t', if node is a terminal sdd or does not contain quantified variables
 * --'g' otherwise
 *
 * set node->shadow to NULL for all nodes
 *
 * set node->map to
 * --NULL, if node contains quantified variables
 * --node, if node does not contain quantified variables
 *
 * node->map is only used temporarily in the initialization function
 ****************************************************************************************/

//this function will leave all bits set to 1 --- bits must be cleared (see bits.c)
static
void initialize(SddNode* node, int* exists_map, SddManager* manager) {
  if(node->bit) return; //already visited
  else node->bit = 1; //first visit
  
  node->shadow_type = 'g'; //default: may be changed later
  node->shadow      = NULL; //all nodes
  node->map         = NULL; //default: may change later
  
  if(node->type==TRUE || node->type==FALSE) { //node will not be quantified
    node->shadow_type = 't'; 
    node->map         = node;
  }
  else if(node->type==LITERAL) { //node will not be quantified
    node->shadow_type = 't';
    SddLiteral var = VAR_OF(node);
    if(exists_map[var]==0) node->map = node; //node will not be quantified
  }
  else {
    int q_vars = 0;
    FOR_each_prime_sub_of_node(prime,sub,node,{
      initialize(prime,exists_map,manager);
      initialize(sub,exists_map,manager);
      q_vars = q_vars || prime->map==NULL || sub->map==NULL;
    });
    if(q_vars==0) { //node will not be quantified
      node->shadow_type = 't'; 
      node->map         = node;
    }
  }
}

static 
void initialize_for_shadows(SddNode* node, int* exists_map, SddManager* manager) {
  initialize(node,exists_map,manager); //leave bits 1
  sdd_clear_node_bits(node);
}

/****************************************************************************************
 * ref/deref the decomposition nodes of terminal shadows
 ****************************************************************************************/

static
void ref_terminal(NodeShadow* shadow, SddShadows* shadows) {
  if(shadow_is_terminal(shadow)) {
    SddNode* node = shadow_node(shadow);
    assert(node);
    if(node->type==DECOMPOSITION) sdd_ref(node,shadows->manager);
  }
}

static
void deref_terminal(NodeShadow* shadow, SddShadows* shadows) {
  if(shadow_is_terminal(shadow)) {
    SddNode* node = shadow_node(shadow);
    assert(node);
    if(node->type==DECOMPOSITION) {
      assert(LIVE(node));
      sdd_deref(node,shadows->manager);
    }
  }
}

static
void ref_nodes_of_terminal_shadows(SddShadows* shadows) {
  shadows_traverse(ref_terminal,shadows);
}

static
void deref_nodes_of_terminal_shadows(SddShadows* shadows) {
  shadows_traverse(deref_terminal,shadows);
}

/****************************************************************************************
 * lca
 ****************************************************************************************/

//returns NULL if elements do not form an xy-decomposition
//returns lca of elements otherwise
//assumes primes form a partition (with no trivial primes)
//assumes at least one non-trivial sub
static
Vtree* lca(SddNode* non_trivial_sub, SddNodeSize size, ElmShadow* elements, SddManager* manager) {
  Vtree* pvt = elements[0].prime->cache->vtree;
  Vtree* svt = non_trivial_sub->vtree;
  assert(pvt && svt);
  if(pvt->position >= svt->position) return NULL;
  Vtree* lca = sdd_vtree_lca(pvt,svt,manager->vtree);
  for(ElmShadow* e=elements; e<elements+size; e++) {
    SddNode* prime = e->prime->cache;
    assert(prime);
    assert(prime->id== e->prime->cache_id);
    SddNode* sub   = e->sub->cache;
    assert(sub);
    assert(sub->id== e->sub->cache_id);
    assert(prime->vtree);
    if(!sdd_vtree_is_sub(prime->vtree,lca->left)) return NULL;
    if(sub->vtree && !sdd_vtree_is_sub(sub->vtree,lca->right)) return NULL;
    assert(sub->vtree==NULL || lca==sdd_vtree_lca(prime->vtree,sub->vtree,manager->vtree));
  }
  return lca;
}
  
/****************************************************************************************
 * quantifying nodes
 *
 * nodes of terminal shadows are protected (referenced) and released by the shadows
 * utility; otherwise, they may be gc'd as they lose their parents during vtree
 * minimization (even if the root sdd node is referenced before quantifying it)
 ****************************************************************************************/

//fix is_partition case

static
void deref_elements(SddNodeSize size, ElmShadow* elements, SddManager* manager) {
  for(ElmShadow* e=elements; e<elements+size; e++) {
    sdd_deref(e->prime->cache,manager);
    sdd_deref(e->sub->cache,manager);
  }
}

//checks whether the node of shadow needs quantification
static inline
int no_quantified_vars(NodeShadow* shadow) {
  assert(!shadow_is_terminal(shadow) || LIVE(shadow_node(shadow)));
  return shadow_is_terminal(shadow) && shadow_node(shadow)==shadow->cache;
}

static
SddNode* quantify_shadow(NodeShadow* shadow, int* exists_map, SddManager* manager) {
  assert(shadow->cache==NULL || shadow->cache->id==shadow->cache_id);
  if(shadow->cache && shadow->cache->id==shadow->cache_id) {
    //cached node exists and has not been gc'd
    assert(ref_count>0);
    --ref_count; 
    sdd_deref(shadow->cache,manager);
    return shadow->cache;
  }
  
  SddNode* q_node;
      
  if(shadow_is_terminal(shadow)) {
    q_node = shadow_node(shadow);
    assert(q_node);
    assert(LIVE(q_node));
    if(q_node->type==LITERAL) {
      SddLiteral var = VAR_OF(q_node);
      if(exists_map[var]) q_node = manager->true_sdd;
    }
  }
  else { 
    SddNodeSize size    = shadow->size;
    ElmShadow* elements = shadow_elements(shadow);
   
    //quantify shadow elements and determine type of decomposition
    int is_true_element      = 0;
    int is_true_subs         = 1;
    int is_same_primes       = 1;
    SddNode* non_trivial_sub = NULL;
    for(ElmShadow* e=elements; e<elements+size; e++) {
      SddNode* prime = quantify_shadow(e->prime,exists_map,manager);
      assert(prime==e->prime->cache);
      sdd_ref(prime,manager);
      SddNode* sub   = quantify_shadow(e->sub,exists_map,manager);
      assert(prime==e->prime->cache);
      sdd_ref(sub,manager);
      assert(LIVE(prime) && LIVE(sub));
      is_true_element |= IS_TRUE(prime) && IS_TRUE(sub);
      is_true_subs    &= IS_TRUE(sub);
      is_same_primes  &= no_quantified_vars(e->prime);
      if(non_trivial_sub==NULL && !TRIVIAL(sub)) non_trivial_sub = sub;
    }  
   
    Vtree* vtree = NULL;
    
    if(is_true_element || is_true_subs) {
      deref_elements(size,elements,manager);
      q_node = manager->true_sdd; 
    }
    else if(non_trivial_sub==NULL && size==2) { //must come after is_true_subs
      deref_elements(size,elements,manager);
      SddNode* sub0 = elements[0].sub->cache;
      q_node = IS_TRUE(sub0)? elements[0].prime->cache: elements[1].prime->cache;
    } 
    
    else if(is_same_primes && non_trivial_sub && (vtree=lca(non_trivial_sub,size,elements,manager))) {
      //partition but may not be compressed
      deref_elements(size,elements,manager);
      GET_node_from_partition(q_node,vtree,manager,
        for(ElmShadow* e=elements; e<elements+size; e++) {
          SddNode* q_prime = e->prime->cache;
          SddNode* q_sub   = e->sub->cache;
          assert(q_prime->vtree && sdd_vtree_is_sub(q_prime->vtree,vtree->left));
          assert(q_sub->vtree==NULL || sdd_vtree_is_sub(q_sub->vtree,vtree->right));
          DECLARE_element(q_prime,q_sub,vtree,manager);
        }
      );
    } 
    
    else { //must construct quantified node through apply
      q_node = manager->false_sdd;
      for(ElmShadow* e=elements; e<elements+size; e++) {
        SddNode* xq_node = e->prime->cache;
        SddNode* yq_node = e->sub->cache;
        assert(LIVE(xq_node) && LIVE(yq_node));
        sdd_deref(xq_node,manager); //release
        sdd_deref(yq_node,manager); //release
        sdd_ref(q_node,manager); //protect
        SddNode* q_element = apply(xq_node,yq_node,CONJOIN,manager,0);
        sdd_deref(q_node,manager); //release
	    q_node = apply(q_element,q_node,DISJOIN,manager,0); 
      }
    } 
  }
  
  //cached nodes are not protected so they may be gc'd
  //the cache_id field will be used to decide whether thet cached node was gc'd
  shadow->cache_id = q_node->id;
  assert(ref_count>=0);
  for(SddRefCount i=0; i<shadow->ref_count-1; i++) { ++ref_count; sdd_ref(q_node,manager); }
      
  return shadow->cache = q_node;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
