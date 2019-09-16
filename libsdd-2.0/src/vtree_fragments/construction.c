/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/shadows.c
SddShadows* shadows_new(SddSize root_count, SddNode** root_nodes, SddManager* manager);
void shadows_free(SddShadows* shadows);

//vtree_fragments/operations.c
int valid_fragment_initial_state(VtreeFragment* fragment);

//local declarations
void initialize_sdd_dag(SddSize root_count, SddNode** root_nodes, SddSize changeable_count, SddNode** changeable_nodes);

/****************************************************************************************
 * A "fragment" is a pair of vtree nodes (x y), where 
 *  
 *   --x is the parent of y, and
 *   --y is not a leaf vtree node
 *
 * A fragment is either:
 *
 *   --left-linear,  having the form x=(y=(a b) c)
 *   --right-linear, having the form x=(a y=(b c))
 *
 * vtree nodes (x y) are called the fragment's internal nodes, while
 * vtree nodes (a b c) are called the fragment's leaves (not necessarily vtree leaves)
 *
 * For internal nodes (x y) and leave nodes (a b c), there are 12 fragments states,
 * which correspond to the 2 permutations of (x y) and 6 permutations of (a b c):
 *
 *  --6 right-linear: 
 *  (a (b c)), (a (c b)), (b (a c)), (b (c a)), (c (a b)), (c (b a))
 *  --6 left-linear : 
 *  ((a b) c), ((a c) b), ((b a) c), ((b c) a), ((c a) b), ((c b) a)
 *
 * One can cycle through these fragment states using only three vtree operations:
 * left-rotate child, right-rotate root, swap child.
 *
 *
 * A number of utilities are provided for cycling through fragment states:
 *
 *  -- vtree_fragment_new(x,y): 
 *       constructs a fragment structure with internal nodes (x y) (at initial state)
 *
 *  -- vtree_fragment_free(fragment): 
 *       frees a fragment structure
 *
 *  -- vtree_fragment_forward(fragment):
 *       tries to advance the fragment to its next state using limited vtree operations
 *       (may fail due to time/space limits)
 *
 *  -- vtree_fragment_rewind(fragment):
 *       brings a fragment back to its initial state
 *
 *  -- vtree_fragment_goto(target-state,fragment):
 *       moves a fragment from its initial state to target-state
 *
 * The typical use of the above functions is as follows: 
 *
 *  --one creates a fragment using the _new function, then tries to cycle through 
 *  its states using the _forward function
 *
 *  --if cycling through all 12 states succeeds, then one can move to any of these states
 *  using the _goto function
 *
 *  --if cycling fails (due to time/space limits), one goes back to the initial state
 *  using the _rewind function
 *
 *  --one finally dispenses of the fragment structure using the _free function
 ****************************************************************************************/

//vtree moves which allow one to cycle through all 12 states of a fragment
//left-rotation applies to child
//right-rotation applies to root
//swapping applies to child
 
//starting from a left-linear state
static char moves_ll[12] = {'r','s','l','s',   'r','s','l','s',   'r','s','l','s'}; 
//the following order of navigation is much worse
//static char moves_ll[12] = {'s','r','s','l',   's','r','s','l',   's','r','s','l'}; 

//starting from a right-linear state
static char moves_rl[12] = {'l','s','r','s',   'l','s','r','s',   'l','s','r','s'};
//the following order of navigation is much worse
//static char moves_rl[12] = {'s','l','s','r',   's','l','s','r',   's','l','s','r'};

//note: the 6 permutations of leaves (a b c) are generated after moves 1, 3, 5, 7, 9, 11

/****************************************************************************************
 * Consider a fragment for root and child, in its initial state.
 *
 * The fragment defines an sdd DAG (also called the fragment DAG) whose roots are:
 *
 * --IR: sdd nodes normalized for root
 * --IC: sdd nodes normalized for child and having no parents at root
 *
 * An sdd node that belongs to the fragment DAG is said to be INSIDE the fragment, or
 * a fragment node. Otherwise, the node is said to be OUTSIDE the fragment.
 *
 * When a fragment is constructed, there are no dead nodes inside or above root.
 * Hence, all fragment nodes must be live.
 *
 * Moreover, if n is a fragment node, then
 *
 * --node m is a FRAGMENT PARENT of n iff m is a parent of n and m is a fragment node.
 * --node n has an EXTERNAL reference iff its reference count (n->ref_count) is greater
 *   than the number of its fragment parents.
 *
 * Note that nodes IR+IC must all have external references. Moreover, a fragment
 * structure saves IR+IC nodes in an array (fragment->root_nodes).
 *
 * Another category of fragments nodes is:
 *
 * --Ic: sdd nodes normalized for child, with parents at root and external references.
 *
 * Nodes Ic are also saved in the fragment structure (fragment->
 *
 * Nodes IR+IC+Ic cover all nodes at root and child, except for nodes at child that have
 * have no external references (these must have parents at root since they are live).
 *
 * IR+IC are important because they define the fragment DAG. Moreover, node IR+IC+Ic are
 * important because they constitute all nodes at root and child that also have 
 * external references (these nodes must preserve their structures).
 ****************************************************************************************/

/****************************************************************************************
 * freeing a fragment
 ****************************************************************************************/

void vtree_fragment_free(VtreeFragment* fragment) {
  if(fragment->shadows) shadows_free(fragment->shadows); //shadows may have not been constructed
  free(fragment->IR_IC_nodes);
  free(fragment->Ic_nodes);
  free(fragment);
}

/****************************************************************************************
 * Identifying nodes IC and Ic
 ****************************************************************************************/

//for each child node n, set n->index to the number of parents that n has at root
void count_internal_parents_of_child_nodes(Vtree* root, Vtree* child) {
  FOR_each_sdd_node_normalized_for(node,child,node->index=0);
  FOR_each_sdd_node_normalized_for(node,root,
    FOR_each_prime_sub_of_node(prime,sub,node,++prime->index;++sub->index));
  //for child node n, n->index is now the number of its parents at root
}

//classification of sdd nodes normalized for the child of a fragment (Id are used in operations.c)
//n in IC iff n->index==0 (no parents at root)
//n in Ic iff n->index>0 (parents at root) and n->ref_count>n->index (external references)
//n in Id iff n->index>0 (parents at root) and n->ref_count==n->index (no external references)
//these macros should be called only after calling count_internal_parents_of_child_nodes()
#define IN_IC(N) (N->index==0)
#define IN_Ic(N) (N->index>0 && N->ref_count>N->index)

/****************************************************************************************
 * constructing a fragment
 ****************************************************************************************/

//type 'r': right-linear fragment
//type 'l': left-linear fragment
VtreeFragment* vtree_fragment_new(Vtree* root, Vtree* child, SddManager* manager) {
  
  VtreeFragment* fragment;
  MALLOC(fragment,VtreeFragment,"vtree_fragment_new");
        
  fragment->manager     = manager;
  fragment->type        = (child==root->right? 'r': 'l');
  fragment->root        = root;
  fragment->child       = child;
  fragment->moves       = (fragment->type=='r'? moves_rl: moves_ll);
  fragment->shadows     = NULL;
  
  fragment->state             = 0;     //state
  fragment->mode              = 'i';   //state
  fragment->cur_root          = root;  //state
  fragment->cur_child         = child; //state

  //counting IC and Ic nodes
  count_internal_parents_of_child_nodes(root,child);
  SddSize IC_count = 0; //child nodes with no parents at root
  SddSize Ic_count = 0; //child nodes with parents at root and external references
  FOR_each_sdd_node_normalized_for(node,child,{
    if(IN_IC(node))      ++IC_count;
    else if(IN_Ic(node)) ++Ic_count;
  });
  
  fragment->IR_IC_nodes = NULL;
  fragment->Ic_nodes    = NULL;
  fragment->IR_IC_count = root->node_count + IC_count;
  fragment->Ic_count    = Ic_count;
  assert(fragment->IR_IC_count!=0 || fragment->Ic_count==0);
  
  if(fragment->IR_IC_count==0) return fragment;
  
  //allocate array to hold roots of the fragment DAG
  CALLOC(fragment->IR_IC_nodes,SddNode*,fragment->IR_IC_count,"vtree_fragment_new");
  //allocate array to hold Ic nodes
  CALLOC(fragment->Ic_nodes,SddNode*,fragment->Ic_count,"vtree_fragment_new");
  
  //save IR nodes
  FOR_each_sdd_node_normalized_for(node,root,*fragment->IR_IC_nodes++ = node);
  //save IC and Ic nodes
  FOR_each_sdd_node_normalized_for(node,child,{
    if(IN_IC(node))      *fragment->IR_IC_nodes++ = node;
    else if(IN_Ic(node)) *fragment->Ic_nodes++    = node;
  });
  fragment->IR_IC_nodes -= fragment->IR_IC_count;
  fragment->Ic_nodes    -= fragment->Ic_count;

  assert(valid_fragment_initial_state(fragment));
  
  return fragment;
}

/****************************************************************************************
 * constructing and freeing fragment shadows
 ****************************************************************************************/
 
 //construct
void construct_fragment_shadows(VtreeFragment* fragment) {
  assert(valid_fragment_initial_state(fragment));

  SddManager* manager   = fragment->manager;
  SddSize IR_IC_count   = fragment->IR_IC_count;
  SddNode** IR_IC_nodes = fragment->IR_IC_nodes;
  SddSize Ic_count      = fragment->Ic_count;
  SddNode** Ic_nodes    = fragment->Ic_nodes;
  
  //set node->shadow_types and initialize node->shadow=NULL    
  initialize_sdd_dag(IR_IC_count,IR_IC_nodes,Ic_count,Ic_nodes);
  
  SddShadows* shadows = fragment->shadows 
                      = shadows_new(IR_IC_count,IR_IC_nodes,manager);
 
  //keeping track of shadows stats
  manager->max_fragment_shadow_count = MAX(shadows->shadow_count,manager->max_fragment_shadow_count);
  manager->max_fragment_shadow_byte_count = MAX(shadows->shadow_byte_count,manager->max_fragment_shadow_byte_count);
}

//free
void free_fragment_shadows(VtreeFragment* fragment) {
  assert(fragment->shadows); //shadows have been constructed
  shadows_free(fragment->shadows);
  fragment->shadows = NULL;
}

/****************************************************************************************
 * initializing nodes in the sdd-DAG and setting their shadow types
 *
 * A node n in the sdd DAG has an external reference iff 
 *
 *     n->ref_count > pcount
 *
 * where pcount is the number of internal parents of n (i.e., parents in the sdd DAG). 
 ****************************************************************************************/

//will visit all nodes in the sdd-DAG EXCEPT its roots
static
void initialize(SddNode* node) {
  if(node->bit) ++node->index; //node visited before
  else { //first visit to node
    node->bit         = 1;
    node->index       = 1;
    node->shadow      = NULL;
    node->shadow_type = '?';
    if(node->type==DECOMPOSITION) {
      FOR_each_prime_sub_of_node(prime,sub,node,initialize(prime);initialize(sub));
    }
  }
}

//will visit all nodes in the sdd-DAG EXCEPT its roots
static
void set_shadow_types(SddNode* node, int parent_is_terminal) {
  assert(LIVE(node)); //node is live
  assert(node->index); //node has internal parents
  if(node->shadow_type=='?' || parent_is_terminal) {
    //if node->shadow_type=='?', then first visit to node, node->index is number of internal parents,
    //and node->ref_count>node->index iff node has external references
    int terminal = node->type!=DECOMPOSITION || parent_is_terminal || node->ref_count>node->index;
    if(terminal) node->shadow_type = 't'; //node will not be gc'd or its elements changed
    else         node->shadow_type = 'g'; //node may be gc'd
  }
  if(--node->index==0) { //last visit to node (type is now finalized)
    node->bit = 0; //bits must be cleared
    if(node->type==DECOMPOSITION) {
      parent_is_terminal = node->shadow_type=='t';
      FOR_each_prime_sub_of_node(prime,sub,node,{
        set_shadow_types(prime,parent_is_terminal);
        set_shadow_types(sub,parent_is_terminal);
      });
    }
  }
}

void initialize_sdd_dag(SddSize root_count, SddNode** root_nodes, SddSize changeable_count, SddNode** changeable_nodes) {
  //first pass: initialize shadows, their types, and count internal parents
  for(SddSize i=0; i<root_count; i++) {
    SddNode* node = root_nodes[i];
    node->index  = 0;
    node->shadow = NULL;
    FOR_each_prime_sub_of_node(prime,sub,node,initialize(prime);initialize(sub));
  }
  
  for(SddSize i=0; i<changeable_count; i++) {
    SddNode* node = changeable_nodes[i]; 
    assert(node->ref_count && node->index); //node is live and non-root
    node->shadow_type = 'c'; //node will not be gc'd but its elements may be changed
  }
  
  //second pass: set shadow types and clear bits
  for(SddSize i=0; i<root_count; i++) {
    SddNode* node = root_nodes[i]; 
    assert(node->ref_count && node->index==0); //node is live and root
    node->shadow_type = 'c'; //node will not be gc'd but its elements may be changed
    FOR_each_prime_sub_of_node(prime,sub,node,set_shadow_types(prime,0);set_shadow_types(sub,0));
  }
}

/****************************************************************************************
 * end
 ****************************************************************************************/
