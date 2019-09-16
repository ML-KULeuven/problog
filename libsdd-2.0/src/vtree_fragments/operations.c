/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/gc.c
void garbage_collect_in(Vtree* vtree, SddManager* manager);

//basic/memory.c
void gc_sdd_node(SddNode* node, SddManager* manager);

//basic/nodes.c
void remove_from_unique_table(SddNode* node, SddManager* manager);
void insert_in_unique_table(SddNode* node, SddManager* manager);

//basic/shadows.c
void shadows_recover(SddShadows* shadows);

//vtree_fragments/construction.c
void count_internal_parents_of_child_nodes(Vtree* root, Vtree* child);
void construct_fragment_shadows(VtreeFragment* fragment);
void free_fragment_shadows(VtreeFragment* fragment);

//vtree_fragments/moves.c
int try_vtree_move(char move, Vtree** root, Vtree** child, SddManager* manager, int limited);
void make_vtree_move(char move, Vtree** root, Vtree** child, SddManager* manager);
void reverse_vtree_move(char move, Vtree** root, Vtree** child, SddManager* manager);

//local declaration
static void recover_fragment_shadows(VtreeFragment* fragment);

/****************************************************************************************
 * Fragment operations: next, goto and rewind
 *
 * A fragment can be in one of three modes: (i: initial) (n: next) (g: goto)
 *
 * A fragment is started in the initial mode.
 *
 * Operations are only applicable in certain modes:
 *
 * --the next operation can only be applied in the initial or next modes
 * --the goto operation can only be applied in the initial or goto modes 
 * --the rewind operation can only by applied in the initial or next modes
 *
 * Operations change the fragment mode:
 *
 * --next puts the fragment in the next mode
 * --goto puts the fragment in the goto mode
 * --rewind puts the fragment in the initial mode
 *
 * Finally, shadows are only present in next mode (not in initial or goto modes)
 ****************************************************************************************/

int vtree_fragment_state(VtreeFragment* fragment) {
  return fragment->state;
}

Vtree* vtree_fragment_root(VtreeFragment* fragment) {
  return fragment->cur_root;
}

//checks the validity of a fragment initial state
int valid_fragment_initial_state(VtreeFragment* fragment) {
    
  char type    = fragment->type;
  Vtree* root  = fragment->root;
  Vtree* child = fragment->child;
    
  return fragment->state==0 &&
         fragment->mode=='i' &&
         root==fragment->cur_root &&
         child==fragment->cur_child &&
         ((type=='r' && child==root->right) || (type=='l' && child==root->left));
}

//check whether fragment is in its initial state
int vtree_fragment_is_initial(VtreeFragment* fragment) {
   
  if(fragment->state==0) { //initial state
    assert(valid_fragment_initial_state(fragment));
    return 1;
  }
  else {
    assert((0 < fragment->state) && (fragment->state <= 11));
    assert(fragment->mode=='n' || fragment->mode=='g');
    return 0; 
  }
}

/****************************************************************************************
 * utilities
 ****************************************************************************************/
 
//returns the new state
static
int update_state(char direction, VtreeFragment* fragment) {
  if(direction=='f') { //next state
    if(++fragment->state==12) fragment->state = 0;
  }
  else { //previous state
    assert(direction=='b');
    if(--fragment->state==-1) fragment->state = 11;
  }
  assert(0 <= fragment->state && fragment->state <= 11);
  return fragment->state; //the new state
}

//get move that will take us to a neighboring state
static
char get_move_to_neighbor(char direction, VtreeFragment* fragment) {
  assert(direction=='f' || direction=='b');
  int state = fragment->state;
  
  if(direction=='f') return fragment->moves[state];
  
  int prev_state = (state==0? 11: state-1);
  char prev_move = fragment->moves[prev_state];
  
  //reverse it
  if(prev_move=='l') return 'r';
  else if(prev_move=='r') return 'l';
  else return 's';
}

/****************************************************************************************
 * Moving a fragment to a neighboring state (with limits).
 ****************************************************************************************/

//moves the fragment to a neighboring state (WITH limits)
//returns 1 if move successful; returns 0 otherwise
//if move is not successful, fragment stays at its current state
int vtree_fragment_next(char direction, VtreeFragment* fragment, int limited) {
  assert(0 <= fragment->state && fragment->state <= 11);
  assert(direction=='f' || direction=='b');
  
  //cannot be in goto mode
  CHECK_ERROR(fragment->mode=='g',ERR_MSG_FRG_N,"vtree_fragment_next");

  //shadows are needed to rewind
  if(fragment->mode=='i') construct_fragment_shadows(fragment);
       
  char move = get_move_to_neighbor(direction,fragment);
  int status;
   
  if(try_vtree_move(move,&fragment->cur_root,&fragment->cur_child,fragment->manager,limited)) {
    update_state(direction,fragment);
    status = 1; //success
  }
  else status = 0; //failure

  if(fragment->state==0) { //at initial state
    fragment->mode  = 'i'; //initial mode
    free_fragment_shadows(fragment); //shadows no longer needed
    assert(valid_fragment_initial_state(fragment));
  } 
  else fragment->mode = 'n'; //now in next mode

  return status;
}

/****************************************************************************************
 * Moving a fragment to a particular state.
 ****************************************************************************************/
 
//moves a fragment from the current state to the target state (WITHOUT limits)
//no shortcuts are used
//returns the fragment root at the target state
Vtree* vtree_fragment_goto(int state, char direction, VtreeFragment* fragment) {
  assert(0 <= state && state <= 11);
  assert(direction=='f' || direction=='b');

  //cannot be in next mode
  CHECK_ERROR(fragment->mode=='n',ERR_MSG_FRG_G,"vtree_fragment_goto");
    
  while(fragment->state != state) {
    char move = get_move_to_neighbor(direction,fragment);
    make_vtree_move(move,&fragment->cur_root,&fragment->cur_child,fragment->manager);
    update_state(direction,fragment);
  }
 
  fragment->mode = (fragment->state==0? 'i': 'g');

  return fragment->cur_root;
}
 
/****************************************************************************************
 * Rewinding a fragment to its initial state (without using vtree operations).
 ****************************************************************************************/

//moves the fragment to its initial state
//returns the fragment root at its initial state
Vtree* vtree_fragment_rewind(VtreeFragment* fragment) {
  assert(0 <= fragment->state && fragment->state <= 11);

  if(fragment->mode=='i') { //already in initial state
    assert(valid_fragment_initial_state(fragment));
    return fragment->root; 
  }
  
  //cannot be in goto mode
  CHECK_ERROR(fragment->mode=='g',ERR_MSG_FRG_R,"vtree_fragment_rewind");

  recover_fragment_shadows(fragment);
  fragment->mode = 'i'; //now in initial mode  
      
  assert(valid_fragment_initial_state(fragment));

  return fragment->root;  
}

/****************************************************************************************
 * Recovering shadows to support fragment rewinding.
 ****************************************************************************************/

//classification of sdd nodes normalized for the child of a fragment (IC and Ic are used in construction.c)
//n in IC iff n->index==0 (no parents at root)
//n in Ic iff n->index>0 (parents at root) and n->ref_count>n->index (external references)
//n in Id iff n->index>0 (parents at root) and n->ref_count==n->index (no external references)
//these macros should be called only after calling count_internal_parents_of_child_nodes()
#define IN_Id(N) (N->index>0 && N->ref_count==N->index)

//recover
void recover_fragment_shadows(VtreeFragment* fragment) {
  assert(fragment->shadows); //shadows have been constructed
  
  SddManager* manager = fragment->manager;
 
  //save current root and child
  Vtree* prev_root  = fragment->cur_root;
  Vtree* prev_child = fragment->cur_child;
  Vtree* prev_child_left  = prev_child->left;
  Vtree* prev_child_right = prev_child->right;
  
  //bring vtree back to its original state (without adjusting sdd nodes)
  while(fragment->state > 0) {
    char move = fragment->moves[--fragment->state];
    reverse_vtree_move(move,&fragment->cur_root,&fragment->cur_child,fragment->manager);
  }
  assert(fragment->state==0);
  assert(fragment->root==fragment->cur_root && fragment->child==fragment->cur_child);
  
  //IR+IC+Ic nodes must appear at prev_root and prev_child (all have external references)
  //nodes at prev_root all belong to IR+IC+Ic
  //nodes at prev_child belong to one of two groups:
  //--those that have external references (all must belong to IR+IC+Ic)
  //--those have have no external references (called Id nodes, all of which must have parents at prev_root)
  
  //except for Id nodes, all nodes at prev_root and prev_child will be updated when recovering shadows
  //(that is, their elements and vtrees will be updated to match the fragment's initial vtree)
  //as for Id nodes:
  //--they are valid in the fragment's initial vtree if cur_child and prev_child are the same
  //--they are invalid otherwise
  //if Id nodes are valid, we just make sure they are at the correct vtree node
  //if Id nodes are invalid, they must be removed from cur_child and gc'd after recovering shadows 
  
  SddNode* invalid_Id_nodes = NULL;
  Vtree* cur_child = fragment->cur_child;
  if(prev_child!=cur_child || prev_child_left!=cur_child->left || prev_child_right!=cur_child->right) {
    //Id nodes are invalid or at the wrong vtree node
    count_internal_parents_of_child_nodes(prev_root,prev_child);
    SddNode* Id_nodes = NULL;
    FOR_each_sdd_node_normalized_for(node,prev_child,
      assert(node->ref_count!=node->index || node->index>0); //no external refs implies parents at prev_root
      if(IN_Id(node)) {
        remove_from_unique_table(node,manager);
        node->next = Id_nodes;
        Id_nodes   = node;
      }
    );
    if(prev_child_left==cur_child->left && prev_child_right==cur_child->right) {
      assert(prev_child!=cur_child);
      //Id nodes are valid but at the wrong vtree node
      FOR_each_linked_node(node,Id_nodes,{
        assert(node->vtree!=cur_child);
        node->vtree = cur_child;
        insert_in_unique_table(node,manager);
      });
    }
    else invalid_Id_nodes = Id_nodes; //Id nodes are invalid
  }
   
  //recover sdd-DAG from shadow-DAG
  shadows_recover(fragment->shadows);
  fragment->shadows = NULL; //shadows are free after recovery

  //invalid_id_nodes may be NULL
  FOR_each_linked_node(node,invalid_Id_nodes,assert(DEAD(node));gc_sdd_node(node,manager));
  
  //no dead nodes initially above fragment, and no dead nodes will be created 
  //above fragment; hence, we only need to gc inside fragment 
  garbage_collect_in(fragment->cur_root,manager);
  
  assert(!FULL_DEBUG || verify_gc(fragment->cur_root,fragment->manager));
  assert(!FULL_DEBUG || verify_counts_and_sizes(fragment->manager));
}
    
/****************************************************************************************
 * end
 ****************************************************************************************/
