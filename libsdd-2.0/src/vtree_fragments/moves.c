/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

/****************************************************************************************
 * vtree move operations: 
 *
 * --try_vtree_move (with limits)
 * --make_vtree_move (without limits)
 * --reverse_vtree_move (light, without adjusting sdd nodes)
 *
 * left-rotate : always apply to the fragment's current child
 * right-rotate: always apply to fragment's current root
 * swap        : always applies to the fragment's current child
 ****************************************************************************************/

//try move in given direction WITH limits
//if move succeeds, return 1, otherwise return 0
int try_vtree_move(char move, Vtree** root, Vtree** child, SddManager* manager, int limited) {
  assert(move=='l' || move=='r' || move=='s');

  if(move=='l') {
    assert(*child==sdd_vtree_right(*root));
    if(sdd_vtree_rotate_left(*child,manager,limited)) {
      //left rotation succeeded
      SWAP(Vtree*,*root,*child); //root/child flip positions
      return 1;
    }
  }
  else if(move=='r') {
    assert(*child==sdd_vtree_left(*root));
    if(sdd_vtree_rotate_right(*root,manager,limited)) {
      //right rotation succeeded
      SWAP(Vtree*,*root,*child); //root/child flip positions
      return 1;
    }
  }
  else {
    assert(move=='s'); 
    assert(*root==sdd_vtree_parent(*child));
    if(sdd_vtree_swap(*child,manager,limited)) {
      //swap succeeded, root/child stay the same
      return 1;
    } 
  } 
  
  return 0; //move failed
}

//make move in given direction WITHOUT limits
//move will always succeed
void make_vtree_move(char move, Vtree** root, Vtree** child, SddManager* manager) {
  assert(move=='l' || move=='r' || move=='s');
  
  if(move=='l') {
    assert(*child==sdd_vtree_right(*root));
    sdd_vtree_rotate_left(*child,manager,0);
    SWAP(Vtree*,*root,*child); //root/child flip positions
  }
  else if(move=='r') {
    assert(*child==sdd_vtree_left(*root));
    sdd_vtree_rotate_right(*root,manager,0);
    SWAP(Vtree*,*root,*child); //root/child flip positions
  }
  else {
    assert(move=='s'); 
    assert(*root==sdd_vtree_parent(*child));
    sdd_vtree_swap(*child,manager,0); //root/child stay the same
  }

}
 
//reverse a vtree move without adjusting sdd nodes
void reverse_vtree_move(char move, Vtree** root, Vtree** child, SddManager* manager) {
  
  if(move=='r') {
    assert(*child==sdd_vtree_right(*root));
    rotate_vtree_left(*child,manager); //light rotation
    SWAP(Vtree*,*root,*child); //root/child flip positions
  }
  else if(move=='l') {
    assert(*child==sdd_vtree_left(*root));
    rotate_vtree_right(*root,manager); //light rotation
    SWAP(Vtree*,*root,*child); //root/child flip positions
  }
  else { //move=='s'
    assert(*root==sdd_vtree_parent(*child)); 
    swap_vtree_children(*child,manager); //light swapping
    //root/child stay the same
  }
  
}

/****************************************************************************************
 * end
 ****************************************************************************************/
