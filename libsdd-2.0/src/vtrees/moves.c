/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//declarations

//vtrees/vtree.c
void update_positions_after_swap(Vtree* vtree);

/****************************************************************************************
 * left rotation of a vtree node
 ****************************************************************************************/

//w = (a x=(b c)) ===left rotation===> x = (w=(a b) c)
//x is left rotatable iff it is not leaf and it is a right child of its parent w
int is_left_rotatable(Vtree* x) {
  return INTERNAL(x) && x->parent && x->parent->right==x;
}

//w = (a x=(b c)) ===left rotation===> x = (w=(a b) c)
//rotation preserves inorder: awbxc ==> awbxc
void rotate_vtree_left(Vtree* x, SddManager* manager) {

  assert(is_left_rotatable(x));
  assert(!FULL_DEBUG || verify_vtree_properties(manager->vtree));
    
  Vtree* w = x->parent;
  Vtree* b = x->left;
  Vtree* p = w->parent;
	
  //adjust pointers
  w->right = b;
  w->parent = x;
  b->parent = w;
  x->left = w;
  x->parent = p;
  if(p!=NULL) { if(w==p->left) p->left = x; else p->right = x; }
	
  //positions invariant
  
  //update linked list (next and prev invariant)
  x->first = w->first;
  w->last  = b->last;
 
  //update var_count
  w->var_count = w->left->var_count + w->right->var_count; //first
  x->var_count = x->left->var_count + x->right->var_count; //second
  
  //update manager root
  if(manager->vtree==w) manager->vtree = x;
   
  assert(!FULL_DEBUG || verify_vtree_properties(manager->vtree));
}


/****************************************************************************************
 * right rotation of a vtree node
 ****************************************************************************************/

//x = (w=(a b) c) ===right rotation===> w = (a x=(b c))
//x right rotatable iff it is not leaf and its left child (w) is not leaf
int is_right_rotatable(Vtree* x) {
  return INTERNAL(x) && INTERNAL(x->left);
}

//x = (w=(a b) c) ===right rotation===> w = (a x=(b c))
//rotation preserves inorder: awbxc ==> awbxc
void rotate_vtree_right(Vtree* x, SddManager* manager) {

  assert(is_right_rotatable(x));
  assert(!FULL_DEBUG || verify_vtree_properties(manager->vtree));
  
  Vtree* w = x->left;
  Vtree* b = w->right;
  Vtree* p = x->parent;
	
  //adjust pointers
  w->right = x;
  w->parent = p;
  b->parent = x;
  x->left = b;
  x->parent = w;
  if(p!=NULL) { if(x==p->left) p->left = w; else p->right = w; }

  //positions invariant
  
  //update linked list (next and prev invariant)
  x->first = b->first;
  w->last  = x->last;
  
  //update var_count
  x->var_count = x->left->var_count + x->right->var_count; //first
  w->var_count = w->left->var_count + w->right->var_count; //second
  
  //update manager root
  if(manager->vtree==x) manager->vtree = w;
  
  assert(!FULL_DEBUG || verify_vtree_properties(manager->vtree));
}

/****************************************************************************************
 * Swapping vtree children
 ****************************************************************************************/

#define LP(V) V==V->parent->left? V->parent: NULL
#define RP(V) V==V->parent->right? V->parent: NULL

//swap children of a vtree node
void swap_vtree_children(Vtree* v, SddManager* manager) {
      
  assert(!FULL_DEBUG || verify_vtree_properties(manager->vtree));
  
  //save boundaries
  Vtree* p_boundary = v->first->prev;
  Vtree* n_boundary = v->last->next;
  
  //switch children
  SWAP(Vtree*,v->left,v->right);
  
  //var counts are invariant
  
  //update linked list
  Vtree* left  = v->left;
  Vtree* right = v->right;
  //first/last: 2 links (first/last of left/right are invariants)
  v->first = left->first;
  v->last  = right->last;
  //node v: 4 links
  v->next  = right->first;
  v->prev  = left->last;
  v->next->prev = v->prev->next = v;
  //boundaries of vtree v: 4 links
  v->first->prev = p_boundary;
  v->last->next  = n_boundary;
  if(p_boundary) p_boundary->next = v->first;
  if(n_boundary) n_boundary->prev = v->last;
  //some ancestors of v
  //we only need to update ancestors that are in the same direction as v's parent
  Vtree* p = v->parent;
  if(p && v==p->left) do p->first = p->left->first; while (p->parent && (p=LP(p)));
  else if(p)          do p->last  = p->right->last; while (p->parent && (p=RP(p)));
  
  //update vtree positions
  update_positions_after_swap(v); //must be done after updating linked list
  
  assert(!FULL_DEBUG || verify_vtree_properties(manager->vtree));
}


/****************************************************************************************
 * end
 ****************************************************************************************/
