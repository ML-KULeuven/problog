/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//local declarations
char cmp_vtrees(Vtree** lca, Vtree* vtree1, Vtree* vtree2);

/****************************************************************************************
 * basic tests
 ****************************************************************************************/

//tests whether vtree1 is a subtree of vtree2
inline
int sdd_vtree_is_sub(const Vtree* vtree1, const Vtree* vtree2) {
  return vtree1->position >= vtree2->first->position && vtree1->position <= vtree2->last->position;
}

/****************************************************************************************
 * lca is the lowest common ancestor
 ****************************************************************************************/

//returns the lca of vtree1 and vtree2 assuming both are sub_vtrees of root
Vtree* sdd_vtree_lca(Vtree* vtree1, Vtree* vtree2, Vtree* root) {
  
  //optimization
  if(vtree1==vtree2) return vtree1;
  else if(vtree1->parent==vtree2->parent) return vtree1->parent; 
  
  SddLiteral p1 = vtree1->position;
  SddLiteral p2 = vtree2->position;
  
  while(1) {
    SddLiteral p  = root->position;
    if (p1 < p && p2 < p)      root = root->left;
    else if (p1 > p && p2 > p) root = root->right;
    else return root;
  }
  
}

//returns the least common ancestor of a set of variables in a vtree
//count is the number of variables
//variables: an array of size count containing the variables
Vtree* sdd_manager_lca_of_literals(int count, SddLiteral* literals, SddManager* manager) {
  assert(count>0);
  Vtree* root  = manager->vtree;
  Vtree* vtree = sdd_manager_vtree_of_var(labs(literals[0]),manager); 
  for(int i=1; i<count; i++) {
    Vtree* leaf = sdd_manager_vtree_of_var(labs(literals[i]),manager); 
    vtree       = sdd_vtree_lca(vtree,leaf,root);
  }
  return vtree;
}

//returns the least common ancestors of a set of elements
//assumes at least two elements, and no trivial primes
Vtree* lca_of_compressed_elements(SddNodeSize size, SddElement* elements, SddManager* manager) {
  assert(size>=2);
 
  Vtree* root  = manager->vtree;
  Vtree* l_lca = NULL;
  Vtree* r_lca = NULL;
 
  for(SddElement* e=elements; e<elements+size; e++) {
    Vtree* p_vtree = e->prime->vtree;
    Vtree* s_vtree = e->sub->vtree;
    assert(p_vtree); //prime cannot be trivial
   
    l_lca = (l_lca? sdd_vtree_lca(p_vtree,l_lca,root): p_vtree);
    
    if(s_vtree && r_lca) r_lca = sdd_vtree_lca(s_vtree,r_lca,root);
    else if(s_vtree) r_lca = s_vtree;
  }
  
  assert(l_lca && r_lca);
  assert(l_lca->position < r_lca->position);
  
  Vtree* lca;
  char c = cmp_vtrees(&lca,l_lca,r_lca);
  
  assert(c=='i');
  assert(lca);
  
  c ='i'; //to suppress compiler warning
  return lca;
}


/****************************************************************************************
 * compare vtrees (assumes vtree1 <= vtree2 in the in-order)
 *
 * returns one of
 *  'e': vtree1 and vtree2 are equal
 *  'l': vtree1 is a sub_vtree of vtree2 (and, hence, of vtree2->left)
 *  'r': vtree2 is a sub_vtree of vtree1 (and, hence, of vtree1->right)
 *  'i': vtrees are incomparable
 *
 * sets vtree to lowest common ancestor of vtree1 and vtree2
 ****************************************************************************************/

//assumes vtree1 <= vtree2 in the in-order
char cmp_vtrees(Vtree** lca, Vtree* vtree1, Vtree* vtree2) {
  assert(vtree1->position <= vtree2->position);
  
  if(vtree1==vtree2) { //equal vtrees
    *lca = vtree1;
    return 'e'; 
  }
  else if(vtree1->position >= vtree2->first->position) { //vtree1 is a sub_vtree of vtree2->left
    *lca = vtree2;
    return 'l';
  }
  else if(vtree2->position <= vtree1->last->position) { //vtree2 is a sub_vtree of vtree1->right
    *lca = vtree1;
    return 'r';
  }
  else { //neither vtree is a sub_vtree of the other
    *lca = vtree1->parent;
    while(vtree2->position > (*lca)->last->position) {
      //vtree2 not subtree of (*lca)->right
      *lca = (*lca)->parent;
    }
    return 'i';
  }
  
}

/****************************************************************************************
 * end
 ****************************************************************************************/
