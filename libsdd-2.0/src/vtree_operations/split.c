/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/nodes.c
void remove_from_unique_table(SddNode* node, SddManager* manager);

/****************************************************************************************
 * when rotating or swapping sdd nodes that are normalized for a given vtree, we need
 * to split these nodes into groups as each group will need to be processed differently.
 ****************************************************************************************/
 
/****************************************************************************************
 * splitting sdd nodes normalized for vtree w = (a x=(b c))
 * (in preparation for LEFT rotating x)
 *
 * split sdd nodes normalized for w as follows:
 * --nodes that depend on b and c are removed from unique table and returned in bc_list
 * --nodes that depend on c only are removed from unique table and returned in c_list
 * --nodes that depend on b only stay in the unique table 
 ****************************************************************************************/

//node is normalized for w = (a x=(b c))
//hence:
//--node must depend on a (i.e., on some variable in vtree a)
//--node must depend on either b or c as well
//
//return
//'b' if node depends on b only
//'c' if node depends on c only
//'^" if node depends on both b and c
static inline
char dependence_on_right_vtree(SddNode* node, Vtree* x) {

  int depends_on_b = 0;
  int depends_on_c = 0;
  SddLiteral x_p   = x->position;
  
  FOR_each_sub_of_node(sub,node,{
    //sub can be true, false, literal or decomposition
    if(NON_TRIVIAL(sub)) {
     SddLiteral sub_p = sub->vtree->position;
     if(sub_p==x_p) return '^';
     else if(sub_p < x_p) depends_on_b = 1;
     else depends_on_c = 1; //sub_p > x_p
     if(depends_on_b && depends_on_c) return '^';
    }
  });
  
  assert(depends_on_b || depends_on_c);
  assert(!(depends_on_b && depends_on_c));
  
  if(depends_on_b) return 'b';
  else return 'c';
  
}

//w = (a x=(b c))
//in preparation of left rotating x:
// --split nodes into bc_list, c_list and rest
// --remove nodes on bc_list and c_list from unique table, while keeping rest in the table
// --return bc_list, c_list and size of bc_list
void split_nodes_for_left_rotate(SddSize* bc_count, SddNode** bc_list, SddNode** c_list, Vtree* w, Vtree* x, SddManager* manager) {

  *bc_count = 0; *bc_list = *c_list = NULL;
  
  FOR_each_sdd_node_normalized_for(n,w,{
    char type = dependence_on_right_vtree(n,x);
    if(type!='b') { //do not remove nodes of type 'b'
      remove_from_unique_table(n,manager); //first
      //->next field is used to index n into unique table (not needed any more)
      if(type=='^') { ++(*bc_count); n->next = *bc_list; *bc_list = n; }
      else { assert(type=='c'); n->next = *c_list; *c_list = n; }
    }
  });

  sort_linked_nodes(*bc_count,bc_list,manager);
  //this seems to be more efficient as it avoids premature violation of size limits
}

/****************************************************************************************
 * splitting sdd nodes normalized for vtree x = (w=(a b) c)
 * (in preparation of RIGHT rotating x)
 *
 * split sdd nodes normalized for x as follows:
 * --nodes that depend on a and b are removed from unique table and returned in ab_list
 * --nodes that depend on a only are removed from unique table and returned in a_list
 * --nodes that depend on b only stay in the unique table 
 ****************************************************************************************/

//node is normalized for x = (w=(a b) c)
//hence:
//--node must depend on c (i.e., must depend on some variable in vtree c)
//--node must depend on either a or b
//
//return
//'a' if node depends only on a
//'b' if node depends only on b
//'^' if node depends on both a and b
//
static inline
char dependence_on_left_vtree(SddNode* node, Vtree* w) {

  int depends_on_a = 0;
  int depends_on_b = 0;
  SddLiteral w_p   = w->position;
  
  FOR_each_prime_of_node(prime,node,{
    assert(NON_TRIVIAL(prime));
    //prime could be literal or decomposition (cannot be true or false)
    SddLiteral prime_p = prime->vtree->position;
    if(prime_p == w_p) return '^';
    else if(prime_p < w_p) depends_on_a = 1;
    else depends_on_b = 1; //prime_p > w_p
    if(depends_on_a && depends_on_b) return '^';
  });
  
  assert(depends_on_a || depends_on_b);
  assert(!(depends_on_a && depends_on_b));
  
  if(depends_on_a) return 'a';
  else return 'b';
  
}

//x = (w=(a b) c)
//in preparation of right rotating x:
// --split nodes into ab_list, a_list and rest
// --remove nodes on ab_list and a_list from the unique table, while keeping rest in the able
// --return ab_list, a_lists and size of ab_list
void split_nodes_for_right_rotate(SddSize *ab_count, SddNode** ab_list, SddNode** a_list, Vtree* x, Vtree* w, SddManager* manager) {

  *ab_count = 0; *ab_list = *a_list = NULL;
  
  FOR_each_sdd_node_normalized_for(n,x,{
    char type = dependence_on_left_vtree(n,w);
    if(type!='b') { //do not remove nodes of type 'b'
      remove_from_unique_table(n,manager); //first
      //->next field is used to index n into unique table (not needed any more)
      if(type=='^') { ++(*ab_count); n->next = *ab_list; *ab_list = n; }
      else { assert(type=='a'); n->next = *a_list; *a_list = n; } 
    }
  });

  sort_linked_nodes(*ab_count,ab_list,manager);
  //this seems to be more efficient as it avoids premature violation of size limits
}

/****************************************************************************************
 * splitting sdd nodes for normalized for v
 * (in preparation of swapping v)
 ****************************************************************************************/

//remove nodes of vtree from unique table and collect them in a linked list
SddNode* split_nodes_for_swap(Vtree* v, SddManager* manager) {
  
  SddSize count = v->node_count; 
  SddNode* list = NULL;
  
  FOR_each_sdd_node_normalized_for(n,v,{
    remove_from_unique_table(n,manager); //first
    //->next field is used to index n into unique table (not needed any more)
    n->next = list;
    list  = n;
  });
  //list is now head of a linked list containing sdd nodes of vtree
  
  sort_linked_nodes(count,&list,manager);
  //this seems to be more efficient as it avoids premature violation of size limits
  
  return list;
}

/****************************************************************************************
 * END
 ****************************************************************************************/
