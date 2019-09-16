/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sddapi.h"
#include "compiler.h"

/****************************************************************************************
 * a key concept in this file is the "litset lca":
 *  --a litset is a set of literals, representing a clause or a term
 *  --the "litset lca" is the lowest vtree node that contains all variables of a litset
 *
 * the litset lca is unique and is stored in the vtree field of a litset
 *
 * this file contains a function for sorting an array of litsets according to their lca 
 ****************************************************************************************/

/****************************************************************************************
 * sort litsets according to their lcas
 ****************************************************************************************/

int litset_cmp_lca(const void* litset1_loc, const void* litset2_loc) {

  LitSet* litset1 = *(LitSet**)litset1_loc;
  LitSet* litset2 = *(LitSet**)litset2_loc;

  Vtree* vtree1 = litset1->vtree;
  Vtree* vtree2 = litset2->vtree;
  SddLiteral p1 = sdd_vtree_position(vtree1);
  SddLiteral p2 = sdd_vtree_position(vtree2);
  
  if(vtree1!=vtree2 && (sdd_vtree_is_sub(vtree2,vtree1) || (!sdd_vtree_is_sub(vtree1,vtree2) && (p1 > p2)))) return 1;
  else if(vtree1!=vtree2 && (sdd_vtree_is_sub(vtree1,vtree2) || (!sdd_vtree_is_sub(vtree2,vtree1) && (p1 < p2)))) return -1;
  else {
	
  SddLiteral l1 = litset1->literal_count;
  SddLiteral l2 = litset2->literal_count;
  
  if(l1 > l2) return 1;
  else if(l1 < l2) return -1;
  else { 
    //so the litset order is unique
  	//without this, final litset order may depend on system
    SddSize id1 = litset1->id;
    SddSize id2 = litset2->id;
    if(id1 > id2) return 1;
    else if(id1 < id2) return -1;
    else return 0;
  }
  }
}

//first: incomparable lcas are left to right, comparabale lcas are top to down
//then: shorter to larger litsets
//last: by id to obtain unique order
void sort_litsets_by_lca(LitSet** litsets, SddSize size, SddManager* manager) {
  //compute lcas of litsets
  for(SddLiteral i=0; i<size; i++) {
    LitSet* litset = litsets[i];
    litset->vtree  = sdd_manager_lca_of_literals(litset->literal_count,litset->literals,manager);
  }
  //sort
  qsort((LitSet**)litsets,size,sizeof(LitSet*),litset_cmp_lca);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
