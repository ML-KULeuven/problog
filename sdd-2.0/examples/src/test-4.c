#include <stdio.h>
#include <stdlib.h>
#include "sddapi.h"

int main(int argc, char** argv) {

  // set up vtree and manager
  Vtree* vtree = sdd_vtree_read("input/rotate-left.vtree");
  SddManager* manager = sdd_manager_new(vtree);

  // construct the term X_1 ^ X_2 ^ X_3 ^ X_4
  SddNode* alpha = sdd_manager_literal(1,manager);
  alpha = sdd_conjoin(alpha,sdd_manager_literal(2,manager),manager);
  alpha = sdd_conjoin(alpha,sdd_manager_literal(3,manager),manager);
  alpha = sdd_conjoin(alpha,sdd_manager_literal(4,manager),manager);

  // to perform a rotate, we need the manager's vtree
  Vtree* manager_vtree = sdd_manager_vtree(manager);
  Vtree* manager_vtree_right = sdd_vtree_right(manager_vtree);

  // obtain the manager's pointer to the root
  Vtree** root_location = sdd_vtree_location(manager_vtree,manager);

  printf("saving vtree & sdd ...\n");
  sdd_vtree_save_as_dot("output/before-rotate-vtree.dot",manager_vtree);
  sdd_save_as_dot("output/before-rotate-sdd.dot",alpha);

  // ref alpha (so it is not gc'd)
  sdd_ref(alpha,manager);
  
  // garbage collect (no dead nodes when performing vtree operations)
  printf("dead sdd nodes = %zu\n", sdd_manager_dead_count(manager));
  printf("garbage collection ...\n");
  sdd_manager_garbage_collect(manager);
  printf("dead sdd nodes = %zu\n", sdd_manager_dead_count(manager));

  printf("left rotating ... ");
  int succeeded = sdd_vtree_rotate_left(manager_vtree_right,manager,0); //not limited
  printf("%s!\n", succeeded?"succeeded":"did not succeed");

  // deref alpha, since ref's are no longer needed
  sdd_deref(alpha,manager);

  // the root changed after rotation, so get the manager's vtree again
  // this time using root_location
  manager_vtree = *root_location;
  //assert(manager_vtree==sdd_manager_vtree(manager));

  printf("saving vtree & sdd ...\n");
  sdd_vtree_save_as_dot("output/after-rotate-vtree.dot",manager_vtree);
  sdd_save_as_dot("output/after-rotate-sdd.dot",alpha);

  sdd_vtree_free(vtree);
  sdd_manager_free(manager);

  return 0;
}
