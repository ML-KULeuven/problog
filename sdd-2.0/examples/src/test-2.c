#include <stdio.h>
#include <stdlib.h>
#include "sddapi.h"

int main(int argc, char** argv) {

  // set up vtree and manager
  SddLiteral var_count = 4;
  const char* type = "right";
  Vtree* vtree = sdd_vtree_new(var_count,type);
  SddManager* manager = sdd_manager_new(vtree);

  // construct the term X_1 ^ X_2 ^ X_3 ^ X_4
  SddNode* alpha = sdd_manager_literal(1,manager);
  alpha = sdd_conjoin(alpha,sdd_manager_literal(2,manager),manager);
  alpha = sdd_conjoin(alpha,sdd_manager_literal(3,manager),manager);
  alpha = sdd_conjoin(alpha,sdd_manager_literal(4,manager),manager);

  // construct the term ~X_1 ^ X_2 ^ X_3 ^ X_4
  SddNode* beta = sdd_manager_literal(-1,manager);
  beta = sdd_conjoin(beta,sdd_manager_literal(2,manager),manager);
  beta = sdd_conjoin(beta,sdd_manager_literal(3,manager),manager);
  beta = sdd_conjoin(beta,sdd_manager_literal(4,manager),manager);

  // construct the term ~X_1 ^ ~X_2 ^ X_3 ^ X_4
  SddNode* gamma = sdd_manager_literal(-1,manager);
  gamma = sdd_conjoin(gamma,sdd_manager_literal(-2,manager),manager);
  gamma = sdd_conjoin(gamma,sdd_manager_literal(3,manager),manager);
  gamma = sdd_conjoin(gamma,sdd_manager_literal(4,manager),manager);

  printf("== before referencing:\n");
  printf("  live sdd size = %zu\n", sdd_manager_live_size(manager));
  printf("  dead sdd size = %zu\n", sdd_manager_dead_size(manager));

  // ref SDDs so that they are not garbage collected
  sdd_ref(alpha,manager);
  sdd_ref(beta,manager);
  sdd_ref(gamma,manager);
  printf("== after referencing:\n");
  printf("  live sdd size = %zu\n", sdd_manager_live_size(manager));
  printf("  dead sdd size = %zu\n", sdd_manager_dead_size(manager));

  // garbage collect
  sdd_manager_garbage_collect(manager);
  printf("== after garbage collection:\n");
  printf("  live sdd size = %zu\n", sdd_manager_live_size(manager));
  printf("  dead sdd size = %zu\n", sdd_manager_dead_size(manager));

  sdd_deref(alpha,manager);
  sdd_deref(beta,manager);
  sdd_deref(gamma,manager);

  printf("saving vtree & shared sdd ...\n");
  sdd_vtree_save_as_dot("output/shared-vtree.dot",vtree);
  sdd_shared_save_as_dot("output/shared.dot",manager);

  sdd_vtree_free(vtree);
  sdd_manager_free(manager);

  return 0;
}
