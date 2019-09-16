#include <stdio.h>
#include <stdlib.h>
#include "sddapi.h"

int main(int argc, char** argv) {

  // set up vtree and manager
  Vtree* vtree = sdd_vtree_read("input/opt-swap.vtree");
  SddManager* manager = sdd_manager_new(vtree);

  printf("reading sdd from file ...\n");
  SddNode* alpha = sdd_read("input/opt-swap.sdd",manager);
  printf("  sdd size = %zu\n", sdd_size(alpha));

  // ref, perform the minimization, and then de-ref
  sdd_ref(alpha,manager);
  printf("minimizing sdd size ... ");
  sdd_manager_minimize(manager); // see also sdd_manager_minimize_limited()
  printf("done!\n");
  printf("  sdd size = %zu\n", sdd_size(alpha));
  sdd_deref(alpha,manager);

  // augment the SDD
  printf("augmenting sdd ...\n");
  SddNode* beta = sdd_disjoin(sdd_manager_literal(4,manager),
                              sdd_manager_literal(5,manager),manager);
  beta = sdd_conjoin(alpha,beta,manager);
  printf("  sdd size = %zu\n", sdd_size(beta));

  // ref, perform the minimization again on new SDD, and then de-ref
  sdd_ref(beta,manager);
  printf("minimizing sdd ... ");
  sdd_manager_minimize(manager);
  printf("done!\n");
  printf("  sdd size = %zu\n", sdd_size(beta));
  sdd_deref(beta,manager);

  sdd_manager_free(manager);
  sdd_vtree_free(vtree);

  return 0;
}
