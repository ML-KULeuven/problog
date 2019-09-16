#include <stdio.h>
#include <stdlib.h>
#include "sddapi.h"

int main(int argc, char** argv) {

  // set up vtree and manager
  SddLiteral var_count = 4;
  SddLiteral var_order[4] = {2,1,4,3};
  const char* type = "balanced";

  Vtree* vtree = sdd_vtree_new_with_var_order(var_count,var_order,type);
  SddManager* manager = sdd_manager_new(vtree);

  // construct a formula (A^B)v(B^C)v(C^D)
  printf("constructing SDD ... ");
  SddNode* f_a = sdd_manager_literal(1,manager);
  SddNode* f_b = sdd_manager_literal(2,manager);
  SddNode* f_c = sdd_manager_literal(3,manager);
  SddNode* f_d = sdd_manager_literal(4,manager);

  SddNode* alpha = sdd_manager_false(manager);
  SddNode* beta;

  beta  = sdd_conjoin(f_a,f_b,manager);
  alpha = sdd_disjoin(alpha,beta,manager);
  beta  = sdd_conjoin(f_b,f_c,manager);
  alpha = sdd_disjoin(alpha,beta,manager);
  beta  = sdd_conjoin(f_c,f_d,manager);
  alpha = sdd_disjoin(alpha,beta,manager);
  printf("done\n");

  printf("saving sdd and vtree ... ");
  sdd_save_as_dot("output/sdd.dot",alpha);
  sdd_vtree_save_as_dot("output/vtree.dot",vtree);
  printf("done\n");

  sdd_vtree_free(vtree);
  sdd_manager_free(manager);

  return 0;
}
