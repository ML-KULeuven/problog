#include <stdio.h>
#include <stdlib.h>
#include "sddapi.h"

// returns an SDD node representing ( node1 => node2 )
SddNode* sdd_imply(SddNode* node1, SddNode* node2, SddManager* manager) {
  return sdd_disjoin(sdd_negate(node1,manager),node2,manager);
}

// returns an SDD node representing ( node1 <=> node2 )
SddNode* sdd_equiv(SddNode* node1, SddNode* node2, SddManager* manager) {
  return sdd_conjoin(sdd_imply(node1,node2,manager),
                     sdd_imply(node2,node1,manager),manager);
}


int main(int argc, char** argv) {

  // set up vtree and manager
  SddLiteral var_count = 5;
  int auto_gc_and_minimize = 0;
  SddManager* m = sdd_manager_create(var_count,auto_gc_and_minimize);

  SddLiteral A = 1, B = 2, C = 3, faulty1 = 4, faulty2 = 5;

  SddNode* delta = sdd_manager_true(m);
  SddNode* alpha;

  ////////// CONSTRUCT KNOWLEDGE BASE //////////

  // ~faulty1 => ( A <=> ~B )
  alpha = sdd_equiv(sdd_manager_literal(A,m),sdd_manager_literal(-B,m),m);
  alpha = sdd_imply(sdd_manager_literal(-faulty1,m),alpha,m);
  delta = sdd_conjoin(delta,alpha,m);

  // faulty1 => ( ( A <=> B ) v ~B )
  alpha = sdd_equiv(sdd_manager_literal(A,m),sdd_manager_literal(B,m),m);
  alpha = sdd_disjoin(alpha,sdd_manager_literal(-B,m),m);
  alpha = sdd_imply(sdd_manager_literal(faulty1,m),alpha,m);
  delta = sdd_conjoin(delta,alpha,m);

  // ~faulty2 => ( B <=> ~C )
  alpha = sdd_equiv(sdd_manager_literal(B,m),sdd_manager_literal(-C,m),m);
  alpha = sdd_imply(sdd_manager_literal(-faulty2,m),alpha,m);
  delta = sdd_conjoin(delta,alpha,m);

  // faulty2 => ( ( B <=> C ) v ~C )
  alpha = sdd_equiv(sdd_manager_literal(B,m),sdd_manager_literal(C,m),m);
  alpha = sdd_disjoin(alpha,sdd_manager_literal(-C,m),m);
  alpha = sdd_imply(sdd_manager_literal(faulty2,m),alpha,m);
  delta = sdd_conjoin(delta,alpha,m);

  ////////// PERFORM QUERY //////////

  int* variables;
  SddLiteral health_vars = 2, health_vars_count, missing_health_vars;

  // make observations
  delta = sdd_condition(A,delta,m);
  delta = sdd_condition(-C,delta,m);

  // check if observations are normal
  SddNode* gamma;
  gamma = sdd_condition(-faulty1,delta,m);
  gamma = sdd_condition(-faulty2,gamma,m);
  int is_abnormal = gamma == sdd_manager_false(m); // sdd_node_is_false(gamma);
  printf("observations normal?  : %s\n", is_abnormal ? "abnormal":"normal");

  // project onto faults
  SddNode* diagnosis = sdd_exists(B,delta,m);
  // diagnosis no longer depends on variables A,B or C

  // count the number of diagnoses
  SddModelCount count = sdd_model_count(diagnosis,m);
  // adjust for missing faults
  variables = sdd_variables(diagnosis,m);
  health_vars_count = variables[faulty1] + variables[faulty2];
  missing_health_vars = health_vars - health_vars_count;
  count <<= missing_health_vars; // multiply by 2^missing_health_vars
  free(variables);

  // find minimum cardinality diagnoses
  SddNode* min_diagnosis = sdd_minimize_cardinality(diagnosis,m);
  variables = sdd_variables(min_diagnosis,m);
  // adjust for missing faults
  if ( variables[faulty1] == 0 )
    min_diagnosis = sdd_conjoin(min_diagnosis,sdd_manager_literal(-faulty1,m),m);
  if ( variables[faulty2] == 0 )
    min_diagnosis = sdd_conjoin(min_diagnosis,sdd_manager_literal(-faulty2,m),m) ;
  free(variables);

  // count the number of minimum cardinality diagnoses, and minimum cardinality
  SddModelCount min_count = sdd_model_count(min_diagnosis,m);
  SddLiteral min_card =  sdd_minimum_cardinality(min_diagnosis);

  printf("sdd model count       : %"PRImcS"\n",count);
  printf("sdd model count (min) : %"PRImcS"\n",min_count);
  printf("sdd cardinality       : %"PRIlitS"\n",min_card);

  ////////// SAVE SDDS //////////

  printf("saving sdd and dot ...\n");

  sdd_save("output/circuit-kb.sdd",delta);
  sdd_save("output/diagnosis.sdd",diagnosis);
  sdd_save("output/diagnosis-min.sdd",min_diagnosis);

  sdd_save_as_dot("output/circuit-kb.dot",delta);
  sdd_save_as_dot("output/diagnosis.dot",diagnosis);
  sdd_save_as_dot("output/diagnosis-min.dot",min_diagnosis);

  ////////// CLEAN UP //////////

  sdd_manager_free(m);

  return 0;
}
