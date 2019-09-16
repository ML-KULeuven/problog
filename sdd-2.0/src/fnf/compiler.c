/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sddapi.h"
#include "compiler.h"

/****************************************************************************************
 * this file contains the fnf-to-sdd compiler, with AUTO gc and sdd-minimize
 ****************************************************************************************/

// local declarations
SddNode* apply_litset(LitSet* litset, SddManager* manager);
  
/****************************************************************************************
 * compiles a cnf or dnf into an sdd
 ****************************************************************************************/

static
SddNode* degenerate_fnf_test(Fnf* fnf, SddManager* manager) {
  BoolOp op     = fnf->op;
  SddSize count = fnf->litset_count;

  if(count==0) return ONE(manager,op);

  for(int i=0; i<count; i++) {
    if (fnf->litsets[i].literal_count == 0)
      return ZERO(manager,op);
  }

  return NULL;
}

SddNode* fnf_to_sdd_auto(Fnf* fnf, SddManager* manager) {
  SddCompilerOptions* options = sdd_manager_options(manager);
  int verbose      = options->verbose;
  BoolOp op        = fnf->op;
  SddSize count    = fnf->litset_count;
  LitSet** litsets = (LitSet**) malloc(count*sizeof(LitSet*));
  for (SddSize i=0; i<count; i++) litsets[i] = fnf->litsets + i;

  if(verbose) { printf("\nclauses: %ld ",count); fflush(stdout); }
  SddNode* node = ONE(manager,op);
  for(int i=0; i<count; i++) {
    sort_litsets_by_lca(litsets+i,count-i,manager);
    sdd_ref(node,manager);
    SddNode* l = apply_litset(litsets[i],manager);
    sdd_deref(node,manager);
    node = sdd_apply(l,node,op,manager);
    if(verbose) { printf("%ld ",count-i-1); fflush(stdout); }
  }
  free(litsets);
  return node;
}

SddNode* fnf_to_sdd_manual(Fnf* fnf, SddManager* manager) {
  SddCompilerOptions* options = sdd_manager_options(manager);
  int verbose      = options->verbose;
  int period       = options->vtree_search_mode;
  BoolOp op        = fnf->op;
  SddSize count    = fnf->litset_count;
  LitSet** litsets = (LitSet**) malloc(count*sizeof(LitSet*));
  for (SddSize i=0; i<count; i++) litsets[i] = fnf->litsets + i;
  sort_litsets_by_lca(litsets,count,manager);

  if(verbose) { printf("\nclauses: %ld ",count); fflush(stdout); }
  SddNode* node = ONE(manager,op);
  for(int i=0; i<count; i++) {
    if(period > 0 && i > 0 && i%period==0) {
      // after every period clauses
      sdd_ref(node,manager);
      if(options->verbose) { printf("* "); fflush(stdout); }
      sdd_manager_minimize_limited(manager);
      sdd_deref(node,manager);
      sort_litsets_by_lca(litsets+i,count-i,manager);
    }

    SddNode* l = apply_litset(litsets[i],manager);
    node = sdd_apply(l,node,op,manager);
    if(verbose) { printf("%ld ",count-i-1); fflush(stdout); }
  }
  free(litsets);
  return node;
}

SddNode* fnf_to_sdd(Fnf* fnf, SddManager* manager) {
  SddNode* test = degenerate_fnf_test(fnf,manager);
  if (test != NULL) return test;
  SddCompilerOptions* options = sdd_manager_options(manager);

  if(options->vtree_search_mode < 0) {
    sdd_manager_auto_gc_and_minimize_on(manager);
    return fnf_to_sdd_auto(fnf,manager);
  } else {
    sdd_manager_auto_gc_and_minimize_off(manager);
    return fnf_to_sdd_manual(fnf,manager);
  }
}

//converts a clause/term into an equivalent sdd
SddNode* apply_litset(LitSet* litset, SddManager* manager) {

  BoolOp op            = litset->op; //conjoin (term) or disjoin (clause)
  SddLiteral* literals = litset->literals;
  SddNode* node        = ONE(manager,op); //will not be gc'd
  
  for(SddLiteral i=0; i<litset->literal_count; i++) {
    SddNode* literal = sdd_manager_literal(literals[i],manager);
    node             = sdd_apply(node,literal,op,manager);
  }
  
  return node;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
