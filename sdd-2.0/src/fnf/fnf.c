/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sddapi.h"
#include "compiler.h"

/****************************************************************************************
 * this file contains some utilities for manipulating Cnf and Dnf (represented as Fnf)
 *
 * Fnf consists of a set of LitSets
 * Litset is a set of literals and can be used to represent either a clause or a term
 *
 * the types Cnf and Dnf are defined as the type Fnf
 ****************************************************************************************/

//checks type of fnf 
int is_cnf(Fnf* fnf) {
  return fnf->op==CONJOIN;
}

int is_dnf(Fnf* fnf) {
  return fnf->op==DISJOIN;
}

//free the memory allocated for FNF
void free_fnf(Fnf* fnf) {
  for(SddSize i=0; i<fnf->litset_count; i++) free(fnf->litsets[i].literals);
  free(fnf->litsets);
  free(fnf);
}

//print FNF in .cnf file format
void print_fnf(char* type, FILE* file, const Fnf* fnf) {
  fprintf(file,"p %s %"PRIlitS" %"PRIsS"\n",type,fnf->var_count,fnf->litset_count);
  for (SddSize i=0; i<fnf->litset_count; i++) {
    LitSet* litset = fnf->litsets + i;
    for (SddLiteral j=0; j<litset->literal_count; j++) fprintf(file,"%"PRIlitS" ",litset->literals[j]);
    fprintf(file,"0\n");
  }
}

void print_cnf(FILE* file, const Cnf* cnf) {
  print_fnf("cnf",file,cnf);
}

void print_dnf(FILE* file, const Dnf* dnf) {
  print_fnf("dnf",file,dnf);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
