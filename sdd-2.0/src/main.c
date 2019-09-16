/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include <time.h>
#include "sddapi.h"
#include "compiler.h"

// forward references
SddCompilerOptions sdd_getopt(int argc, char **argv); // in getopt.c
char* ppc(SddSize n); // pretty print (in libsdd: util.c)

void print_node(SddNode* node, SddManager* manager) {
  char* s = NULL;
  printf(    " sdd size               : %s \n",s=ppc(sdd_size(node))); free(s);
  printf(    " sdd node count         : %s \n",s=ppc(sdd_count(node))); free(s);

  clock_t c1, c2;
  c1 = clock();
  SddModelCount mc = sdd_global_model_count(node,manager);
  c2 = clock();
  printf(    " sdd model count        : %s    %.3f sec\n",s=ppc(mc),(float)(c2-c1)/CLOCKS_PER_SEC); free(s);
}

/****************************************************************************************
 * start
 ****************************************************************************************/
 
int main(int argc, char** argv) {

  //get options from command line (and defaults)
  SddCompilerOptions options = sdd_getopt(argc,argv);

  Fnf* fnf = NULL;
  Vtree* vtree;
  SddNode* node;
  SddManager* manager;
  clock_t c1, c2;
  char* s;

  if(options.cnf_filename!=NULL) {
    printf("\nreading cnf...");
    fnf = read_cnf(options.cnf_filename);
    printf("vars=%"PRIlitS" clauses=%"PRIsS"",fnf->var_count,fnf->litset_count);
  }
  else if(options.dnf_filename!=NULL) {
    printf("\nreading dnf...");
    fnf = read_dnf(options.dnf_filename);
    printf("vars=%"PRIlitS" terms=%"PRIsS"",fnf->var_count,fnf->litset_count);
  }
  
  if(options.vtree_filename!=NULL) {
    printf("\nreading initial vtree...");
    vtree = sdd_vtree_read(options.vtree_filename);
  } 
  else {
    printf("\ncreating initial vtree (%s)...",options.initial_vtree_type);
    vtree = sdd_vtree_new(fnf->var_count,options.initial_vtree_type);
  }
  
  printf("\ncreating manager...");
  //create manager
  manager = sdd_manager_new(vtree);
  //no longer needed
  sdd_vtree_free(vtree);
  //passing compiler options to manager
  sdd_manager_set_options(&options,manager);

  if(options.sdd_filename==NULL) {
    printf("\ncompiling..."); fflush(stdout);
    c1 = clock();
    node = fnf_to_sdd(fnf,manager);
    c2 = clock();
    float secs = (float)(c2-c1)/CLOCKS_PER_SEC;
    printf("\n\ncompilation time        : %.3f sec\n",secs);
  } else {
    printf("\nreading sdd from file..."); fflush(stdout);
    c1 = clock();
    node = sdd_read(options.sdd_filename,manager);
    c2 = clock();
    float secs = (float)(c2-c1)/CLOCKS_PER_SEC;
    printf("\n\nread time               : %.3f sec\n",secs);
  }

  print_node(node,manager);
  if(options.verbose)
    sdd_manager_print(manager);

  if(options.minimize_cardinality) {
    printf("\nminimizing cardinality..."); fflush(stdout);
    c1 = clock();
    node = sdd_global_minimize_cardinality(node,manager);
    c2 = clock();
    SddLiteral min_card = sdd_minimum_cardinality(node);
    printf("\n");
    print_node(node,manager);
    printf(" min cardinality        : %ld   %.3f sec\n",min_card,(float)(c2-c1)/CLOCKS_PER_SEC);
  }
  
  Vtree* manager_vtree = sdd_manager_vtree(manager);

  if(options.post_search==1) {
    sdd_ref(node,manager);
    printf("\ndynamic vtree (post compilation)\n");
    printf(    " sdd initial size       : %s\n",s=ppc(sdd_size(node))); free(s);
    fflush(stdout);
    c1 = clock();
    sdd_manager_minimize_limited(manager);
    c2 = clock();
    printf("\n");
    printf(    " dynamic vtree time     : %.3f sec\n",(float)(c2-c1)/CLOCKS_PER_SEC);
    print_node(node,manager);
    sdd_deref(node,manager);
    if(options.verbose)
      sdd_manager_print(manager);
  }

  if(options.output_sdd_filename != NULL) {
    printf("saving compiled sdd ...");
    sdd_save(options.output_sdd_filename,node);
    printf("done\n");
  }

  if(options.output_sdd_dot_filename != NULL) {
    printf("saving compiled sdd (dot)...");
    sdd_save_as_dot(options.output_sdd_dot_filename,node);
    printf("done\n");
  }

  if(options.output_vtree_filename != NULL) {
    printf("saving vtree...");
    sdd_vtree_save(options.output_vtree_filename,manager_vtree);
    printf("done\n"); 
  }

  if(options.output_vtree_dot_filename != NULL) {
    printf("saving vtree (dot)...");
    sdd_vtree_save_as_dot(options.output_vtree_dot_filename,manager_vtree);
    printf("done\n"); 
  }

  printf("freeing...");
  fflush(stdout);
  if(options.cnf_filename!=NULL || options.dnf_filename!=NULL) {
    free_fnf(fnf);
  }
  sdd_manager_free(manager);
  printf("done\n"); 

  return 0;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
