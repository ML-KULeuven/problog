/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 1.1.1, January 31, 2014
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/
 
#include <stdio.h>
#include <stdlib.h>

/****************************************************************************************
 * this file contains macros, definitions of structures, and forward references
 * used by the fnf-to-sdd compiler (auto and manual versions)
 ****************************************************************************************/

/****************************************************************************************
 * macros 
 ****************************************************************************************/
 
//OP: constant, which is either CONJOIN or DISJOIN
//M: manager
#define ZERO(M,OP) (OP==CONJOIN? sdd_manager_false(M): sdd_manager_true(M))
#define ONE(M,OP) (OP==CONJOIN? sdd_manager_true(M): sdd_manager_false(M))

//data associated with vtree nodes (clause or term)
#define DATA(V,F) ((VtreeData*)sdd_vtree_data(V))->F
#define NEW_DATA(data) data = (VtreeData*)malloc(sizeof(VtreeData))
#define FREE_DATA(V) { free(DATA(V,litsets)); free((VtreeData*)sdd_vtree_data(V)); }

/****************************************************************************************
 * compiler options: populated by main.c using command-line arguments
 * default values for some of these options can be found in parameter.h
 ****************************************************************************************/
 
#ifndef COMPILER_H_
#define COMPILER_H_

typedef struct {
  //input files
  char* cnf_filename;        //input cnf filename
  char* dnf_filename;        //input dnf filename
  char* vtree_filename;      //input vtree filename
  //output files
  char* output_vtree_filename;     //output vtree filename (.vtree, plain text)
  char* output_vtree_dot_filename; //output vtree filename (.dot)
  char* output_sdd_filename;       //output sdd filename (.sdd, plain text)
  char* output_sdd_dot_filename;   //output sdd filename (.dot)
  //flags
  int auto_minimize_and_gc;
  int minimize_cardinality;   //construct sdd that has only minimum cardinality models
  //initial vtree
  char* initial_vtree_type;   //initial vtree for manager
  //compilation controls
  float gc_threshold;         //threshold for invoking garbage collection
  float vtree_search_threshold; // sdd growth threshold for invoking vtree search
  int vtree_search_mode;      // vtree search mode
} SddCompilerOptions;


//compiler options are treated as manager options
typedef SddCompilerOptions SddManagerOptions;

/****************************************************************************************
 * structures for representing:
 * --cnfs and dnfs (called fnfs)
 * --clauses and terms (called litsets)
 *
 * this struct should not be changed as it is also defined in the sdd library
 ****************************************************************************************/
 
typedef struct {
  SddSize id;
  SddLiteral literal_count;
  SddLiteral* literals; 
  BoolOp op; //DISJOIN (clause) or CONJOIN (term)
  Vtree* vtree;
  unsigned bit:1;
} LitSet;

typedef struct {
  SddLiteral var_count; // number of variables
  SddSize litset_count; // number of literal sets
  LitSet* litsets;  // array of literal sets
  BoolOp op; //CONJOIN (CNF) or DISJOIN (DNF)
} Fnf;

typedef Fnf Cnf;
typedef Fnf Dnf;

/****************************************************************************************
 * structures for associating additional information with vtree nodes:
 * --VtreeData is used to store clauses or terms at each vtree node
 ****************************************************************************************/
 
typedef struct {
  SddLiteral litset_count;
  LitSet** litsets;
} VtreeData;

/****************************************************************************************
 * function declaration
 ****************************************************************************************/

// these are library functions
Cnf* sdd_cnf_read(const char* filename);
Dnf* sdd_dnf_read(const char* filename);

/****************************************************************************************
 * forward references 
 ****************************************************************************************/
 
void free_vtree_data(Vtree* vtree);
void distribute_fnf_over_vtree(Fnf* fnf, SddManager* manager);
void sort_litsets_by_lca(LitSet** litsets, SddSize litset_count, SddManager* manager);
Vtree* vtree_search(Vtree* vtree, SddManager* manager);


#endif // COMPILER_H_

/****************************************************************************************
 * end
 ****************************************************************************************/
