/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
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
 
//M: manager
//V: vtree
#define ZERO(M,OP) (OP==CONJOIN? sdd_manager_false(M): sdd_manager_true(M))
#define ONE(M,OP) (OP==CONJOIN? sdd_manager_true(M): sdd_manager_false(M))

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
  char* sdd_filename;        //input sdd filename (.sdd, plain text)
  //output files
  char* output_vtree_filename;     //output vtree filename (.vtree, plain text)
  char* output_vtree_dot_filename; //output vtree filename (.dot)
  char* output_sdd_filename;       //output sdd filename (.sdd, plain text)
  char* output_sdd_dot_filename;   //output sdd filename (.dot)
  //flags
  int minimize_cardinality;   //construct sdd that has only minimum cardinality models
  //initial vtree
  char* initial_vtree_type;   //initial vtree for manager
  //compilation controls
  int vtree_search_mode;      // vtree search mode
  int post_search;            // post-compilation search
  int verbose;                // print manager
} SddCompilerOptions;

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
 * function declaration
 ****************************************************************************************/

Cnf* read_cnf(const char* filename);
Dnf* read_dnf(const char* filename);
void free_fnf(Fnf* fnf);

SddNode* fnf_to_sdd(Fnf* fnf, SddManager* manager);

/****************************************************************************************
 * forward references 
 ****************************************************************************************/
 
void sort_litsets_by_lca(LitSet** litsets, SddSize litset_count, SddManager* manager);


#endif // COMPILER_H_

/****************************************************************************************
 * end
 ****************************************************************************************/
