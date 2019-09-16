/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//local declarations
static Fnf* parse_fnf_file(char* buffer);

/****************************************************************************************
 * reading fnf files
 ****************************************************************************************/
 
//Reads a FNF from a file
Fnf* read_fnf(const char* filename) {
  char* buffer = read_file(filename);
  char* filtered = filter_comments(buffer);
  Fnf* fnf = parse_fnf_file(filtered);
  //cleanup
  free(buffer);
  free(filtered);
  return fnf;
}

Cnf* sdd_cnf_read(const char* filename) {
 Cnf* cnf = read_fnf(filename);
 //declare as cnf
  cnf->op=CONJOIN;
  for(SddSize i=0; i<cnf->litset_count; i++) cnf->litsets[i].op=DISJOIN;
  return cnf;
}

Dnf* sdd_dnf_read(const char* filename) {
 Dnf* dnf = read_fnf(filename);
 //declare as dnf
  dnf->op=DISJOIN;
  for(SddSize i=0; i<dnf->litset_count; i++) dnf->litsets[i].op=CONJOIN;
  return dnf;
}


/****************************************************************************************
 * parsing a .cnf/.dnf files (same format)
 ****************************************************************************************/

//Helper function: if test confirmed, print message and exit.
void test_parse_fnf_file(int test, const char* message) {
  if (test) {
    fprintf(stderr,".cnf parse error: %s\n",message);
    exit(1);
  }
}

//Helper function: strtok for int's, for reading cnf's
int cnf_int_strtok() {
  static const char* whitespace = " \t\n\v\f\r";
  char* token = strtok(NULL,whitespace);
  test_parse_fnf_file(token == NULL,"Unexpected end of file.");
  return atoi(token);
}

//Parses a cnf from a .cnf cstring, where comments have been filtered out
//This initializes the appropriate fields in the Fnf struct
Fnf* parse_fnf_file(char* buffer) {
  const char* whitespace = " \t\n\v\f\r";

  //id
  SddSize id=0;
  
  // initialize Fnf
  Fnf* cnf;
  MALLOC(cnf,Fnf,"parse_fnf_file");
  cnf->var_count = 0;
  cnf->litset_count = 0;
  cnf->litsets = NULL;

  // 1st token is "p" then check "cnf"
  char* token = strtok(buffer,whitespace);
  test_parse_fnf_file(token == NULL || strcmp(token,"p") != 0,
                      "Expected header \"p cnf\".");
  token = strtok(NULL,whitespace);
  test_parse_fnf_file(token == NULL || strcmp(token,"cnf") != 0,
                      "Expected header \"p cnf\".");

  // read variable & clause count
  cnf->var_count = cnf_int_strtok();
  cnf->litset_count = cnf_int_strtok();
  CALLOC(cnf->litsets,LitSet,cnf->litset_count,"parse_fnf_file");

  // read in clauses
  // assume longest possible clause is #-vars * 2
  LitSet* clause;
  SddLiteral* temp_clause;
  CALLOC(temp_clause,SddLiteral,cnf->var_count*2,"parse_fnf_file");
  SddLiteral lit;
  SddLiteral lit_index;
  for(SddSize clause_index = 0; clause_index < cnf->litset_count; clause_index++) {
    lit_index = 0;
    while (1) { // read a clause
      lit = cnf_int_strtok();
      if (lit == 0) break;
      test_parse_fnf_file(lit_index >= cnf->var_count*2,
                          "Unexpected long clause.");
      temp_clause[lit_index] = lit;
      lit_index++;
    }
    clause = &(cnf->litsets[clause_index]);
    clause->id = id++;
    clause->bit = 0;
    clause->literal_count = lit_index;
    CALLOC(clause->literals,SddLiteral,clause->literal_count,"parse_fnf_file");
    for(lit_index = 0; lit_index < clause->literal_count; lit_index++)
      clause->literals[lit_index] = temp_clause[lit_index];
  }

  free(temp_clause);
  return cnf;
}


/****************************************************************************************
 * end
 ****************************************************************************************/
