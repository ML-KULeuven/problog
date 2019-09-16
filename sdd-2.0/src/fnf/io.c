/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include <string.h>
#include "sddapi.h"
#include "compiler.h"

//local declarations
static Fnf* parse_fnf_file(char* buffer);

/****************************************************************************************
 * general file reading
 ****************************************************************************************/

//Reads a file given filename.  Returns a cstring of length the size of the file
static char* read_file(const char* filename) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    printf("Could not open the file %s\n",filename);
    exit(1);
  }

  // lookup file size
  fseek(file,0,SEEK_END);
  unsigned int file_size = ftell(file);
  rewind(file);

  // allocate memory
  char* buffer = (char*)calloc(file_size+1,sizeof(char));

  // read the whole file
  unsigned int result = fread(buffer,sizeof(char),file_size,file);
  if (result != file_size) {
    printf("Could not read the file %s\n",filename);
    exit(1);
  }
  buffer[file_size] = 0; // null terminate

  fclose(file);
  return buffer;
}

//Filters commented lines (beginning with 'c') and returns a cstring
//whose length is the length of the resulting file
static char* filter_comments(const char* buffer) {
  int is_comment_line, is_eol;
  unsigned int file_length = 0, line_length;
  const char* read_head = buffer;
  char* write_head;

  // count size of filtered string
  while (*read_head != '\0') {
    is_comment_line = (*read_head == 'c');
    line_length = 0;
    while (*read_head != '\0') {
      is_eol = (*read_head == '\n');
      read_head++;
      line_length++;
      if (is_eol) break;
    }
    if (!is_comment_line)
      file_length += line_length;
  }

  // copy filtered string
  char* filtered = (char*)calloc(file_length+1,sizeof(char));
  read_head = buffer;
  write_head = filtered;
  while (*read_head != '\0') {
    is_comment_line = (*read_head == 'c');
    while (*read_head != '\0') {
      is_eol = (*read_head == '\n');
      if (!is_comment_line) {
        *write_head = *read_head;
        write_head++;
      }
      read_head++;
      if (is_eol) break;
    }
  }
  *write_head = '\0';
  return filtered;
}

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

Cnf* read_cnf(const char* filename) {
 Cnf* cnf = read_fnf(filename);
 //declare as cnf
  cnf->op=CONJOIN;
  for(SddSize i=0; i<cnf->litset_count; i++) cnf->litsets[i].op=DISJOIN;
  return cnf;
}

Dnf* read_dnf(const char* filename) {
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
static void test_parse_fnf_file(int test, const char* message) {
  if (test) {
    fprintf(stderr,".cnf parse error: %s\n",message);
    exit(1);
  }
}

//Helper function: strtok for int's, for reading cnf's
static int cnf_int_strtok() {
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
  Fnf* cnf = (Fnf*)malloc(sizeof(Fnf));
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
  cnf->litsets = (LitSet*)calloc(cnf->litset_count,sizeof(LitSet));

  // read in clauses
  // assume longest possible clause is #-vars * 2
  LitSet* clause;
  SddLiteral* temp_clause = (SddLiteral*)calloc(cnf->var_count*2,sizeof(SddLiteral));
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
    clause->literals = (SddLiteral*)calloc(clause->literal_count,sizeof(SddLiteral));
    for(lit_index = 0; lit_index < clause->literal_count; lit_index++)
      clause->literals[lit_index] = temp_clause[lit_index];
  }

  free(temp_clause);
  return cnf;
}


/****************************************************************************************
 * end
 ****************************************************************************************/
