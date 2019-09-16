/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include <signal.h>
#include "sdd.h"


void press(char* str) {
  printf("\n%s, press enter?",str); fflush(stdout);
  getchar();
}

/****************************************************************************************
 * printing stack when assertion fails
 ****************************************************************************************/

#define CALLS_COUNT 11
//prints the last CALLS_COUNT-1 calls, excluding print_trace()
void print_trace () {
  void *array[CALLS_COUNT];
  int size;
  char **strings;

  size    = backtrace(array,CALLS_COUNT);
  strings = backtrace_symbols(array,size);

  printf("\n");
  for (int i=0; i<size-1; i++) printf("%s\n",strings[CALLS_COUNT-i-1]);

  free (strings);
}

/****************************************************************************************
 * printing utilities
 ****************************************************************************************/

//pretty prints a number into a string, adding commas after thousands
//for example, 1234567 is mapped into "1,234,567"
//please free after use
char* ppc(SddSize n) {
  //count digits
  int c = 0;
  if(n) {
    SddSize m = n;
    while(m) { c++; m /= 10; }
    c = c + (c-1)/3 + 1; //accounting for commas and terminal
  } else c=2;
  //create string
  char* str;
  CALLOC(str,char,c,"ppc");
  //fill string
  str += c;
  *--str = '\0';
  if(n) {
    int l=0;
    while(n) {
      int d = n%10;
      n /= 10;
      *--str = (char)(d+'0');
      if(n && ++l==3) { l=0; *--str = ','; }
    }
  }
  else *--str = '0'; 
  return str;
}

//given an integer i, returns a string "vt_i.gv"
//please free after use
char* int_to_file_name(char* fname, int i) {  
  int digits = 0;
  if(i==0) digits = 1;
  else {
    int tmp = i;
    while (tmp) { tmp /= 10; digits++; }
  }
  char* string;
  CALLOC(string,char,digits+strlen(fname)+5,"int_to_vtree_file_name");
  sprintf(string,"%s_%d.gv",fname,i);
  
  return string;
}

//given a literal, returns a cstring representation, for .dot files
//please free after use
char* literal_to_label(SddLiteral lit) {
  static const char* nice_var_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  SddLiteral var = lit < 0 ? -lit : lit; // abs

  char* var_string;
  if (var <= 26) {
    CALLOC(var_string,char,2,"literal_to_label");
    var_string[0] = nice_var_names[var-1];
    var_string[1] = '\0';
  } else {
    int digits = 0;
    if ( var == 0 ) // var shouldn't be zero ...
      digits = 1;
    else {
      SddLiteral tmp_var = var;
      while (tmp_var) { tmp_var /= 10; digits++; }
    }
    CALLOC(var_string,char,digits+1,"literal_to_label");
    sprintf(var_string,"%"PRIlitS"",var);
  }

  char* lit_string;
  if (lit < 0) {
    CALLOC(lit_string,char,strlen(var_string)+5+1,"literal_to_label");
    sprintf(lit_string,"&not;%s",var_string);
    free(var_string);
  } else {
    lit_string = var_string;
  }

  return lit_string;
}


/****************************************************************************************
 * parsing utilities
 ****************************************************************************************/

//Helper function: if test confirmed, print message and exit
void test_and_exit(int test, const char* message) {
  if (test) {
    fprintf(stderr,"parse error: %s\n",message);
    exit(1);
  }
}

//Helper function: error for unknown node type during parse
void unexpected_node_type_error(char node_type) {
  fprintf(stderr,"parse error: ");
  fprintf(stderr,"Unexpected node type %c\n",node_type);
  exit(1);
}


//Helper function: strtok for reading sdd's, vtree's, etc
//For reading the first token of a buffer, which is a constant
//e.g., "sdd" in .sdd files or "vtree" in .vtree files
void header_strtok(char* buffer, const char* expected_token) {
  static const char* whitespace = " \t\n\v\f\r";
  char* token = strtok(buffer,whitespace);
  test_and_exit(token == NULL,"Unexpected end of file.");
  if (strcmp(token,expected_token) != 0) {
    fprintf(stderr,"parse error: ");
    fprintf(stderr,"Expected token %s\n",expected_token);
    exit(1);
  }
}

//Helper function: strtok for int's, for parsing sdd's, vtree's, etc
//Assumes a token has been read before: this function uses strtok(NULL,.)
int int_strtok() {
  static const char* whitespace = " \t\n\v\f\r";
  char* token = strtok(NULL,whitespace);
  test_and_exit(token == NULL,"Unexpected end of file.");
  return atoi(token);
}

//Helper function: strtok for char's, for parsing sdd's, vtree's, etc
//Assumes a token has been read before: this function uses strtok(NULL,.)
char char_strtok() {
  static const char* whitespace = " \t\n\v\f\r";
  char* token = strtok(NULL,whitespace);
  test_and_exit(token == NULL,"Unexpected end of file.");
  test_and_exit(strlen(token) != 1,"Expected node type.");
  return token[0];
}


//Reads a file given filename.  Returns a cstring of length the size of the file
char* read_file(const char* filename) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    printf("Could not open the file %s\n",filename);
    exit(1);
  }

  // lookup file size
  // AC: I don't think this is the best way to do this
  fseek(file,0,SEEK_END);
  unsigned int file_size = ftell(file);
  rewind(file);

  // allocate memory
  char* buffer;
  CALLOC(buffer,char,file_size+1,"read_file");

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
char* filter_comments(const char* buffer) {
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
  char* filtered;
  CALLOC(filtered,char,file_length+1,"read_file");
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
 * end
 ****************************************************************************************/
