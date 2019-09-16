/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include <ctype.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include "sddapi.h"
#include "compiler.h"

void print_help(const char* PACKAGE, int exit_value);
const char* libsdd_version();

/****************************************************************************************
 * start
 ****************************************************************************************/
 
SddCompilerOptions sdd_getopt(int argc, char **argv) {
  char* PACKAGE = "sdd";

  //default options 
  SddCompilerOptions options = 
    {
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL, //file names
    0, //flag
    "balanced", //vtree type
    -1, //vtree search mode
    0, //post-compilation search
    0 //verbose
    };
  int option;
  int input_count = 0;
     
  while ((option = getopt(argc,argv,"c:d:v:s:W:V:R:S:hmt:r:qp")) != -1) {
    switch (option) {
    //input
    case 'c':
      options.cnf_filename = optarg;
      input_count++;
      break;
    case 'd':
      options.dnf_filename = optarg;
      input_count++;
      break;
    case 'v':
      options.vtree_filename = optarg;
      break;
    case 's':
      options.sdd_filename = optarg;
      input_count++;
      break;
    //output
    case 'W':
      options.output_vtree_filename = optarg;
      break;
    case 'V':
      options.output_vtree_dot_filename = optarg;
      break;
    case 'R':
      options.output_sdd_filename = optarg;
      break;
    case 'S':
      options.output_sdd_dot_filename = optarg;
      break;
    //flags
    case 'h': //HELP
      print_help(PACKAGE,0);
      break;
    case 'm': 
      options.minimize_cardinality = 1;
      break;
    //options with arguments
    case 't':
      options.initial_vtree_type = optarg;
      break;
    case 'r':
      options.vtree_search_mode = strtol(optarg,NULL,10);
      break;
    case 'q':
      options.post_search = 1;
      break;
    case 'p':
      options.verbose = 1;
      break;
    default:
      print_help(PACKAGE,1);
    }
  }

  //checking validity of options  
  if(input_count != 1) {
    fprintf(stderr, "%s: must specify exactly one of a cnf, dnf or sdd file\n",PACKAGE);
    print_help(PACKAGE,1);
  }
  if (options.sdd_filename!=NULL && options.vtree_filename==NULL) {
    fprintf(stderr, "%s: sdd file must be specified with vtree file\n",PACKAGE);
    print_help(PACKAGE,1);
  }
  if(strcmp(options.initial_vtree_type,"left") && 
     strcmp(options.initial_vtree_type,"right") &&
     strcmp(options.initial_vtree_type,"vertical") && 
     strcmp(options.initial_vtree_type,"balanced") && 
     strcmp(options.initial_vtree_type,"random")) {
   fprintf(stderr, "%s: initial vtree type must be one of: left, right, vertical, balanced, or random\n",PACKAGE);
    print_help(PACKAGE,1);
  } 
 
  return options;
}


void print_help(const char* PACKAGE, int exit_value) {
  printf("%s: Sentential Decision Diagram, Compiler\n", PACKAGE);
  printf("%s\n",libsdd_version());
  printf("%s (-c FILE | -d FILE | -s FILE) [-v FILE | -t TYPE] [-WVRS FILE] [-r MODE] [-mhqp]\n", PACKAGE);

  printf("  -c FILE         set input CNF file\n"); 
  printf("  -d FILE         set input DNF file\n");  
  printf("  -v FILE         set input VTREE file\n");
  printf("  -s FILE         set input SDD file\n"); 

  printf("  -W FILE         set output VTREE file\n");
  printf("  -V FILE         set output VTREE (dot) file\n");
  printf("  -R FILE         set output SDD file\n");
  printf("  -S FILE         set output SDD (dot) file\n");

  printf("  -m              minimize the cardinality of compiled sdd\n");

  printf("  -t TYPE         set initial vtree type (left/right/vertical/balanced/random)\n");

  printf("  -r K            if K>0: invoke vtree search every K clauses\n");
  printf("                    if K=0: disable vtree search\n");
  printf("                    by default (no -r option), dynamic vtree search is enabled\n");
  printf("  -q              perform post-compilation vtree search\n");

  printf("  -h              print this help and exit\n");
  printf("  -p              verbose output\n");
  exit(exit_value);
}

/****************************************************************************************
 * end
 ****************************************************************************************/
