/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//declarations

//vtrees/vtree.c
Vtree* new_leaf_vtree(SddLiteral var);
Vtree* new_internal_vtree(Vtree* left_child, Vtree* right_child);

//local declarations
static Vtree* parse_vtree_file(char* buffer);

/****************************************************************************************
 * reading vtrees from .vtree file
 ****************************************************************************************/

//Reads a vtree from a .vtree file
Vtree* sdd_vtree_read(const char* filename) {
  char* buffer = read_file(filename);
  char* filtered = filter_comments(buffer);
  Vtree* vtree = parse_vtree_file(filtered);
  free(buffer);
  free(filtered);
  return vtree;
}

/****************************************************************************************
 * printing vtrees to .vtree file
 *
 * the ->position of a vnode is used as its id to ensure an inorder labeling
 ****************************************************************************************/

//prints a vtree node
void print_vtree_node(FILE* file, const Vtree* vnode) {
  if(LEAF(vnode)) {
    fprintf(file,"L %"PRIsS" %"PRIlitS"",vnode->position,vnode->var);
  } 
  else { // internal node
    print_vtree_node(file,vnode->left);
    print_vtree_node(file,vnode->right);
    fprintf(file,"I %"PRIsS" %"PRIsS" %"PRIsS"",vnode->position,vnode->left->position,vnode->right->position);
  }
  fprintf(file,"\n");
}

void print_vtree_header(FILE* file) {
  static const char* header = 
    "c ids of vtree nodes start at 0\n"
    "c ids of variables start at 1\n"
    "c vtree nodes appear bottom-up, children before parents\n"
    "c\n"
    "c file syntax:\n"
    "c vtree number-of-nodes-in-vtree\n"
    "c L id-of-leaf-vtree-node id-of-variable\n"
    "c I id-of-internal-vtree-node id-of-left-child id-of-right-child\n"
    "c\n";
  fprintf(file,"%s",header);
}

//prints a vtree in .vtree file format
void print_vtree(FILE* file, const Vtree* vtree) {
  SddLiteral count = 2*(vtree->var_count) -1;
  print_vtree_header(file);
  fprintf(file,"vtree %"PRIsS"\n",count);
  print_vtree_node(file,vtree);
}

//saves vtree to file
void sdd_vtree_save(const char* fname, Vtree* vtree) { 
  FILE *file = fopen(fname,"w");
  print_vtree(file,vtree);
  fclose(file); 
}

/****************************************************************************************
 * printing vtrees to .dot file
 *
 * the ->position of a vnode is used as its id to ensure an inorder labeling
 ****************************************************************************************/
extern SddManager* man;

void print_vtree_nodes_as_dot(FILE* file, const Vtree* vtree) {
  SddLiteral position = vtree->position;
  char* shape = "plaintext";
  
  if(LEAF(vtree)) {
    SddLiteral var = vtree->var;
    char* var_string = literal_to_label(var);
    fprintf(file,"\nn%"PRIsS" [label=\"%s\",fontname=\"Times-Italic\","
            "fontsize=14,shape=\"%s\",fixedsize=true,width=.25,height=.25"
            "]; ",position,var_string,shape);
    free(var_string);
  } 
  else {
    fprintf(file,"\nn%"PRIsS" [label=\"%"PRIsS"\",fontname=\"Times\","
           "shape=\"%s\",fontsize=12,fixedsize=true,width=.2,height=.18]; ",
           position,position,shape);
    print_vtree_nodes_as_dot(file,vtree->left);
    print_vtree_nodes_as_dot(file,vtree->right);
  }
}

void print_vtree_edges_as_dot(FILE* file, const Vtree* vtree, const Vtree* parent) {
  SddLiteral position = vtree->position;
  if(LEAF(vtree)) {
    if(parent != NULL) {
      SddLiteral parent_position = vtree->parent->position;
      fprintf(file,"\nn%"PRIsS"->n%"PRIsS" [headclip=true,arrowhead=none,headlabel=\"%"PRIsS"\","
              "labelfontname=\"Times\",labelfontsize=10];",parent_position,position,position);
    }
  } 
  else { //internal node
    if(parent != NULL) {
      SddLiteral parent_position = vtree->parent->position;
      fprintf(file,"\nn%"PRIsS"->n%"PRIsS" [arrowhead=none];",parent_position,position);
    }
    print_vtree_edges_as_dot(file,vtree->left,vtree);
    print_vtree_edges_as_dot(file,vtree->right,vtree); // right first?
  }
}

//prints a vtree in .dot file format
void print_vtree_as_dot(FILE* file, const Vtree* vtree) {
  fprintf(file,"\ndigraph vtree {");
  fprintf(file,"\n\noverlap=false");
  fprintf(file,"\n");

  print_vtree_nodes_as_dot(file,vtree);
  //allows printing vtrees of internal nodes (i.e., subtrees -- hence, the NULL below)
  print_vtree_edges_as_dot(file,vtree,NULL);

  fprintf(file,"\n\n");
  fprintf(file,"\n}");
}

void sdd_vtree_save_as_dot(const char* fname, Vtree* vtree) { 
  FILE *file = fopen(fname,"w");
  print_vtree_as_dot(file,vtree);
  fclose(file); 
}


/****************************************************************************************
 * Parses a vtree from a .vtree cstring, where comments have been filtered out
 *
 * Returns root of vtree
 ***************************************************************************************/

//->position is just used for temporary indexing --- its final value is irrelevant
Vtree* parse_vtree_file(char* buffer) {
  //1st token is "vtree"
  header_strtok(buffer,"vtree");

  //read vtree node count
  SddLiteral node_count = int_strtok();
  
  //create a map from positions to vtree nodes
  Vtree** vtree_node_list;
  CALLOC(vtree_node_list,Vtree*,node_count,"parse_vtree_file");
  
  Vtree* vnode = NULL; //to avoid compiler warning
  for(SddLiteral count=0; count<node_count; count++) {
  
    //read node type (leaf or internal)
    char node_type = char_strtok();
    
    //read position of vtree
    SddLiteral position = int_strtok();

    //construct vtree node
    if(node_type == 'L') { //leaf
      //read index of var associated with leaf vtree
      SddLiteral var = int_strtok();
      //construct leaf vtree
      vnode = new_leaf_vtree(var);
    } else if(node_type == 'I') { // internal
      //read positions of left and right children
      SddLiteral left_position = int_strtok();
      SddLiteral right_position = int_strtok();
      //lookup left and right children
      Vtree* left = vtree_node_list[left_position];
      Vtree* right = vtree_node_list[right_position];
      //construct internal vtree
      vnode = new_internal_vtree(left,right);
    } else {
      unexpected_node_type_error(node_type);
    }
    //save position
    vnode->position = position;
    
    //index constructed vtree node by its position
    vtree_node_list[position] = vnode;
  }
  
  free(vtree_node_list);
  // last node is root
  return vnode;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
