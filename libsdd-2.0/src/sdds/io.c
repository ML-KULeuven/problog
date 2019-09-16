/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//local declarations
static SddNode* parse_sdd_file(char* buffer, SddManager* manager);

/****************************************************************************************
 * printing sdd to .dot file
 ****************************************************************************************/

//terminal nodes are smaller than decomposition nodes
//decomposition node n1 is smaller than decomposition node n2 iff n1->vtree->position < n2->vtree->position
int sdd_node_comparator(const void* xp, const void* yp) { 

  SddNode* x = *((SddNode**) xp);
  SddNode* y = *((SddNode**) yp);
  
  //vtree positions start at 0
  SddSize xx = x->type==DECOMPOSITION? x->vtree->position: -1;
  SddSize yy = y->type==DECOMPOSITION? y->vtree->position: -1;

  if (xx > yy) return -1;
  else if (xx < yy) return 1;
  else return 0;

}


char* get_sdd_node_label(SddNode* node) {

  char* true_label  =  (char*) "&#8868;"; //watch this cast
  char* false_label = (char*) "&#8869;"; //watch this cast
  char* no_label = (char*) ""; //watch this cast

  if(node->type==TRUE) return true_label;
  else if(node->type==FALSE) return false_label;
  else if(node->type==LITERAL) return literal_to_label(LITERAL_OF(node));
  else return no_label;
  
}


void print_terminal_sdd_node_as_dot(FILE* file, SddNode* node) {
 
  char* label = get_sdd_node_label(node);
  fprintf(file,"\nn%"PRIsS" [label= \"%s\",shape=box]; ",node->id,label);
  if (node->type==LITERAL) free(label);
  return;
	
}

void print_decomposition_sdd_node_as_dot(FILE* file, SddNode* node) {

  static const char* node_format = "\nn%"PRIsS" [label= \"%"PRIsS"\",style=filled,fillcolor=gray95,shape=circle,height=.25,width=.25]; ";
  static const char* element_format = "\nn%"PRIsS"e%"PRIsS"\n"
    "      [label= \"<L>%s|<R>%s\",\n"
    "      shape=record,\n"
    "      fontsize=20,\n"
    "      fontname=\"Times-Italic\",\n"
    "      fillcolor=white,\n"
    "      style=filled,\n"
    "      fixedsize=true,\n"
    "      height=.30, \n"
    "      width=.65];\n";

  static const char* or_format = "\nn%"PRIsS"->n%"PRIsS"e%"PRIsS" [arrowsize=.50];";
  static const char* prime_format = "\nn%"PRIsS"e%"PRIsS":L:c->n%"PRIsS""
    " [arrowsize=.50,tailclip=false,arrowtail=dot,dir=both];";
  static const char* sub_format = "\nn%"PRIsS"e%"PRIsS":R:c->n%"PRIsS""
    " [arrowsize=.50,tailclip=false,arrowtail=dot,dir=both];";

  //decision node
  fprintf(file,node_format,node->id,node->vtree->position);

  SddSize i=0;
  FOR_each_prime_sub_of_node(prime,sub,node,{
  
    char* prime_label = get_sdd_node_label(prime);
	char* sub_label = get_sdd_node_label(sub);
	//element: prime & sub
	fprintf(file,element_format,node->id,i,prime_label,sub_label);

	if(prime->type == LITERAL) free(prime_label);
	if(sub->type == LITERAL) free(sub_label);
	//edge into element
	fprintf(file,or_format,node->id,node->id,i);
	//edge out of prime cell
	if(prime->type==DECOMPOSITION) fprintf(file,prime_format,node->id,i,prime->id);
	//edge out of sub cell
    if(sub->type==DECOMPOSITION) fprintf(file,sub_format,node->id,i,sub->id);
	++i;
  });
}

//two nodes have the same rank iff they are normalized for the same vtree (better visualization)
void print_sdd_node_ranks(FILE* file, SddSize count, SddNode** nodes) {
  assert(count>0);
  
  while(count) {
    assert((*nodes)->type==DECOMPOSITION);
    Vtree* vtree = (*nodes)->vtree; //vtree of next group of nodes (same rank)
	fprintf(file,"\n{rank=same; ");
	while(count && (*nodes)->vtree==vtree) { 
	  fprintf(file,"n%"PRIsS" ",(*nodes)->id); 
	  --count;
	  ++nodes; 
	}
	fprintf(file,"}");
  }
  fprintf(file,"\n");

}

void print_sdd_nodes_as_dot(FILE* file, SddSize count, SddNode** nodes) {
  assert(count>1);
  
  //sort nodes so that:
  //--terminal nodes appear first, followed by decomposition nodes
  //--decomposition nodes are sorted by their vtrees
  qsort(nodes,count,sizeof(SddNode*),sdd_node_comparator); 

  //skip terminal nodes
  while((*nodes)->type!=DECOMPOSITION) { --count; ++nodes; }
  
  assert(count!=0); //at least one decomposition node
  
  //declare the ranks of decomposition nodes (equal rank iff normalized for same vtree)
  print_sdd_node_ranks(file,count,nodes);
  
  //print decomposition nodes (terminal nodes will be printed as a side effect)
  for(SddSize i=0; i<count; i++) print_decomposition_sdd_node_as_dot(file,nodes[i]);
}


//fills in the nodes of an sdd into the nodes array 
SddNode** collect_all_nodes(SddNode* node, SddNode** nodes) {

  if(node->bit==0) return nodes-1;
  node->bit=0;
  
  *nodes = node;
  SddNode** end = nodes;
  
  if(node->type==DECOMPOSITION) {
    FOR_each_prime_sub_of_node(prime,sub,node,{
	  end=collect_all_nodes(prime,1+end);
	  end=collect_all_nodes(sub,1+end);
	});
  }

  return end;
}


//prints an SDD in .dot file format
void print_sdd_as_dot(FILE* file, SddNode* node) {
  
  fprintf(file,"\ndigraph sdd {");
  fprintf(file,"\n\noverlap=false");
  fprintf(file,"\n");
	
  if(node->type!=DECOMPOSITION) { 
    //a single terminal node
    print_terminal_sdd_node_as_dot(file,node);
  }
  else {
  
    //count sdd nodes
    SddSize count = sdd_all_node_count_leave_bits_1(node); 
    //all nodes are now marked 1

    //put them in an array
  	SddNode** nodes;
  	CALLOC(nodes,SddNode*,count,"print_sdd_as_dot");
  	collect_all_nodes(node,nodes);
  	//all nodes are now marked 0

    //print nodes
    print_sdd_nodes_as_dot(file,count,nodes);
    
  	free(nodes);
  }

  fprintf(file,"\n\n");
  fprintf(file,"\n}");
}

void sdd_save_as_dot(const char* fname, SddNode *node) { 
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"sdd_save_as_dot");
  
  FILE *file = fopen(fname,"w");
  print_sdd_as_dot(file,node);
  fclose(file); 
}


/****************************************************************************************
 * printing sdd to .dot file: all nodes normalized for a vtree node
 ****************************************************************************************/

//prints an SDD in .dot file format
void print_sdds_as_dot(FILE* file, Vtree* vtree) {
  
  fprintf(file,"\ndigraph sdd {");
  fprintf(file,"\n\noverlap=false");
  fprintf(file,"\n");

  if(LEAF(vtree)) {
    print_terminal_sdd_node_as_dot(file,vtree->nodes); //positive literal
    print_terminal_sdd_node_as_dot(file,vtree->nodes->vtree_next); //negative literal
  }
  else {
  
    //count nodes
    SddSize count = 0;
    FOR_each_sdd_node_normalized_for(node,vtree,count += sdd_all_node_count_leave_bits_1(node));
    //all nodes are now marked 1

    //put nodes into an array
    SddNode** nodes;
    CALLOC(nodes,SddNode*,count,"print_sdds_as_dot");
    SddNode** end = nodes-1;
    FOR_each_sdd_node_normalized_for(node,vtree,end = collect_all_nodes(node,end+1));
    //all nodes are now marked 0

    //print nodes
    print_sdd_nodes_as_dot(file,count,nodes);
    
  	free(nodes);
  }

  fprintf(file,"\n\n");
  fprintf(file,"\n}");
}

//saving a multi-rooted sdd (roots are all nodes normalized for vtree)
void save_shared_sdd_as_dot_vt(const char* fname, Vtree* vtree) { 
  FILE *file = fopen(fname,"w");
  print_sdds_as_dot(file,vtree);
  fclose(file); 
}

//saves the multi-rooted sdd of manager
void sdd_shared_save_as_dot(const char* fname, SddManager* manager) {
  save_shared_sdd_as_dot_vt(fname,manager->vtree);
}


/****************************************************************************************
 * printing sdd to .sdd file
 *
 * saved nodes are numbered continguously starting from 0
 ****************************************************************************************/

//used to contiguously id nodes and elements before saving
SddSize node_id_counter;

void print_sdd_header(FILE* file, SddSize count) {
  static const char* header = 
    "c ids of sdd nodes start at 0\n"
    "c sdd nodes appear bottom-up, children before parents\n"
    "c\n"
    "c file syntax:\n"
	"c sdd count-of-sdd-nodes\n"
    "c F id-of-false-sdd-node\n"
    "c T id-of-true-sdd-node\n"
    "c L id-of-literal-sdd-node id-of-vtree literal\n"
    "c D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub}*\n"
    "c\n";
  fprintf(file,"%s",header);
  fprintf(file,"sdd %"PRIsS"\n",count);
}

//index stores new (continguous) ids
void print_sdd_node_file(FILE* file, SddNode* node) {
  
  Vtree* vtree = node->vtree; //NULL for trivial nodes

  if(node->type==TRUE) fprintf(file,"T %"PRIsS"\n",node->index);
  else if(node->type==FALSE) fprintf(file,"F %"PRIsS"\n",node->index);
  else if(node->type==LITERAL) fprintf(file,"L %"PRIsS" %"PRIsS" %"PRIlitS"\n",node->index,vtree->position,LITERAL_OF(node));
  else {//decomposition 
    fprintf(file,"D %"PRIsS" %"PRIsS" %"PRInsS"",node->index,vtree->position,node->size);
	FOR_each_prime_sub_of_node(prime,sub,node,fprintf(file," %"PRIsS" %"PRIsS"",prime->index,sub->index));
    fprintf(file,"\n");     
  }
}

void print_sdd_recurse(FILE* file, SddNode* node) {
  if (node->bit==0) return; //node already visited (i.e., already printed)
  node->bit=0;

  node->index = node_id_counter++; //new id
  if(node->type==DECOMPOSITION) {
    FOR_each_prime_sub_of_node(prime,sub,node,{
      print_sdd_recurse(file,prime);
      print_sdd_recurse(file,sub);
  	});
  }
  print_sdd_node_file(file,node);
}

void print_sdd(FILE* file, SddNode* node) {
  SddSize count = sdd_all_node_count_leave_bits_1(node);
  //all node bits are now set to 1
  print_sdd_header(file,count);
  node_id_counter=0;
  print_sdd_recurse(file,node);
  //all node bits are now set to 0
}

void sdd_save(const char* fname, SddNode *node) { 
  CHECK_ERROR(GC_NODE(node),ERR_MSG_GC,"save_sdd");
  assert(!GC_NODE(node));
  
  FILE *file = fopen(fname,"w");
  print_sdd(file,node);
  fclose(file); 
}


/****************************************************************************************
 * reading sdd
 ****************************************************************************************/


//reads an SDD from a .sdd file
SddNode* sdd_read(const char* filename, SddManager* manager) {
  char* buffer = read_file(filename);
  char* filtered = filter_comments(buffer);
  SddNode* node;
  
  //auto gc and minimize will not be invoked during reading
  WITH_no_auto_mode(manager,{
    node = parse_sdd_file(filtered,manager);
  });
  
  free(buffer);
  free(filtered);
  return node;
}


/*****************************************************************************************
 * ids of sdd nodes start at 0 
 * ids of node are continguous
 * sdd nodes appear bottom-up, children before parents
 * 
 * file syntax:
 * sdd count-of-sdd-nodes
 * F id-of-false-sdd-node
 * T id-of-true-sdd-node
 * L id-of-literal-sdd-node id-of-vtree literal
 * D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub}*
*****************************************************************************************/

SddNode* parse_sdd_file(char* buffer, SddManager* manager) {
  Vtree** pos2vnode_map(Vtree* vtree);

  Vtree* vtree       = manager->vtree;
  Vtree** vtree_list = pos2vnode_map(vtree); //maps ids to vnodes

  // 1st token is "sdd"
  header_strtok(buffer,"sdd");

  // read count of nodes
  SddSize node_count = int_strtok();

  //create map from ids to nodes
  SddNode** node_list;
  CALLOC(node_list,SddNode*,node_count,"parse_sdd_file");
  
  //create buffers for primes/subs
  SddNodeSize max_size = 16;
  SddNode** prime_list;
  CALLOC(prime_list,SddNode*,max_size,"parse_sdd_file");
  SddNode** sub_list;
  CALLOC(sub_list,SddNode*,max_size,"parse_sdd_file");
  
  SddNode* root = NULL;
  while (node_count--) {
    char node_type      = char_strtok();
    SddSize sdd_node_id = int_strtok();

    if(node_type=='T') node_list[sdd_node_id]=manager->true_sdd;
    else if(node_type=='F') node_list[sdd_node_id]=manager->false_sdd;
    else if(node_type=='L') {
      int_strtok(); //id of vtree: not used
	  SddLiteral lit = int_strtok();
	  node_list[sdd_node_id] = sdd_manager_literal(lit,manager);
	}
     else { //(node_type == 'D')
	    Vtree* vnode     = vtree_list[int_strtok()];
        SddNodeSize size = int_strtok();
        
        if(size > max_size) { //make sure prime/sub buffers are large enough
          max_size = size;
          REALLOC(prime_list,SddNode*,max_size,"parse_sdd_file");
          REALLOC(sub_list,SddNode*,max_size,"parse_sdd_file");
        }
        
        //read/store elements and check if structured
        int structured_elements = 1;
        for(SddNodeSize i=0; i<size; i++) {
          SddNode* prime = prime_list[i] = node_list[int_strtok()];
          SddNode* sub   = sub_list[i]   = node_list[int_strtok()];
          structured_elements &= sdd_vtree_is_sub(prime->vtree,vnode->left);
          structured_elements &= TRIVIAL(sub) || sdd_vtree_is_sub(sub->vtree,vnode->right);
        }
        
        SddNode* node;
		if(structured_elements) {
		  GET_node_from_partition(node,vnode,manager,{
			for(SddNodeSize i=0; i<size; i++) {
			  DECLARE_element(prime_list[i],sub_list[i],vnode,manager);
			}
		  });
		}
		else {
      	  node = manager->false_sdd;
		  for(SddNodeSize i=0; i<size; i++) {
	        SddNode* element = sdd_apply(prime_list[i],sub_list[i],CONJOIN,manager);
	        node             = sdd_apply(node,element,DISJOIN,manager);
		  }
		}
		node_list[sdd_node_id] = node;
    }
    
	root = node_list[sdd_node_id];
  }

  free(vtree_list);
  free(node_list);
  free(prime_list);
  free(sub_list);

  return root;
}

/****************************************************************************************
 * end
 ****************************************************************************************/
