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
void set_vtree_properties(Vtree* vtree);

/****************************************************************************************
 * copy a vtree 
 ****************************************************************************************/

Vtree* copy_vtree(Vtree* vtree) {  
  Vtree* copy;
  
  if(LEAF(vtree)) copy = new_leaf_vtree(vtree->var);
  else {
    Vtree* left = copy_vtree(vtree->left);
    Vtree* right = copy_vtree(vtree->right);
    copy = new_internal_vtree(left,right);
  }
  
  copy->some_X_constrained_vars = vtree->some_X_constrained_vars;

  return copy;
}

/****************************************************************************************
 * constructing right linear vtree using natural order
 ****************************************************************************************/

//constructs a right linear vtree with variables first_var, first_var+1, ..., last_var
static
Vtree* new_right_linear_vtree_aux(SddLiteral first_var, SddLiteral last_var) {
  Vtree* leaf_vtree = new_leaf_vtree(first_var);
  if(first_var==last_var) return leaf_vtree;
  else {
    Vtree* right = new_right_linear_vtree_aux(first_var+1,last_var);
    return new_internal_vtree(leaf_vtree,right);
  }
}
 
//construct a right linear vtree with variables 1, 2, ..., var_count
static
Vtree* new_right_linear_vtree(SddLiteral var_count) {
  return new_right_linear_vtree_aux(1,var_count);
}
  
/****************************************************************************************
 * constructing left linear vtree using natural order
 ****************************************************************************************/
 
//constructs a left linear vtree with variables first_var, first_var+1, ..., last_var
static
Vtree* new_left_linear_vtree_aux(SddLiteral first_var, SddLiteral last_var) {
  Vtree* leaf_vtree = new_leaf_vtree(last_var);
  if(first_var==last_var) return leaf_vtree;
  else {
    Vtree* left = new_left_linear_vtree_aux(first_var,last_var-1);
    return new_internal_vtree(left,leaf_vtree);
  }
}

//constructs a left linear vtree with variables 1, 2, ..., var_count
static
Vtree* new_left_linear_vtree(SddLiteral var_count) {
  return new_left_linear_vtree_aux(1,var_count);
}

/****************************************************************************************
* constructing vertical vtree using natural order: alternates between left and right linear
****************************************************************************************/

//constructs a vertical vtree (alternates between right-linear and left-linear) with
//variables first_var, first_var+1, ..., last_var
static
Vtree* new_vertical_vtree_aux(SddLiteral first_var, SddLiteral last_var, int is_left) {
  Vtree* leaf_vtree = new_leaf_vtree(is_left? last_var: first_var);
  if(first_var==last_var) return leaf_vtree;
  else if(is_left) {
    Vtree* left = new_vertical_vtree_aux(first_var,last_var-1,0);
    return new_internal_vtree(left,leaf_vtree);
  }
  else {
    Vtree* right = new_vertical_vtree_aux(first_var+1,last_var,1);
    return new_internal_vtree(leaf_vtree,right);
  }
}
 
//constructs a vertical vtree (alternates between right-linear and left-linear) with
//variables 1, 2, ..., var_count
static
Vtree* new_vertical_vtree(SddLiteral var_count) {
  return new_vertical_vtree_aux(1,var_count,0);
}

/****************************************************************************************
* constructing balanced vtree
****************************************************************************************/
 
//constructs a balanced vtree with variables first_var, first_var+1, ..., last_var
static
Vtree* new_balanced_vtree_aux(SddLiteral first_var, SddLiteral last_var) {
  if(first_var==last_var) return new_leaf_vtree(first_var);
  else {
    SddLiteral mid_var = first_var + ((last_var-first_var+1)/2) -1; //integer division
    Vtree* left  = new_balanced_vtree_aux(first_var,mid_var);
    Vtree* right = new_balanced_vtree_aux(mid_var+1,last_var);
    return new_internal_vtree(left,right);
  }
}

//constructs a balanced vtree with variables 1, 2, ..., var_count
static
Vtree* new_balanced_vtree(SddLiteral var_count) {
  return new_balanced_vtree_aux(1,var_count);
}

/****************************************************************************************
* constructing random vtree
****************************************************************************************/

Vtree* new_random_vtree_aux(SddLiteral var_count, SddLiteral* labels, SddLiteral* unused_count) {
  assert(var_count >= 1);
  assert(var_count <= *unused_count);
  if(var_count==1) {
    SddLiteral uc  = *unused_count;
    SddLiteral i   = rand()%uc; //random index
    SddLiteral var = labels[i]; //random label to use next
    labels[i]      = labels[uc-1]; //last label in array overrides used label
    --(*unused_count); //one more used label                                 
    return new_leaf_vtree(var);             
  }
  else {
    assert(var_count >= 2);
    //int balance   = 35+rand()%31; //balance in [35..65]
    //SddLiteral lc = MAX(1,round(var_count*(balance/100.0))); //random balanced
    //SddLiteral lc = var_count/2; //balanced
    SddLiteral lc = 1+rand()%(var_count-1); //random
    assert(lc>0);                                   
    assert(lc<var_count);
    SddLiteral rc = var_count-lc;
    Vtree* left  = new_random_vtree_aux(lc,labels,unused_count);
    Vtree* right = new_random_vtree_aux(rc,labels,unused_count);
    return new_internal_vtree(left,right);                      
  }
}

//constructs a random vtree with variables 1, 2, ..., var_count
Vtree* new_random_vtree(SddLiteral var_count) {
  SddLiteral* labels = (SddLiteral*) calloc(var_count,sizeof(SddLiteral));
  for(SddLiteral i=0; i<var_count; i++) labels[i] = i+1; //assign labels
  SddLiteral unused_count = var_count;
  srand(time(NULL));
  Vtree* vtree = new_random_vtree_aux(var_count,labels,&unused_count);
  free(labels);
  return vtree;
}

/****************************************************************************************
 * constructing vtrees of different types, using natural or given variable order
 ****************************************************************************************/

//returns a vtree with var_count variables
//the vtree dissects the natural variable order 1..var_count
Vtree* sdd_vtree_new(SddLiteral var_count, const char* type) {
  Vtree* vtree = NULL;
  if(strcmp(type,"left")==0) vtree = new_left_linear_vtree(var_count);
  else if(strcmp(type,"right")==0) vtree = new_right_linear_vtree(var_count);
  else if(strcmp(type,"vertical")==0) vtree = new_vertical_vtree(var_count);
  else if(strcmp(type,"balanced")==0) vtree = new_balanced_vtree(var_count);
  else if(strcmp(type,"random")==0) vtree = new_random_vtree(var_count);
  CHECK_ERROR(vtree==NULL,ERR_MSG_VTREE,"new_vtree");
  set_vtree_properties(vtree);
  return vtree;
}

//replaces each variable i in the vtree with the variable var_order[i-1]
static
void replace_var_order_of_vtree(SddLiteral* var_order, Vtree* vtree) {
  if(LEAF(vtree)) vtree->var = var_order[vtree->var-1];
  else {
    replace_var_order_of_vtree(var_order,vtree->left);
    replace_var_order_of_vtree(var_order,vtree->right);
  } 
}

//returns a vtree with var_count variables
//the vtree dissects the variable order var_order[0]...var_order[var_count-1]
Vtree* sdd_vtree_new_with_var_order(SddLiteral var_count, SddLiteral* var_order, const char* type) {
  Vtree* vtree = sdd_vtree_new(var_count,type);
  replace_var_order_of_vtree(var_order,vtree);
  return vtree;
}

/****************************************************************************************
 * constructing X-constrained vtrees of different types, using natural variable order
 *
 * an X-constrained vtree is one in which there is a right-most vtree node that
 * contains all variables not in X. vtree search will maintain this property
 ****************************************************************************************/
 
//returns an X-constrained vtree
//var_count    : number of variables in vtree
//is_X_var     : array of size 1+var_count
//is_X_var[var]: 1 if var belongs to X, 0 otherwise
//is_X_var[0]  : not used
//
//the number of variables in X must be less than var_count (can be 0)
Vtree* sdd_vtree_new_X_constrained(SddLiteral var_count, SddLiteral* is_X_var, const char* type) {

  SddLiteral X_count = 0; //count of variables in X
  for(SddLiteral var=1; var<=var_count; var++) X_count += is_X_var[var];
  
  //X cannot contain all vtree variables
  assert(X_count < var_count);
  
  if(X_count==0) return sdd_vtree_new(var_count,type); //no X variables
  
  SddLiteral XP_count = var_count - X_count; //count of variables in complement of X
  
  //create variable order for vtree, with X vars first, then 0 (dummy), then XP vars
  //variables appear in natural order within each of X and XP
  SddLiteral start_X  = 0;
  SddLiteral start_XP = 1+X_count;
  SddLiteral* var_order = (SddLiteral*) malloc((1+var_count)*sizeof(SddLiteral));
  
  var_order[X_count] = 0; //dummy var
  for(SddLiteral var=1; var<=var_count; var++) {
    if(is_X_var[var]) var_order[start_X++]  = var;
    else              var_order[start_XP++] = var;
  }
  assert(start_X==X_count && start_XP==1+var_count);
   
  Vtree* up_vtree   = sdd_vtree_new_with_var_order(1+X_count,var_order,type);
  Vtree* down_vtree = sdd_vtree_new_with_var_order(XP_count,var_order+X_count+1,type);
  
  free(var_order);

  Vtree* hook = up_vtree;
  while(INTERNAL(hook->right)) hook = hook->right;
  //hook->right now pointing to dummy leaf
  
  sdd_vtree_free(hook->right);
  hook->right = down_vtree;
  down_vtree->parent = hook;
  hook->var_count = hook->left->var_count + hook->right->var_count;
  
  set_vtree_properties(up_vtree);

  //mark all vtree nodes that contain some variables in X 
  FOR_each_vtree_node(v,up_vtree,v->some_X_constrained_vars = 1);
  FOR_each_vtree_node(v,down_vtree,v->some_X_constrained_vars = 0);
  //nodes that only contain variables in XP will be marked 0

  return up_vtree;
}


/****************************************************************************************
 * end
 ****************************************************************************************/
