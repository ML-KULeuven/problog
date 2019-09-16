/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//local declarations
static void cache_true_wmcs(Vtree* vtree, WmcManager* wmc_manager);
static SddWmc wmc_of_missing(SddWmc wmc, Vtree* vtree, Vtree* sub_vtree, WmcManager* wmc_manager);
static void update_derivatives_of_missing(SddWmc dr_wmc, Vtree* vtree, Vtree* sub_vtree, WmcManager* wmc_manager);
static void update_derivatives_of_unused(SddWmc drv_wmc, Vtree* vtree, WmcManager* wmc_manager);

/****************************************************************************************
 * log-space: macro utilities
 ****************************************************************************************/

static int log_mode;

#define ZEROW (log_mode? -INFINITY: 0)
#define ONEW (log_mode? 0: 1)
#define IS_ZEROW(A) (A==ZEROW)
#define IS_ONEW(A) (A==ONEW)
//#define NUM_TO_LOG(A) (log(A))
#define MULT(A,B) (log_mode? (A+B): (A*B))
#define ADD(A,B) (log_mode? (IS_ZEROW(A)? B: (IS_ZEROW(B)? A: (A<B? B+log1p(exp(A-B)): A+log1p(exp(B-A))))): (A+B))
#define DIV(A,B) (log_mode? (A-B): (A/B))
#define INC(A,B) A = ADD(A,B)


/****************************************************************************************
 * node, literal and vtree properties: macro utilities
 ****************************************************************************************/
 
//wmc of a node (n is sdd node, m is wmc_manager)
#define WMC(n,m) (m->node_wmcs[n->index])
//derivative of a node (n is sdd node, m is wmc_manager)
#define DRV(n,m) (m->node_derivatives[n->index])
//wmc of true (v is vtree, m is wmc_manager)
#define USED_TRUE_WMC(v,m) (m->used_true_wmcs[v->position])
#define UNUSED_TRUE_WMC(v,m) (m->unused_true_wmcs[v->position])

//increment node derivative
#define INC_NODE_DRV(n,d,m) INC(m->node_derivatives[n->index],d)
//increment literal derivative
#define INC_LIT_DRV(l,d,m) INC(m->literal_derivatives[l],d)


/****************************************************************************************
 * WMC manager properties: macro utilities
 ****************************************************************************************/

//properties of a wmc manager which are inherited from an sdd manager
#define ROOT(m) (m->sdd_manager->vtree)
#define VAR_COUNT(m) (m->sdd_manager->var_count)

/****************************************************************************************
 * interface functions for weighted model count
 ****************************************************************************************/

WmcManager* wmc_manager_new(SddNode* node, int lm, SddManager* manager) {
  WmcManager* wmc_manager;
  MALLOC(wmc_manager,WmcManager,"wmc_manager_new");
  
  log_mode                 = lm; //declare mode
  wmc_manager->log_mode    = lm; //save mode
  wmc_manager->node        = node;
  wmc_manager->sdd_manager = manager;
  
  //nodes are sorted so children appear before parents in the array
  //n->index contains the location of node n in the array
  SddSize node_count; //how many nodes in the sdd
  wmc_manager->nodes      = sdd_topological_sort(node,&node_count);
  wmc_manager->node_count = node_count;

  //save node indices since ->index field of an sdd node is used by many other
  //operations such as conditioning, saving, etc
  CALLOC(wmc_manager->node_indices,SddSize,node_count,"wmc_manager_new");
  for(SddSize i=0; i<node_count; i++) wmc_manager->node_indices[i] = wmc_manager->nodes[i]->index;
  
  //allocate memory for node wmcs and derivatives
  CALLOC(wmc_manager->node_wmcs,SddWmc,node_count,"wmc_manager_new");
  CALLOC(wmc_manager->node_derivatives,SddWmc,node_count,"wmc_manager_new"); 
  
  //allocate memory for literal wmcs and derivatives
  SddLiteral literal_count = 2*manager->var_count +1;  //one extra
  CALLOC(wmc_manager->literal_weights,SddWmc,literal_count,"wmc_manager_new");
  CALLOC(wmc_manager->literal_derivatives,SddWmc,literal_count,"wmc_manager_new"); 
  //initialize literal weights
  for(SddLiteral i=0; i<literal_count; i++) wmc_manager->literal_weights[i] = ONEW;
  
  //for better literal indexing
  wmc_manager->literal_weights     += manager->var_count;
  wmc_manager->literal_derivatives += manager->var_count;
  
  //allocate memory for the wmcs of true (with respect to each vtree node)
  SddLiteral vtree_count = 2*manager->var_count -1;
  CALLOC(wmc_manager->used_true_wmcs,SddWmc,vtree_count,"wmc_manager_new");
  CALLOC(wmc_manager->unused_true_wmcs,SddWmc,vtree_count,"wmc_manager_new");
  
  return wmc_manager;
}


void wmc_manager_free(WmcManager* wmc_manager) {
  free(wmc_manager->nodes);
  free(wmc_manager->node_indices);
  free(wmc_manager->node_wmcs);
  free(wmc_manager->node_derivatives);
  free(wmc_manager->literal_weights-VAR_COUNT(wmc_manager));
  free(wmc_manager->literal_derivatives-VAR_COUNT(wmc_manager));
  free(wmc_manager->used_true_wmcs);
  free(wmc_manager->unused_true_wmcs);
  free(wmc_manager);
}

//returns a zero weight appropriate domain
SddWmc wmc_zero_weight(WmcManager* wmc_manager) {
  log_mode = wmc_manager->log_mode;
  return ZEROW;
}

//returns a one weight in appropriate domain
SddWmc wmc_one_weight(WmcManager* wmc_manager) {
  log_mode = wmc_manager->log_mode;
  return ONEW;
}

//sets the weight of a literal
//literal is an integer <> 0
void wmc_set_literal_weight(const SddLiteral literal, const SddWmc weight, WmcManager* wmc_manager) {
  wmc_manager->literal_weights[literal] = weight;
}

//returns the weight of a literal
//literal is an integer <> 0
SddWmc wmc_literal_weight(const SddLiteral literal, const WmcManager* wmc_manager) {
  return wmc_manager->literal_weights[literal];
}

//returns the derivative of a literal
//literal is an integer <> 0
SddWmc wmc_literal_derivative(const SddLiteral literal, const WmcManager* wmc_manager) {
  return wmc_manager->literal_derivatives[literal];
}

//returns the marginal wmc of a literal
//literal is an integer <> 0
SddWmc wmc_literal_pr(const SddLiteral literal, const WmcManager* wmc_manager) {
  log_mode = wmc_manager->log_mode;
  return DIV(MULT(wmc_manager->literal_derivatives[literal],
                  wmc_manager->literal_weights[literal]),
             wmc_manager->wmc);
}

/****************************************************************************************
 * computing weighted model count and sets derivatives of literals as a side effect
 *
 * literal weights can be set using wmc_set_literal_weight()
 * literal derivaties can be recovered using wmc_literal_derivative()
 *
 ****************************************************************************************/

static inline
void initialize_wmc(WmcManager* wmc_manager) {

  //recover node indices in case they were changed by other operations
  for(SddSize i=0; i<wmc_manager->node_count; i++) {
    wmc_manager->nodes[i]->index = wmc_manager->node_indices[i];
  }
  
  //initialize derivatives
  for(SddSize i=0; i<wmc_manager->node_count; i++) wmc_manager->node_derivatives[i] = ZEROW;
  for(SddLiteral i=1; i<=VAR_COUNT(wmc_manager); i++) {
    wmc_manager->literal_derivatives[i]  = ZEROW;
    wmc_manager->literal_derivatives[-i] = ZEROW;
  }
  
  //declare used/unused variables
  set_sdd_variables(wmc_manager->node,wmc_manager->sdd_manager);
  
  //compute true constants for used/unsused
  cache_true_wmcs(ROOT(wmc_manager),wmc_manager);
}


//NOTE: when node is trivial, we don't know the vtree for which the node is normalized
//we assume that it is normalized for the vtree root (manager->vtree)
SddWmc wmc_propagate(WmcManager* wmc_manager) {
  
  //set mode
  log_mode        = wmc_manager->log_mode;
  SddNode* node   = wmc_manager->node; //root of sdd
  SddNode** nodes = wmc_manager->nodes; //sorted nodes of sdd
  Vtree* root     = ROOT(wmc_manager); 
  
  //INITIALIZE
  initialize_wmc(wmc_manager);
  
  //the following assumes that a trivial node is normalized for the vtree root
  if(node->type==FALSE) return wmc_manager->wmc = ZEROW; //all derivatives are ZEROW
  if(node->type==TRUE) { //all variables are unused
    SddWmc wmc = UNUSED_TRUE_WMC(root,wmc_manager);
    update_derivatives_of_unused(ONEW,root,wmc_manager);
    return wmc_manager->wmc = wmc;
  }
  
  //FIRST PASS
  
  //compute weighted model counts
  SddWmc wmc = ZEROW; //to avoid compiler warning
  SddSize i  = wmc_manager->node_count;
  
  while(i--) { //visit children before parents
    SddNode* n = *nodes++;
    if(n->type==FALSE)        wmc = ZEROW;
    else if(n->type==TRUE)    wmc = ONEW; //trick!
    else if(n->type==LITERAL) wmc = wmc_literal_weight(LITERAL_OF(n),wmc_manager);
    else { //decomposition
      Vtree* left  = n->vtree->left;
      Vtree* right = n->vtree->right;
      wmc = ZEROW;
	  FOR_each_prime_sub_of_node(prime,sub,n,{
	    SddWmc prime_wmc = WMC(prime,wmc_manager);
	    SddWmc sub_wmc   = WMC(sub,wmc_manager);
	    if(!IS_ZEROW(prime_wmc) && !IS_ZEROW(sub_wmc)) {
	      //assuming gaps cannot be ZEROW
	      prime_wmc = wmc_of_missing(prime_wmc,left,prime->vtree,wmc_manager);
	      sub_wmc   = wmc_of_missing(sub_wmc,right,sub->vtree,wmc_manager);
	      assert(!IS_ZEROW(prime_wmc) && !IS_ZEROW(sub_wmc));
	      INC(wmc,MULT(prime_wmc,sub_wmc));
	    }
	  });
    }
    //save weighted model count
    WMC(n,wmc_manager) = wmc;
  }
  
  //wmc of node over used variables 
  SddWmc node_wmc = WMC(node,wmc_manager);
  //wmc of true over unused variables
  SddWmc unused_wmc = UNUSED_TRUE_WMC(root,wmc_manager);
  //wmc of node over all variables
  wmc = MULT(node_wmc,unused_wmc);

  //SECOND PASS
  
  //compute derivatives for unused variables (if any)
  update_derivatives_of_unused(node_wmc,root,wmc_manager);
  
  //compute derivatives for variables inside node->vtree
  i = wmc_manager->node_count;
  DRV(node,wmc_manager) = unused_wmc; //root of sdd
  
  while(i--) { //visit parents before children
    SddNode* n = *(--nodes);
    SddWmc drv = DRV(n,wmc_manager);
    if(IS_ZEROW(drv)) continue; //no update for derivatives
    //ignoring true and false nodes
    //false nodes do not affect derivatives
    //true nodes are handled implicitly by update_derivative_gap
    if(n->type==LITERAL) INC_LIT_DRV(LITERAL_OF(n),drv,wmc_manager);
    else if(n->type==DECOMPOSITION) { //propagate derivative downwards
      Vtree* left  = n->vtree->left;
      Vtree* right = n->vtree->right;
	  FOR_each_prime_sub_of_node(prime,sub,n,{
	    SddWmc prime_wmc     = WMC(prime,wmc_manager);
	    SddWmc sub_wmc       = WMC(sub,wmc_manager);
	    if(!IS_ZEROW(prime_wmc) || !IS_ZEROW(sub_wmc)) { //otherwise, no derivative update
	      SddWmc prime_wmc_gap = wmc_of_missing(ONEW,left,prime->vtree,wmc_manager);
	      SddWmc sub_wmc_gap   = wmc_of_missing(ONEW,right,sub->vtree,wmc_manager);
	      assert(!IS_ZEROW(prime_wmc_gap) && !IS_ZEROW(sub_wmc_gap));
	      SddWmc product = MULT(drv,MULT(prime_wmc_gap,sub_wmc_gap));
	      assert(!IS_ZEROW(product));
	      if(!IS_ZEROW(prime_wmc)) INC_NODE_DRV(sub,MULT(prime_wmc,product),wmc_manager);
	      if(!IS_ZEROW(sub_wmc))   INC_NODE_DRV(prime,MULT(sub_wmc,product),wmc_manager);
	      if(!IS_ZEROW(prime_wmc) && !IS_ZEROW(sub_wmc)) {
	        product = MULT(drv,MULT(prime_wmc,sub_wmc));
	        assert(!IS_ZEROW(product));
	        update_derivatives_of_missing(MULT(product,sub_wmc_gap),left,prime->vtree,wmc_manager);
	        update_derivatives_of_missing(MULT(product,prime_wmc_gap),right,sub->vtree,wmc_manager);
	      }
	    }
	  });
    }
  }
  
  return wmc_manager->wmc = wmc;
}


/****************************************************************************************
 * computing (and caching) wmc of true over used and unused variables
 *
 * a variable is
 *  --USED if it is referenced by the SDD of a WMC manager;
 *  --UNUSED otherwise
 ****************************************************************************************/

static
void cache_true_wmcs(Vtree* vtree, WmcManager* wmc_manager) {
  if(LEAF(vtree)) {
    SddLiteral var = vtree->var;
    SddWmc pw = wmc_literal_weight(var,wmc_manager);
    SddWmc nw = wmc_literal_weight(-var,wmc_manager);
    SddWmc sum = ADD(pw,nw);
    assert(!IS_ZEROW(sum)); 
    if(vtree->all_vars_in_sdd) { //used var
      USED_TRUE_WMC(vtree,wmc_manager)   = sum; 
      UNUSED_TRUE_WMC(vtree,wmc_manager) = ONEW;
    }
    else { //unused var
      USED_TRUE_WMC(vtree,wmc_manager)   = ONEW;
      UNUSED_TRUE_WMC(vtree,wmc_manager) = sum; 
    }
  }
  else {
    cache_true_wmcs(vtree->left,wmc_manager);
    cache_true_wmcs(vtree->right,wmc_manager);
    
    SddWmc l = USED_TRUE_WMC(vtree->left,wmc_manager);
    SddWmc r = USED_TRUE_WMC(vtree->right,wmc_manager);
    USED_TRUE_WMC(vtree,wmc_manager) = MULT(l,r);
    
    l = UNUSED_TRUE_WMC(vtree->left,wmc_manager);
    r = UNUSED_TRUE_WMC(vtree->right,wmc_manager);
    UNUSED_TRUE_WMC(vtree,wmc_manager) = MULT(l,r);
  }
}

/****************************************************************************************
 * computing wmc of true over used variables in vtree, but not in sub-vtree (could be NULL)
 *
 * this is needed to handle 
 * --true subs: depends on where they appear (i.e., in which decomposition node)
 * --primes not normalized for vtree->left: gap between vtree->right and sub->vtree
 * --subs not normalized for vtree->right: gap between vtree->right and sub->vtree
 *
 ****************************************************************************************/
 
static
SddWmc wmc_of_missing(SddWmc wmc, Vtree* vtree, Vtree* sub_vtree, WmcManager* wmc_manager) {
  assert(!IS_ZEROW(wmc));
  
  wmc = MULT(wmc,USED_TRUE_WMC(vtree,wmc_manager));
  if(sub_vtree!=NULL) wmc = DIV(wmc,USED_TRUE_WMC(sub_vtree,wmc_manager));
  
  return wmc;
}

/****************************************************************************************
 * update the derivatives of:
 * --all used variables in vtree but not in sub-vtree (could be NULL) 
 * --all unused variables in vtree
 *
 * the first computation is needed to handle 
 * --true subs: depends on where they appear (i.e., in which decomposition node)
 * --primes not normalized for vtree->left: gap between vtree->right and sub->vtree
 * --subs not normalized for vtree->right: gap between vtree->right and sub->vtree
 *
 ****************************************************************************************/
 
//update the derivates of all USED variables in vtree, but not in sub_vtree (could be NULL)
static
void update_derivatives_of_missing(SddWmc drv_wmc, Vtree* vtree, Vtree* sub_vtree, WmcManager* wmc_manager) {
  assert(!IS_ZEROW(drv_wmc));
  
  if(vtree==sub_vtree || vtree->no_var_in_sdd) return;
  else if(LEAF(vtree)) {
    SddLiteral var = vtree->var; //must be used
    INC_LIT_DRV(var,drv_wmc,wmc_manager);
    INC_LIT_DRV(-var,drv_wmc,wmc_manager);
  }
  else {
    SddWmc l_wmc = MULT(drv_wmc,USED_TRUE_WMC(vtree->left,wmc_manager));
    SddWmc r_wmc = MULT(drv_wmc,USED_TRUE_WMC(vtree->right,wmc_manager));
    
    if(sub_vtree!=NULL && sdd_vtree_is_sub(sub_vtree,vtree)) {
      SddWmc s_wmc = USED_TRUE_WMC(sub_vtree,wmc_manager);
      if(sdd_vtree_is_sub(sub_vtree,vtree->left)) l_wmc = DIV(l_wmc,s_wmc);
      else r_wmc = DIV(r_wmc,s_wmc);
    }
    
    update_derivatives_of_missing(r_wmc,vtree->left,sub_vtree,wmc_manager);
    update_derivatives_of_missing(l_wmc,vtree->right,sub_vtree,wmc_manager);
  }
}

//update the derivates of all UNUSED variables in vtree
static
void update_derivatives_of_unused(SddWmc drv_wmc, Vtree* vtree, WmcManager* wmc_manager) {
  assert(!IS_ZEROW(drv_wmc));
  
  if(vtree->all_vars_in_sdd==0) {
    if(LEAF(vtree)) {
      SddLiteral var = vtree->var; //must be unused
      INC_LIT_DRV(var,drv_wmc,wmc_manager);
      INC_LIT_DRV(-var,drv_wmc,wmc_manager);
    }
    else {
      SddWmc l_wmc = UNUSED_TRUE_WMC(vtree->left,wmc_manager);
      SddWmc r_wmc = UNUSED_TRUE_WMC(vtree->right,wmc_manager);
      update_derivatives_of_unused(MULT(r_wmc,drv_wmc),vtree->left,wmc_manager);
      update_derivatives_of_unused(MULT(l_wmc,drv_wmc),vtree->right,wmc_manager);
    }
  }
}

/****************************************************************************************
 * end
 ****************************************************************************************/
