/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/memory.c
SddElement* new_elements(SddNodeSize size, SddManager* manager);

//basic/multiply.c
int multiply_decompositions(SddElement* elements1, SddNodeSize size1, SddElement* elements2, SddNodeSize size2, 
 BoolOp op, Vtree* vtree, SddManager* manager, int limited, void fn(SddNode* prime, SddNode* sub, Vtree* vtree, SddManager* manager));

//sdds/apply.c
SddNode* apply(SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager, int limited);

/****************************************************************************************
 *
 * compute the compressed cartesian product of a collection of sets: set1 x set2 x .... x setn
 *
 * --each set is a partition {(p,s)}
 * --the product of two sets {(pi,si)} x {(pj,sj)} is {(pi*pj,si+si) | pi*pj<>false} 
 * --the cartesian product is a partition by definition
 * --it will also be guaranteed to be compressed
 *
 * three stacks are used to manage the process:
 *  --stack1 will contain the current cartesian product
 *  --stack2 will contain the next defined partition
 *  --stack3 is used to multiple stack1 and stack2, compress the result, and then move back to stack1
 *
 * the following functions are meant to assist in this computation:
 *
 * --open_cartesian_product(...) is called first to start the computation of a cartesian product
 * --for each set:
 *   --open_partition(...) is called to declare the beginning of a set
 *   --declare_element_of_partition(p,s) is called to declare a member of a set
 *   --close_partition(...) is called when all members of the set have been declared
 * --close_cartesian_product(...) is called to end the computation of a cartesian product
 *
 * observation: the size of a cartesian product cannot get smaller as a result of 
 * multiplying it with a partition. hence, the size of the cartesian product will be 
 * monotonically increasing as we add another set to the product
 *
 ****************************************************************************************/

//compresses stack3 and puts result in stack1
//this is more efficient than relying on the compression utility in basic/partitions.c
#define COMPRESS_S3_to_S1(manager,limited) {\
  RESET_STACK(cp_stack1,manager);\
  sort_uncompressed_elements(STACK_SIZE(cp_stack3,manager),STACK_START(cp_stack3,manager));\
  /* equal subs are now adjacent */\
  POP_ELM(SddNode* prev_prime,SddNode* prev_sub,cp_stack3,manager);\
  while(!IS_STACK_EMPTY(cp_stack3,manager)) {\
    POP_ELM(SddNode* prime,SddNode* sub,cp_stack3,manager);\
	if(sub==prev_sub) {\
	  /* apply below will not invoke gc or minimization since vtree_search is on */\
	  prev_prime = apply(prime,prev_prime,DISJOIN,manager,limited);\
	  if(prev_prime==NULL) return 0; /* limits exceeded */\
	}\
	else { /* just popped a new element */\
	  PUSH_ELM(prev_prime,prev_sub,cp_stack1,manager);\
      prev_prime = prime;\
	  prev_sub = sub;\
	}\
  }\
  PUSH_ELM(prev_prime,prev_sub,cp_stack1,manager);\
  assert(STACK_SIZE(cp_stack1,manager)==1 ||\
         elements_sorted_and_compressed(STACK_SIZE(cp_stack1,manager),STACK_START(cp_stack1,manager)));\
}

/****************************************************************************************
 * cartesian product
 ****************************************************************************************/

//declare the beginning of a cartesian product
void open_cartesian_product(SddManager* manager) {
  RESET_STACK(cp_stack1,manager);
  //start with a partition on stack1 that has a single element: true.false
  PUSH_ELM(manager->true_sdd,manager->false_sdd,cp_stack1,manager);
}

//declare the end of a cartesian product and return its size and elements
int close_cartesian_product(int compress, SddNodeSize* size, SddElement** elements, Vtree* vtree, SddManager* manager, int limited) {
  //stack1 already compressed (but not sorted)
  if(compress) {
    SWITCH_ELM_STACKS(cp_stack1,cp_stack3,manager); //uncompressed elements in stack3 now
    COMPRESS_S3_to_S1(manager,limited); //compressed elements in stack1 now (may fail returning with 0)
  }
  *size     = STACK_SIZE(cp_stack1,manager);
  *elements = new_elements(*size,manager);
  memcpy(*elements,STACK_START(cp_stack1,manager),*size*sizeof(SddElement));
  assert(*size > 1);
  return 1;
}

/****************************************************************************************
 * partition
 ****************************************************************************************/
 
//declare the beginning of a new set in the cartesian product
void open_partition(SddManager* manager) {
  RESET_STACK(cp_stack2,manager);
}

void declare_element_of_partition(SddNode* prime, SddNode* sub, Vtree* vtree, SddManager* manager) {
  assert(!IS_FALSE(prime));
  assert(IS_TRUE(prime) || sdd_vtree_is_sub(prime->vtree,vtree->left));
  assert(TRIVIAL(sub) || sdd_vtree_is_sub(sub->vtree,vtree->right));
  PUSH_ELM(prime,sub,cp_stack2,manager);
}

static inline
void push_element_to_stack3(SddNode* prime, SddNode* sub, Vtree* vtree, SddManager* manager) {
  //prime could be true
  assert(!IS_FALSE(prime));
  assert(IS_TRUE(prime) || sdd_vtree_is_sub(prime->vtree,vtree->left));
  assert(TRIVIAL(sub) || sdd_vtree_is_sub(sub->vtree,vtree->right));
  PUSH_ELM(prime,sub,cp_stack3,manager);
}
  
//declare the end of a new set in the cartesian product
//multiply the partitions in stack_1 and stack_2, possibly compress, then put result on stack_1
int close_partition(int compress, Vtree* vtree, SddManager* manager, int limited) {
  
  SddElement* elements1 = STACK_START(cp_stack1,manager);
  SddElement* elements2 = STACK_START(cp_stack2,manager);
  SddSize size1         = STACK_SIZE(cp_stack1,manager);
  SddSize size2         = STACK_SIZE(cp_stack2,manager);

  assert(size1 >= 1 && size2 >= 1);

  //multiply elements1 and elements2 and put on stack3
  RESET_STACK(cp_stack3,manager);
  //timeout check embedded into multiply_decompositions
  int success = 
    multiply_decompositions(
      elements1,size1,
      elements2,size2,
      DISJOIN,vtree,manager,limited,
      push_element_to_stack3);
      
  if(!success) return 0; //time limit exceeded
  
  if(0 && compress) { //compress stack3 and put result on stack1 (no trimming is possible)
    COMPRESS_S3_to_S1(manager,limited); //may fail returning with 0
  }
  else { //no compression: just switch stack1 and stack3
    SWITCH_ELM_STACKS(cp_stack1,cp_stack3,manager);
  }
  
  //current cartesian product in stack1
  if(limited && 
     STACK_SIZE(cp_stack1,manager) > manager->vtree_ops.cartesian_product_limit) {
	++manager->vtree_ops.failed_count_cp;
	return 0; //failure
  }
  else return 1; //success
  
}


/****************************************************************************************
 * end
 ****************************************************************************************/
