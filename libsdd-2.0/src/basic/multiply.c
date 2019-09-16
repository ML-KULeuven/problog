/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//basic/apply.c
SddNode* apply(SddNode* node1, SddNode* node2, BoolOp op, SddManager* manager, int limited);

/****************************************************************************************
 * multiplying two decompositions
 *
 * if 1st decomposition has elements pi.si and 2nd decomposition has elements pj.sj,
 * then define prime P = pi CONJOIN pj, sub S = si OP si, and 
 * call code fn for each P.S, where prime P is not false
 *
 * when computing a product of two decompositions: (p11 p12 ... p1n) x (p21 p22 ... p2m)
 * --observation 1: 
 *  if p1i = p2j, then p1k x p2j = false for all k<>i
 * --observation 2:
 * if p1i = !p2j, then p1k x p2j = p1k for all k<>i
 * --observation 3:
 *  if p1i*p2j=p2j, then pik x p2j = false for all k<>i
 *
 * the above observations are used to skip some conjoin operations whose result 
 * can be predicted upfront
 ****************************************************************************************/

//multiplies decompositions elements1/size1 and elements2/size2
//(p1,s1)*(p2,s2) is defined as (p1 and p2,s1 op s2) when p1 and p2 is consistent
//calls function fn on each element of the product

#define SIZE_THR 1024

//returns 1 if time limit is not exceeded, 0 otherwise
//
//assumes that fn has no sides effets that need to be undone upon failure due to time out
//assumes that either:
// --elements1/2 are referenced and fn will end up referencing its prime and sub arguments, or
// --calls to apply inside multiply_decompositions will not invoke minimization/gc
//
//multiply_decompositions is only called in two places, which both satisfy the above assumptions
//
//note: since this is called by multiply cartesian products, it is possible that size1=1 or size2=1
//
int multiply_decompositions(
 
 SddElement* elements1, SddNodeSize size1, 
 SddElement* elements2, SddNodeSize size2, 
 BoolOp op, Vtree* vtree, SddManager* manager, int limited, 
 void fn(SddNode* prime, SddNode* sub, Vtree*, SddManager* manager)) {
 
  //INITIALIZE
  
  SddElement E1[SIZE_THR]; //allocated on stack
  SddElement E2[SIZE_THR]; //allocated on stack
  SddElement* e1 = E1;
  SddElement* e2 = E2;
  SddNode* MS1[SIZE_THR]; //allocated on stack
  SddNode* MS2[SIZE_THR]; //allocated on stack
  SddNode** ms1 = MS1;
  SddNode** ms2 = MS2;
  //when size1 or size2 are large enough, allocate on heap to avoid stack overflow
  if(size1 > SIZE_THR) CALLOC(e1,SddElement,size1,"multiply_decompositions"); //on heap
  if(size2 > SIZE_THR) CALLOC(e2,SddElement,size2,"multiply_decompositions"); //on heap
  if(size1 > SIZE_THR) CALLOC(ms1,SddNode*,size1,"multiply_decompositions"); //on heap
  if(size2 > SIZE_THR) CALLOC(ms2,SddNode*,size2,"multiply_decompositions"); //on heap
  SddElement* e1_start = e1;
  SddElement* e2_start = e2;
  SddElement* e1_end   = e1+size1;
  SddElement* e2_end   = e2+size2;
  SddNode* p1 = NULL;
  SddNode* p2 = NULL;
  SddNode* s1 = NULL;
  SddNode* s2 = NULL;
  
  //PREPROCESS ELEMENTS: identify opportunities for avoiding a full quadratic complexity
  
  //mark primes of elements1
  for(SddElement* i=elements1; i<elements1+size1; i++) i->prime->dit = 1;
  
  //copy elements2 to e2: elements with common primes appearing first in e2
  for(SddElement* i=elements2; i<elements2+size2; i++) {
    i->prime->dit = !(i->prime->dit);
    if(i->prime->dit==0) *e2_start++ = *i; else *(--e2_end) = *i;
    if(i->prime->negation!=NULL && i->prime->negation->dit==1) { p2 = i->prime; s2 = i->sub; }
  }
  
  //copy elements1 to e1: elements with common primes appearing first in e1
  for(SddElement* i=elements1; i<elements1+size1; i++) {
    if(i->prime->dit==0) *e1_start++ = *i; else *(--e1_end) = *i;
    i->prime->dit = 0;
    if(p2!=NULL && i->prime==p2->negation) { p1 = i->prime; s1 = i->sub; }
  }
  //primes of elements1 are now unmarked
  
  // unmark primes of elements2
  for(SddElement* i=elements2; i<elements2+size2; i++) i->prime->dit = 0;
  
  //MULTIPLY
  
  if(p1==NULL) goto compute_product;
  
  //p1 and p2 are complementary: p1 in e1 and p2 in e2
  //multiply has LINEAR complexity in this case
  assert(p2!=NULL);
    
  //iterate over elements of e1 except for p1
  for(SddElement* i=e1; i<e1+size1; i++) {
    if(i->prime==p1) continue;
    SddNode* prime = i->prime;
  	SddNode* sub = apply(i->sub,s2,op,manager,limited);
  	if(sub==NULL) goto failure; //exceeded time limit
  	fn(prime,sub,vtree,manager); //execute code
  }
    
  //iterate over elements of e2 except for p2
  for(SddElement* j=e2; j<e2+size2; j++) {
    if(j->prime==p2) continue;
    SddNode* prime = j->prime;
  	SddNode* sub = apply(j->sub,s1,op,manager,limited);
  	if(sub==NULL) goto failure; //exceeded time limit
  	fn(prime,sub,vtree,manager); //execute code
  }
  
  goto success; //done  
  
  compute_product:
  //must compute products
  
  //A1: multiply common elements (usually, this leads to more added elements than B)
  //the following fragment is quadratic in size1 and size2 (more efficient than A2 when the product is small enough)
  if((e1_end-e1)*(e2_end-e2) <= 64) {
    e2_start = e2; // so it can be changed 
    for(SddElement* i=e1; i<e1_end; i++) {
      for(SddElement* j=e2_start; j<e2_end; j++) {
        if(i->prime==j->prime) {
          SddNode* prime = i->prime;
	      SddNode* sub = apply(i->sub,j->sub,op,manager,limited);
	      if(sub==NULL) goto failure; //exceeded time limit
  	      fn(prime,sub,vtree,manager); //execute code
		  *j = *e2_start++; // exclude *j from further iterations
		  break;
	    }
      }
    }
  }
  //A2: multiply common elements (usually, this leads to more added elements than B)
  //the following fragment is linear in size1 and size2
  else {  
    //save the multiply subs of e1 and e2
    for(SddElement* i=e1; i<e1_end; i++) ms1[i-e1] = i->prime->multiply_sub;
    for(SddElement* i=e2; i<e2_end; i++) ms2[i-e2] = i->prime->multiply_sub; 
    //clear the multiply subs of e1
    for(SddElement* i=e1; i<e1_end; i++) i->prime->multiply_sub = NULL;
    //set the multiply subs of e2
    for(SddElement* i=e2; i<e2_end; i++) i->prime->multiply_sub = i->sub;
    //identify elements of e1 that are common with e2 (have the same prime)
    for(SddElement* i=e1; i<e1_end; i++) {
      if(i->prime->multiply_sub!=NULL) {
        SddNode* prime = i->prime; //an element of e2 has the same prime
	    SddNode* sub = apply(i->sub,prime->multiply_sub,op,manager,limited);
	    if(sub==NULL) { //exceeded time limit
	      //recover the multiply subs of e1 and e2
	      for(SddElement* i=e1; i<e1_end; i++) i->prime->multiply_sub = ms1[i-e1];
          for(SddElement* i=e2; i<e2_end; i++) i->prime->multiply_sub = ms2[i-e2];
	      goto failure;
	    }
  	    fn(prime,sub,vtree,manager); //execute code
      }
    }
    //recover the multiply subs of e1 and e2
    for(SddElement* i=e1; i<e1_end; i++) i->prime->multiply_sub = ms1[i-e1];
    for(SddElement* i=e2; i<e2_end; i++) i->prime->multiply_sub = ms2[i-e2]; 
  }
  
  //B: multiply non-common elements (usually, this leads to smaller added elements than A)
  
  for(SddElement* i=e1_end; i<e1+size1; i++) {
    for(SddElement* j=e2_end; j<e2+size2; j++) {
      SddNode* prime = apply(i->prime,j->prime,CONJOIN,manager,limited);
      if(prime==NULL) goto failure; //exceeded time limit
	  if(!IS_FALSE(prime)) {
	    SddNode* sub = apply(i->sub,j->sub,op,manager,limited);
	    if(sub==NULL) goto failure; //exceeded time limit
  	    fn(prime,sub,vtree,manager); //execute code
	  }
	  if(prime==i->prime) break;
	  if(prime==j->prime) *j = *e2_end++; /* exclude *j from further iterations */
	}
  } 
  
  success:
    if(size1 > SIZE_THR) { free(e1); free(ms1); }
    if(size2 > SIZE_THR) { free(e2); free(ms2); }
    return 1;
  
  failure:
    if(size1 > SIZE_THR) { free(e1); free(ms1); }
    if(size2 > SIZE_THR) { free(e2); free(ms2); }
    return 0;
}


/****************************************************************************************
 * end
 ****************************************************************************************/
