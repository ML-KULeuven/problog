/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 1.1.1, January 31, 2014
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

/****************************************************************************************
 * this file contains the definitions of parameters that control the behavior of the
 * fnf-to-sdd compiler and the used vtree search algorithm
 ****************************************************************************************/
 
/****************************************************************************************
 * initial vtree used by the fnf-to-sdd compiler
 * other options possible: 
 * --"left": left-linear vtree
 * --"right": right-linear vtree
 * --"vertical": alternates between left-linear and right-linear
 ****************************************************************************************/

//initial vtree to start with
#define INITIAL_VTREE "balanced"

/****************************************************************************************
 * fnf-to-sdd compiler options
 * these options can be passed on using command-line arguments
 ****************************************************************************************/

//ratio of dead node to total nodes before garbage collection
#define GC_THRESHOLD .25

//invoke dynamic vtree search when sdd grows by the following factor since last dynamic vtree search
#define VTREE_SEARCH_THRESHOLD 1.2

//mode of vtree search (0 no, 1 post-compilation, 2 dynamic)
#define VTREE_SEARCH_MODE 2

/****************************************************************************************
 * vtree search algorithm options
 ****************************************************************************************/

//stop dynamic vtree search when percentage reduction in size is below threshold
#define DYNAMIC_VTREE_CONVERGENCE_THRESHOLD 1.0
 
//maximum size of a cartesian product that a limited swap or right rotate is allowed to create
#define CARTESIAN_PRODUCT_LIMIT 1024

//comment this line out to deactivate time limits
#define _TIME_LIMITS

//time limits for dynamic vtree search (measured in terms of the number of applies)
//each 5M applies take around 1sec on a modern computer

#ifdef _TIME_LIMITS
#define TIME_LIMIT_LR  25000000
#define TIME_LIMIT_RR  25000000
#define TIME_LIMIT_SW  25000000
#else
#define TIME_LIMIT_LR  0
#define TIME_LIMIT_RR  0
#define TIME_LIMIT_SW  0
#endif

//comment this line out to deactivate size limits
#define _SIZE_LIMITS

//size limits for dynamic vtree search (measured in terms of size increase)

#ifdef _SIZE_LIMITS
#define SIZE_LIMIT_LR     1.20
#define SIZE_LIMIT_RR     1.20
#define SIZE_LIMIT_SW     1.20
#else
#define SIZE_LIMIT_LR     0
#define SIZE_LIMIT_RR     0
#define SIZE_LIMIT_SW     0
#endif

#endif // PARAMETERS_H_

/****************************************************************************************
 * end
 ****************************************************************************************/
