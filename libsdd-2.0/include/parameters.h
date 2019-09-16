/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

/****************************************************************************************
 * garbage collection
 ****************************************************************************************/

//buckets for storing gc'd nodes according to their size
#define GC_BUCKETS_COUNT 4

/****************************************************************************************
 * manager stacks
 ****************************************************************************************/

#define INITIAL_SIZE_ELEMENT_STACK 2048
#define INITIAL_SIZE_COMPRESSION_STACK 2048
#define INITIAL_SIZE_NODE_BUFFER 2048

/****************************************************************************************
 * hash table parameters (for unique nodes)
 ****************************************************************************************/

//default qsize of unique node table
#define INITIAL_SIZE_UNIQUE_NODE_TABLE 0

//number of lookups to trigger check for resize
#define AGE_BEFORE_HASH_RESIZE_CHECK 100
//number of entries over size that would trigger an increase of hash size
#define LOAD_TO_INCREASE_HASH_SIZE 0.80
//number of entries over size that would trigger a decrease of hash size
#define LOAD_TO_DECREASE_HASH_SIZE 0.05

/****************************************************************************************
 * computation cache parameters
 ****************************************************************************************/

//640007, 1280023, 2560021, 5000011
#define COMPUTED_CACHE_SIZE 2560021

/****************************************************************************************
 * vtree search options
 ****************************************************************************************/

//these parameters haver setter functions
#define VTREE_SEARCH_TIME_LIMIT   (clock_t)  180*CLOCKS_PER_SEC
#define VTREE_FRAGMENT_TIME_LIMIT (clock_t)   60*CLOCKS_PER_SEC
#define VTREE_OP_TIME_LIMIT       (clock_t)   30*CLOCKS_PER_SEC
#define VTREE_APPLY_TIME_LIMIT    (clock_t)   10*CLOCKS_PER_SEC
#define VTREE_OP_SIZE_LIMIT       1.2
#define VTREE_OP_MEMORY_LIMIT     3.0
#define INITIAL_CONVERGENCE_THRESHOLD 1.0
#define CARTESIAN_PRODUCT_LIMIT   8*1024

//these parameters do not have setter functions
#define LIMITS_CHECK_FREQUENCY    100
#define VTREE_OP_SIZE_MIN         16
#define VTREE_OP_MEMORY_MIN       100.0

#define FRAGMENT_SEARCH_BACKWARD 0

#endif // PARAMETERS_H_

/****************************************************************************************
 * end
 ****************************************************************************************/
