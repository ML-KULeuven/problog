/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#include "sdd.h"

//util.c
char* ppc(SddSize n);

/****************************************************************************************
 * printing manager stats
 ****************************************************************************************/

//str is a control string for printing size (in mega bytes) of allocated and gc'd nodes/elements
//str must contain four %f control
void print_manager_free_str_memory_MB(const char* str, SddManager* manager) {
  //some elements may have been allocated and/or then freed by rotate or swap
  assert(manager->stats.max_element_count >= manager->sdd_size+manager->gc_element_count);
  float node_memory     = TYPE2MB(manager->gc_node_count,SddNode);
  float element_memory  = TYPE2MB(manager->gc_element_count,SddElement);
  printf(str,node_memory,element_memory,node_memory+element_memory);
}

//str is a control string for printing size (in mega bytes) of allocated nodes/elements
//str must contain four %f control
void print_manager_str_memory_MB(const char* str, SddManager* manager) {
  float node_memory = TYPE2MB(manager->node_count,SddNode);
  float element_memory = TYPE2MB(manager->sdd_size,SddElement);
  printf(str,node_memory,element_memory,node_memory+element_memory);
}

#define DIV(a,b) ((b)==0? 0: ((a)/(b)))

//print manager stats
void sdd_manager_print(SddManager* manager) {

  char* s1; char* s2;
  printf(                           "\nBASIC STATS\n");
  printf(                           " recursed apply counts    \t: %s total (%.3f%% top-level)\n",s1=ppc(manager->stats.apply_count),100.0*manager->stats.apply_count_top/manager->stats.apply_count);
  free(s1);  
  printf(                           " sdd size                 \t: %s live, %s dead\n",s1=ppc(sdd_manager_live_size(manager)),s2=ppc(sdd_manager_dead_size(manager)));
  free(s1); free(s2);  
  printf(                           " sdd count                \t: %s live, %s dead\n",s1=ppc(sdd_manager_live_count(manager)),s2=ppc(sdd_manager_dead_count(manager)));
  free(s1); free(s2);
  printf(                           " largest decomposition    \t: %"PRInsS" compressed, %"PRInsS" uncompressed\n",manager->stats.max_decomposition_size,manager->stats.max_uncompressed_decomposition_size);
  
  SddSize op_count            = manager->vtree_ops.lr_count+manager->vtree_ops.rr_count+manager->vtree_ops.sw_count;
  SddSize failed_time_count   = manager->vtree_ops.failed_lr_count_time+manager->vtree_ops.failed_rr_count_time+manager->vtree_ops.failed_sw_count_time;
  SddSize failed_size_count   = manager->vtree_ops.failed_lr_count_size+manager->vtree_ops.failed_rr_count_size+manager->vtree_ops.failed_sw_count_size;
  SddSize failed_memory_count = manager->vtree_ops.failed_lr_count_memory+manager->vtree_ops.failed_rr_count_memory+manager->vtree_ops.failed_sw_count_memory;
  
  printf(                           "\nVTREE OPERATIONS        \t:         lr         rr         sw\n");
  printf(                           " total                    \t: %10"PRIsS" %10"PRIsS" %10"PRIsS"\n",manager->vtree_ops.lr_count,manager->vtree_ops.rr_count,manager->vtree_ops.sw_count);
  printf(                           " failed (time)            \t: %10"PRIsS" %10"PRIsS" %10"PRIsS"  (%.1f%%)\n",manager->vtree_ops.failed_lr_count_time,manager->vtree_ops.failed_rr_count_time,manager->vtree_ops.failed_sw_count_time,op_count==0?0:((100.0*failed_time_count)/op_count));
  printf(                           " failed (size)            \t: %10"PRIsS" %10"PRIsS" %10"PRIsS"  (%.1f%%)\n",manager->vtree_ops.failed_lr_count_size,manager->vtree_ops.failed_rr_count_size,manager->vtree_ops.failed_sw_count_size,op_count==0?0:((100.0*failed_size_count)/op_count));
  printf(                           " failed (memory)          \t: %10"PRIsS" %10"PRIsS" %10"PRIsS"  (%.1f%%)\n",manager->vtree_ops.failed_lr_count_memory,manager->vtree_ops.failed_rr_count_memory,manager->vtree_ops.failed_sw_count_memory,op_count==0?0:((100.0*failed_memory_count)/op_count));
  printf(                           " failed (cartesian prods) \t: %10"PRIsS"                        (rr+sw)\n",manager->vtree_ops.failed_count_cp);

  printf(                           "\nNODES AND ELEMENTS      \t:         n         e\n");
  print_manager_str_memory_MB(      " memory (allocated)       \t:%10.1f%10.1f  (%.1f MB)\n",manager);
  print_manager_free_str_memory_MB( " memory (free)            \t:%10.1f%10.1f  (%.1f MB)\n",manager);
  
  SddHash* hash = manager->unique_nodes;
  printf(                           "\nHASH TABLES              \t:\n");
  printf(                           " nodes:\n");
  printf(                           "   size                     \t:%10s (%.1f MBs, %.1f%% saturation)\n",s1=ppc(hash->size),TYPE2MB(hash->size,SddComputed*),saturation(hash)); free(s1);
  printf(                           "   hit rate                 \t:%10.1f%%\n",hit_rate(hash));
  printf(                           "   ave lookup cost          \t:%10.1f\n",ave_lookup_cost(hash));
  printf(                           "   increase-size count      \t:%10"PRIsS"\n",hash->increase_size_count);
  printf(                           "   decrease-size count      \t:%10"PRIsS"\n",hash->decrease_size_count);
  printf(                           " computed:\n");
  printf(                           "   size                     \t:%10s (%.1f MBs)\n",s1=ppc(COMPUTED_CACHE_SIZE),TYPE2MB(2*COMPUTED_CACHE_SIZE,SddComputed)); free(s1);
  printf(                           "   hit rate                 \t:%10.1f%%\n",100.0*manager->computed_cache_hit_count/manager->computed_cache_lookup_count);
  printf(                           "   saturation               \t:%10.1f%%\n",100.0*manager->computed_count/(2*COMPUTED_CACHE_SIZE));
 
  SddManagerVtreeOps ops = manager->vtree_ops;
  printf(                           "\nMINIMIZATION OPTIONS:\n");
  printf(                           " time limits              \t:%10.2f secs (apply), %.1f secs (operation), %.1f secs (fragment), %.1f secs (search)\n",(double)ops.apply_time_limit/CLOCKS_PER_SEC,(double)ops.op_time_limit/CLOCKS_PER_SEC,(double)ops.fragment_time_limit/CLOCKS_PER_SEC,(double)ops.search_time_limit/CLOCKS_PER_SEC);
  printf(                           " size limit               \t:%10.1f (min: %d)\n",ops.op_size_limit,VTREE_OP_SIZE_MIN);
  printf(                           " memory limit             \t:%10.1f (min: %.1f MB)\n",ops.op_memory_limit,ops.op_memory_limit>0?VTREE_OP_MEMORY_MIN:0.0);
  printf(                           " cartesian-product limits \t:%10"PRIsS"\n",ops.cartesian_product_limit);
  printf(                           " convergence threshold    \t:%10.1f%%\n",ops.convergence_threshold);

  int icount = manager->auto_search_invocation_count; 
  printf(                           "\nAUTO GC & MINIMIZE      \t:\n");
  printf(                           " gc invocation count      \t:%10d\n",manager->auto_gc_invocation_count);
  printf(                           " search invocation count  \t:%10d (%.1f secs)\n",icount,((float)manager->stats.auto_search_time)/CLOCKS_PER_SEC);
  
  if(icount) {
    printf(                           "   time per search        \t:%10.1f ave secs, %.1f max secs\n",((float)manager->stats.auto_search_time)/CLOCKS_PER_SEC/icount,((float)manager->stats.auto_max_search_time)/CLOCKS_PER_SEC);   
    printf(                           "   triggers               \t:%10.1f%% global, %.1f%% local, %.1f%% recursive\n",100.0*manager->auto_search_invocation_count_global/icount,100.0*manager->auto_search_invocation_count_local/icount,100.0*manager->auto_search_invocation_count_recursive/icount);
    printf(                           "   average iterations     \t:%10.1f\n",((float)manager->auto_search_iteration_count)/icount);
    printf(                           "   average reduction      \t:%10.1f%%\n",manager->auto_search_reduction_sum/icount);
    printf(                           "   aborted searches       \t:%10d apply, %d operation, %d fragment, %d search\n",manager->auto_search_invocation_count_aborted_apply,manager->auto_search_invocation_count_aborted_operation,manager->auto_search_invocation_count_aborted_fragment,manager->auto_search_invocation_count_aborted_search);
  }
  
  printf(                           "\nVTREE FRAGMENTS         \t:\n");
  printf(                           " max shadow memory        \t:%10.1f MB\n",manager->max_fragment_shadow_byte_count/(1024*1024.0));      
  printf(                           "   completed              \t:%8.2f%%\n",100.0*manager->completed_fragment_count/manager->fragment_count);
  printf(                           "   successful             \t:%8.2f%%\n",100.0*manager->successful_fragment_count/manager->fragment_count);
  if(FRAGMENT_SEARCH_BACKWARD) {
  printf(                           "   backward | completed   \t:%8.2f%%\n",100.0*manager->backward_completed_fragment_count/manager->completed_fragment_count);
  printf(                           "   backward | successful  \t:%8.2f%%\n",100.0*manager->backward_successful_fragment_count/manager->successful_fragment_count);
  }
  printf(                           "   successful | completed \t:%8.2f%%\n",100.0*manager->successful_completed_fragment_count/manager->completed_fragment_count);
  if(manager->fragment_count!=manager->completed_fragment_count)
  printf(                           "   successful | not completed\t:%8.2f%%\n",100.0*(manager->successful_fragment_count-manager->successful_completed_fragment_count)/(manager->fragment_count-manager->completed_fragment_count));
  
  #ifdef NDEBUG
  #else
    printf("\nASSERTIONS ON! ASSERTIONS ON! ASSERTIONS ON!\n");
  #endif
}



/****************************************************************************************
 * end
 ****************************************************************************************/
