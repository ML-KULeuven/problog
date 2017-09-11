/****************************************************************************************
 * MIP Search Function
 ****************************************************************************************/

#ifndef MIPSEARCH_H_
#define MIPSEARCH_H_
#include "sddapi.h"

void sdd_manager_set_mip_minimize(SddManager* manager);

void sdd_manager_add_var_after_last_withtype(SddManager* manager, int b);

void sdd_manager_mip_minimize ( SddManager *manager );

#endif
 
