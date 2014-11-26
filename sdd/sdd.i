 %module sdd
 %{
 /* Includes the header in the wrapper code */
 #include "include/sddapi.h"
 %}
 
 /* Parse the header file to generate wrappers */
 %include "include/sddapi.h"
