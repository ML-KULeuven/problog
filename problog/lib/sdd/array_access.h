#include "sddapi.h"

SddNode* sdd_array_element(SddNode** arr, int element) {
    return arr[element];
}

int sdd_array_int_element(int* arr, int element) {
    return arr[element];
}