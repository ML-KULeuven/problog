/****************************************************************************************
 * The Sentential Decision Diagram Package
 * sdd version 2.0, January 8, 2018
 * http://reasoning.cs.ucla.edu/sdd
 ****************************************************************************************/

#ifndef STACKS_H_
#define STACKS_H_

/****************************************************************************************
 * stacks
 *
 * a stack named X is assumed to have three fields:
 *
 * -- X_start:    a pointer to start of stack
 * -- X_top:      a pointer to top of stack (where next element will be added)
 * -- X_capacity: number of elements stack can hold
 *
 ****************************************************************************************/

//check if stack size is equal to its capacity, doubling its capacity if this is the case
//
//T: type of elements on stack 
//stack: name of stack
//manager: manager hosting stack
#define RESIZE_STACK_IF_FULL(T,stack,manager) {\
  if(manager->top_##stack == manager->start_##stack + manager->capacity_##stack) {\
    /* save relative location of current top */\
    SddSize size = manager->top_##stack - manager->start_##stack;\
    /* double capacity of stack */\
    manager->capacity_##stack *= 2;\
    /* reallocate stack with new capacity */\
    REALLOC(manager->start_##stack,T,manager->capacity_##stack,"stack");\
    /* reset top of new stack since realloc may have move the array */\
    manager->top_##stack = manager->start_##stack + size;\
  }\
}

//check whether stack is empty
#define IS_STACK_EMPTY(stack,manager) (manager->top_##stack == manager->start_##stack)
//number of elements currently on stack
#define STACK_SIZE(stack,manager) (manager->top_##stack - manager->start_##stack)
//start of stack
#define STACK_START(stack,manager) manager->start_##stack
//top of stack (where next element will go)
#define STACK_TOP(stack,manager) manager->top_##stack
//empty the stack (make top point to start of stack)
#define RESET_STACK(stack,manager) manager->top_##stack = manager->start_##stack
//remove and return element at top of stack (assumes stack is not empty)
#define POP_STACK(stack,manager) *(--manager->top_##stack)
//return (without removing) element at top of stack (assumes stack is not empty)
#define PEEK_STACK(stack,manager) *(manager->top_##stack-1)

//add an element to the top of the stack
//E: element
//T: its type
#define PUSH_STACK(E,T,stack,manager) {\
  RESIZE_STACK_IF_FULL(T,stack,manager);\
  *(manager->top_##stack)++ = E;\
}

//the following is specific to element stacks

#define POP_ELM(p,s,stack,manager)\
  SddElement* e = --manager->top_##stack;\
  p = e->prime;\
  s = e->sub;\

#define PUSH_ELM(p,s,stack,manager) {\
  RESIZE_STACK_IF_FULL(SddElement,stack,manager);\
  SddElement* e = (manager->top_##stack)++;\
  e->prime = p;\
  e->sub   = s;\
}

#define SWITCH_ELM_STACKS(stack1,stack2,manager) {\
  SWAP(SddElement*,manager->start_##stack1,manager->start_##stack2);\
  SWAP(SddElement*,manager->top_##stack1,manager->top_##stack2);\
  SWAP(SddSize,manager->capacity_##stack1,manager->capacity_##stack2);\
}

#endif // STACKS_H_

/****************************************************************************************
 * end
 ****************************************************************************************/
