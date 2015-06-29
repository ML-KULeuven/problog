#include <setjmp.h>
extern sigjmp_buf exception_buffer;
extern int exception_status;

#define try if ((exception_status = sigsetjmp(exception_buffer,1)) == 0)
#define catch(val) else if (exception_status == val)
#define throw(val) siglongjmp(exception_buffer,val)
#define finally else

#define KeyboardInterrupt     1
#define DivisionByZero 2
#define OutOfMemory    3