#include <stdlib.h>
void* xyz_malloc(size_t n) { return malloc(n); }
void xyz_free(void* p) { free(p); }

#include <pthread.h>
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
void xyz_lock(pthread_mutex_t* m) { pthread_mutex_lock(m); }
void xyz_unlock(pthread_mutex_t* m) { pthread_mutex_unlock(m); }

#include <setjmp.h>
jmp_buf env;

void xyz_throw(int code) { longjmp(env, code); }
int xyz_try() { return setjmp(env); }

#include <sched.h>
#include <unistd.h>
void xyz_pin_to_core(int core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core,&cpuset);
  pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&cpuset);
}

// hooks for OpenCL / CUDA, later auto-generated from compiler
void xyz_launch_vecadd(float* A,float* B,float* C,int N);

