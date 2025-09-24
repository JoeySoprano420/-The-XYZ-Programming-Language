#include "libxyzrt.h"
#include <stdlib.h>
#include <unistd.h>
#include <sched.h>

/* ---------- Memory ---------- */
void* xyz_malloc(size_t n) { return malloc(n); }
void xyz_free(void* p) { free(p); }

/* ---------- Mutex ---------- */
void xyz_mutex_init(xyz_mutex_t* m) { pthread_mutex_init(m, NULL); }
void xyz_mutex_lock(xyz_mutex_t* m) { pthread_mutex_lock(m); }
void xyz_mutex_unlock(xyz_mutex_t* m) { pthread_mutex_unlock(m); }
void xyz_mutex_destroy(xyz_mutex_t* m) { pthread_mutex_destroy(m); }

/* ---------- Exceptions ---------- */
int xyz_try(xyz_try_env* e) { return setjmp(e->env); }
void xyz_throw(xyz_try_env* e, int code) { longjmp(e->env, code); }

/* ---------- Scheduling ---------- */
void xyz_pin_to_core(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core,&cpuset);
    pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&cpuset);
}

/* ---------- GPU Coercion Stubs ---------- */
void xyz_launch_vecadd(float* A,float* B,float* C,int N) {
    for(int i=0;i<N;i++) {
        A[i] = B[i] + C[i];
    }
}

#include "libxyzrt.h"
#include <stdlib.h>
#include <unistd.h>
#include <sched.h>
#include <string.h>

/* ---------- Memory ---------- */
void* xyz_malloc(size_t n) { return malloc(n); }
void xyz_free(void* p) { free(p); }

/* ---------- Mutex ---------- */
void xyz_mutex_init(xyz_mutex_t* m) { pthread_mutex_init(m, NULL); }
void xyz_mutex_lock(xyz_mutex_t* m) { pthread_mutex_lock(m); }
void xyz_mutex_unlock(xyz_mutex_t* m) { pthread_mutex_unlock(m); }
void xyz_mutex_destroy(xyz_mutex_t* m) { pthread_mutex_destroy(m); }

/* ---------- Exceptions ---------- */
int xyz_try(xyz_try_env* e) { return setjmp(e->env); }
void xyz_throw(xyz_try_env* e, int code) { longjmp(e->env, code); }

/* ---------- Scheduling ---------- */
void xyz_pin_to_core(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core,&cpuset);
    pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&cpuset);
}

/* ---------- GPU Coercion Stubs ---------- */
void xyz_launch_vecadd(float* A,float* B,float* C,int N) {
    for(int i=0;i<N;i++) {
        A[i] = B[i] + C[i];
    }
}

/* ---------- I/O ---------- */
void xyz_print_int(long v) { printf("%ld\n", v); }
void xyz_print_str(const char* s) { printf("%s\n", s); }
long xyz_read_int() { long v; scanf("%ld",&v); return v; }
void xyz_read_str(char* buf, size_t max) { fgets(buf,max,stdin); }

#include "libxyzrt.h"
#include <stdlib.h>
#include <unistd.h>
#include <sched.h>
#include <string.h>

/* ---------- Memory ---------- */
void* xyz_malloc(size_t n) { return malloc(n); }
void xyz_free(void* p) { free(p); }

/* ---------- Mutex ---------- */
void xyz_mutex_init(xyz_mutex_t* m) { pthread_mutex_init(m, NULL); }
void xyz_mutex_lock(xyz_mutex_t* m) { pthread_mutex_lock(m); }
void xyz_mutex_unlock(xyz_mutex_t* m) { pthread_mutex_unlock(m); }
void xyz_mutex_destroy(xyz_mutex_t* m) { pthread_mutex_destroy(m); }

/* ---------- Exceptions ---------- */
int xyz_try(xyz_try_env* e) { return setjmp(e->env); }
void xyz_throw(xyz_try_env* e, int code) { longjmp(e->env, code); }

/* ---------- Scheduling ---------- */
void xyz_pin_to_core(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core,&cpuset);
    pthread_setaffinity_np(pthread_self(),sizeof(cpu_set_t),&cpuset);
}

/* ---------- GPU Coercion Stubs ---------- */
void xyz_launch_vecadd(float* A,float* B,float* C,int N) {
    for(int i=0;i<N;i++) {
        A[i] = B[i] + C[i];
    }
}

/* ---------- I/O ---------- */
void xyz_print_int(long v) { printf("%ld\n", v); }
void xyz_print_str(const char* s) { printf("%s\n", s); }
long xyz_read_int() { long v; scanf("%ld",&v); return v; }
void xyz_read_str(char* buf, size_t max) {
    if(fgets(buf, max, stdin)) {
        size_t len = strlen(buf);
        if(len>0 && buf[len-1]=='\n') buf[len-1]=0;
    }
}

