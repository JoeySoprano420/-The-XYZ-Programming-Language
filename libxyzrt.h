#ifndef LIBXYZRT_H
#define LIBXYZRT_H

#include <stddef.h>
#include <pthread.h>
#include <setjmp.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Memory ---------- */
void* xyz_malloc(size_t n);
void xyz_free(void* p);

/* ---------- Mutex ---------- */
typedef pthread_mutex_t xyz_mutex_t;
void xyz_mutex_init(xyz_mutex_t* m);
void xyz_mutex_lock(xyz_mutex_t* m);
void xyz_mutex_unlock(xyz_mutex_t* m);
void xyz_mutex_destroy(xyz_mutex_t* m);

/* ---------- Exceptions ---------- */
typedef struct {
    jmp_buf env;
} xyz_try_env;

int xyz_try(xyz_try_env* e);
void xyz_throw(xyz_try_env* e, int code);

/* ---------- Scheduling ---------- */
void xyz_pin_to_core(int core);

/* ---------- GPU Coercion Stubs ---------- */
void xyz_launch_vecadd(float* A,float* B,float* C,int N);

/* ---------- I/O ---------- */
void xyz_print_int(long v);
void xyz_print_str(const char* s);
long xyz_read_int();
void xyz_read_str(char* buf, size_t max);

#ifdef __cplusplus
}
#endif

#endif
