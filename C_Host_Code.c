// host_force.c
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

int main() {
    // Example arrays
    float pos_y[N], vel_y[N];
    for (int i = 0; i < N; i++) {
        pos_y[i] = 0.0f;
        vel_y[i] = 0.0f;
    }

    // Platform/Device
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, NULL);

    // Load kernel source
    FILE *f = fopen("force.cl", "r");
    fseek(f, 0, SEEK_END); long size = ftell(f); rewind(f);
    char *src = (char*)malloc(size+1);
    fread(src, 1, size, f); src[size] = '\0'; fclose(f);

    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, NULL);
    clBuildProgram(prog, 1, &device, NULL, NULL, NULL);
    cl_kernel krn = clCreateKernel(prog, "apply_force", NULL);

    // Buffers
    cl_mem pos_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float)*N, pos_y, NULL);
    cl_mem vel_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float)*N, vel_y, NULL);

    // Set args
    float g = 9.8f;
    clSetKernelArg(krn, 0, sizeof(cl_mem), &pos_buf);
    clSetKernelArg(krn, 1, sizeof(cl_mem), &vel_buf);
    clSetKernelArg(krn, 2, sizeof(float), &g);
    clSetKernelArg(krn, 3, sizeof(int), &N);

    // Launch
    size_t global = N;
    clEnqueueNDRangeKernel(q, krn, 1, NULL, &global, NULL, 0, NULL, NULL);

    // Read back
    clEnqueueReadBuffer(q, pos_buf, CL_TRUE, 0, sizeof(float)*N, pos_y, 0, NULL, NULL);
    clEnqueueReadBuffer(q, vel_buf, CL_TRUE, 0, sizeof(float)*N, vel_y, 0, NULL, NULL);

    // Check first values
    printf("pos[0] = %f, vel[0] = %f\n", pos_y[0], vel_y[0]);

    // Cleanup
    clReleaseMemObject(pos_buf);
    clReleaseMemObject(vel_buf);
    clReleaseKernel(krn);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    free(src);

    return 0;
}
