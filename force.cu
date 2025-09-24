// force.cu — CUDA kernel for apply Force("down", 9.8)

extern "C" __global__
void apply_force(float *pos_y, float *vel_y, float g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vel_y[i] += g;        // update velocity with gravity
        pos_y[i] += vel_y[i]; // integrate position
    }
}

